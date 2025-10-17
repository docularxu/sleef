#!/bin/bash
# Build RISC-V Vector versions of GELU kernel
# Supports both native compilation and cross-compilation
#
# Usage:
#   ./build_riscv.sh              # Auto-detect (native or cross-compile)
#   FORCE_CROSS=1 ./build_riscv.sh  # Force cross-compilation even on RISC-V

set -e

echo "=========================================="
echo "RISC-V Vector GELU Kernel Builder"
echo "=========================================="
echo ""

# Detect actual hardware architecture (not from user environment)
ACTUAL_ARCH=$(uname -m)

# Allow forcing cross-compilation with FORCE_CROSS=1
if [ "${FORCE_CROSS}" = "1" ]; then
    echo "⚠ FORCE_CROSS set - forcing cross-compilation mode"
    ARCH="forced-cross"
else
    ARCH="$ACTUAL_ARCH"
fi

echo "Detected hardware: $ACTUAL_ARCH"

if [ "$ARCH" = "riscv64" ]; then
    # Native compilation on RISC-V machine
    RV_GCC="gcc"
    echo "✓ Detected native RISC-V machine"
    echo "✓ Using native GCC: $RV_GCC"
else
    # Cross-compilation from another architecture
    echo "✓ Detected $ARCH architecture - using cross-compiler"
    
    # Check for RISC-V cross-compiler
    if command -v riscv64-unknown-linux-gnu-gcc &> /dev/null; then
        RV_GCC="riscv64-unknown-linux-gnu-gcc"
        echo "✓ Found RISC-V cross-compiler: $RV_GCC"
    elif command -v riscv64-linux-gnu-gcc &> /dev/null; then
        RV_GCC="riscv64-linux-gnu-gcc"
        echo "✓ Found RISC-V cross-compiler: $RV_GCC"
    else
        echo "✗ RISC-V cross-compiler not found!"
        echo ""
        echo "Please install the RISC-V GNU toolchain:"
        echo ""
        echo "Option 1: Download pre-built toolchain"
        echo "  https://github.com/riscv-collab/riscv-gnu-toolchain/releases"
        echo ""
        echo "Option 2: Install via package manager"
        echo "  # Ubuntu/Debian:"
        echo "  sudo apt-get install gcc-riscv64-linux-gnu"
        echo ""
        echo "  # macOS (via Homebrew):"
        echo "  brew tap riscv/riscv"
        echo "  brew install riscv-tools"
        echo ""
        echo "Option 3: Build from source"
        echo "  git clone https://github.com/riscv-collab/riscv-gnu-toolchain"
        echo "  cd riscv-gnu-toolchain"
        echo "  ./configure --prefix=/opt/riscv"
        echo "  make linux"
        echo ""
        exit 1
    fi
fi

# Check RISC-V GCC version
echo ""
$RV_GCC --version | head -1

# Check for SLEEF library and set paths
echo ""
SLEEF_FOUND=""
SLEEF_INCLUDE=""
SLEEF_LIBDIR=""

# Check common installation locations
if [ -f "$HOME/.local/include/sleef.h" ]; then
    echo "✓ Found SLEEF in: $HOME/.local"
    SLEEF_INCLUDE="$HOME/.local/include"
    SLEEF_LIBDIR="$HOME/.local/lib"
    SLEEF_FOUND="yes"
elif [ -f "/usr/local/include/sleef.h" ]; then
    echo "✓ Found SLEEF in: /usr/local"
    SLEEF_INCLUDE="/usr/local/include"
    SLEEF_LIBDIR="/usr/local/lib"
    SLEEF_FOUND="yes"
elif [ -f "/usr/include/sleef.h" ]; then
    echo "✓ Found SLEEF in: /usr"
    SLEEF_INCLUDE="/usr/include"
    SLEEF_LIBDIR="/usr/lib"
    SLEEF_FOUND="yes"
else
    echo "⚠ Warning: SLEEF headers not found in standard locations"
    echo "  Searched: $HOME/.local, /usr/local, /usr"
    echo "  You may need to set SLEEF_PREFIX=/path/to/sleef"
fi

# Build options (can be overridden with environment variables)
CFLAGS="${CFLAGS:--O3 -Wall -Wextra}"
LDFLAGS="${LDFLAGS:--lsleef -lm}"

# Add SLEEF paths if found
if [ -n "$SLEEF_FOUND" ]; then
    CFLAGS="$CFLAGS -I$SLEEF_INCLUDE"
    LDFLAGS="-L$SLEEF_LIBDIR $LDFLAGS"
    
    # For runtime (especially when using shared libraries)
    if [ -z "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$SLEEF_LIBDIR"
    else
        export LD_LIBRARY_PATH="$SLEEF_LIBDIR:$LD_LIBRARY_PATH"
    fi
    echo "  Include: $SLEEF_INCLUDE"
    echo "  Library: $SLEEF_LIBDIR"
fi

# Allow manual override with SLEEF_PREFIX
if [ -n "$SLEEF_PREFIX" ]; then
    echo "⚙  Using SLEEF_PREFIX=$SLEEF_PREFIX"
    CFLAGS="$CFLAGS -I$SLEEF_PREFIX/include"
    LDFLAGS="-L$SLEEF_PREFIX/lib $LDFLAGS"
    export LD_LIBRARY_PATH="$SLEEF_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# RISC-V Vector specific options
# Customize these based on your hardware by setting environment variables:
#
# Common RISC-V ISA extensions you might want to add to -march:
#   _zba, _zbb, _zbs    - Bit manipulation (address gen, basic, single bit)
#   _zbc                - Carryless multiplication (for optimized operations)
#   _zvbb               - Vector bit manipulation
#   _zfh, _zfhmin       - Scalar half-precision float (FP16) support
#   _zvfh, _zvfhmin     - Vector half-precision float (for future FP16 GELU)
#   _zvl128b, _zvl256b  - Minimum vector length guarantees
#   _zkt, _zvkt         - Crypto extensions (scalar and vector)
#
# Examples:
#   RVV_VLEN=256 ./build_riscv.sh                    # Force 256-bit vectors
#   RVV_MAX_LMUL=2 ./build_riscv.sh                  # Limit to LMUL=2
#   MARCH="rv64gcv_zba_zbb" ./build_riscv.sh         # Custom march string
#   MARCH_PRESET="jupiter" ./build_riscv.sh          # Use preset (jupiter, spacemit, sophgo, etc.)
#   EXTRA_CFLAGS="-g" ./build_riscv.sh               # Add debug symbols

# Base -march string (can be fully overridden)
if [ -z "$MARCH" ]; then
    # Check for preset configurations
    case "$MARCH_PRESET" in
        "spacemit"|"k1")
            # SpacemiT K1 / BananaPi F3 (X60 CPU)
            # zba/zbb/zbs: May help with address calc and loop control
            # zfh/zvfh: Future FP16 support
            MARCH="rv64gcv_zba_zbb_zbs_zfh_zvfh"
            echo "⚙  Using SpacemiT K1 preset (X60 optimized)"
            ;;
        "jupiter"|"milkv-jupiter")
            # Milk-V Jupiter (SpacemiT K1 X60)
            # zba/zbb/zbs: May help with address calc and loop control
            # zfh/zvfh: Future FP16 support
            MARCH="rv64gcv_zba_zbb_zbs_zfh_zvfh"
            echo "⚙  Using Milk-V Jupiter preset (X60 optimized)"
            ;;
        "starfive2"|"visionfive2")
            # StarFive VisionFive 2
            MARCH="rv64gcv"
            echo "⚙  Using VisionFive 2 preset"
            ;;
        "sophgo"|"sg2042")
            # Sophgo SG2042
            # zba/zbb/zbs: May help with address calc and loop control
            MARCH="rv64gcv_zba_zbb_zbs_zfh_zvfh"
            echo "⚙  Using Sophgo SG2042 preset"
            ;;
        "sifive")
            # SiFive boards
            MARCH="rv64gcv"
            echo "⚙  Using SiFive preset"
            ;;
        *)
            # Default: rv64gc + vector
            MARCH="rv64gcv"
            ;;
    esac
fi

echo "⚙  Using -march=$MARCH"

# Vector length specification
if [ -n "$RVV_VLEN" ]; then
    echo "⚙  Setting vector length: VLEN=$RVV_VLEN"
    CFLAGS="$CFLAGS -mrvv-vector-bits=zvl${RVV_VLEN}b"
fi

# LMUL limit
if [ -n "$RVV_MAX_LMUL" ]; then
    echo "⚙  Setting max LMUL: $RVV_MAX_LMUL"
    CFLAGS="$CFLAGS -mrvv-max-lmul=$RVV_MAX_LMUL"
fi

# Additional optimizations for RVV
CFLAGS="$CFLAGS -ftree-vectorize"             # Enable auto-vectorization
CFLAGS="$CFLAGS -fno-math-errno"              # Math operations don't set errno

# Allow extra custom flags
if [ -n "$EXTRA_CFLAGS" ]; then
    echo "⚙  Adding extra CFLAGS: $EXTRA_CFLAGS"
    CFLAGS="$CFLAGS $EXTRA_CFLAGS"
fi

# Display build configuration
echo ""
echo "Build configuration:"
echo "  Compiler: $RV_GCC"
echo "  CFLAGS:   $CFLAGS"
echo "  LDFLAGS:  $LDFLAGS"
echo ""

echo ""
echo "Building RISC-V Vector versions..."
echo "===================================="

# Build RVVM1 (LMUL=1)
echo ""
echo "Building gelu_rvvm1 (LMUL=1)..."
$RV_GCC $CFLAGS -march=$MARCH -DENABLE_RVVM1 gelu_kernel.c $LDFLAGS -o gelu_rvvm1
if [ $? -eq 0 ]; then
    echo "✓ Successfully built gelu_rvvm1"
    ls -lh gelu_rvvm1
else
    echo "✗ Failed to build gelu_rvvm1"
    exit 1
fi

# Build RVVM2 (LMUL=2)
echo ""
echo "Building gelu_rvvm2 (LMUL=2)..."
$RV_GCC $CFLAGS -march=$MARCH -DENABLE_RVVM2 gelu_kernel.c $LDFLAGS -o gelu_rvvm2
if [ $? -eq 0 ]; then
    echo "✓ Successfully built gelu_rvvm2"
    ls -lh gelu_rvvm2
else
    echo "✗ Failed to build gelu_rvvm2"
    exit 1
fi

# Build shared libraries
echo ""
echo "Building shared libraries..."
$RV_GCC $CFLAGS -march=$MARCH -DENABLE_RVVM1 -fPIC -shared gelu_kernel.c $LDFLAGS -o libgelu_rvvm1.so
$RV_GCC $CFLAGS -march=$MARCH -DENABLE_RVVM2 -fPIC -shared gelu_kernel.c $LDFLAGS -o libgelu_rvvm2.so

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Created files:"
ls -lh gelu_rvvm* libgelu_rvvm*.so 2>/dev/null || true

echo ""
if [ "$ARCH" = "riscv64" ]; then
    echo "To run (native RISC-V):"
    if [ -n "$SLEEF_LIBDIR" ] && [ "$SLEEF_LIBDIR" != "/usr/lib" ]; then
        echo "  # Make sure SLEEF library is in your path:"
        echo "  export LD_LIBRARY_PATH=$SLEEF_LIBDIR:\$LD_LIBRARY_PATH"
        echo ""
    fi
    echo "  ./gelu_rvvm1        # Test RVVM1"
    echo "  ./gelu_rvvm2        # Test RVVM2"
else
    echo "To run on RISC-V hardware:"
    echo "  1. Copy binaries to RISC-V system"
    echo "  2. Ensure SLEEF library is installed"
    if [ -n "$SLEEF_LIBDIR" ] && [ "$SLEEF_LIBDIR" != "/usr/lib" ]; then
        echo "  3. Set: export LD_LIBRARY_PATH=$SLEEF_LIBDIR:\$LD_LIBRARY_PATH"
        echo "  4. Run: ./gelu_rvvm1"
    else
        echo "  3. Run: ./gelu_rvvm1"
    fi
    echo ""
    echo "To test with QEMU:"
    echo "  qemu-riscv64 -cpu rv64,v=true ./gelu_rvvm1"
fi
echo ""

