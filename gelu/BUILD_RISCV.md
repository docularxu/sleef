# Building GELU for RISC-V Vector Extension

This guide shows you how to build the RISC-V Vector (RVV) versions of the GELU kernel.

## Quick Start

### Native Build (on RISC-V machine)

If you're **on a RISC-V machine**, just build directly:

```bash
cd gelu/

# Automatic build (detects native RISC-V)
./build_riscv.sh

# Or use make directly
make            # Builds both rvvm1 and rvvm2
make rvvm1      # Build just RVVM1
make rvvm2      # Build just RVVM2

# Build shared libraries
make libs
```

The script auto-detects you're on RISC-V and uses native `gcc`.

### Cross-Compilation (from x86-64/ARM64)

If you're **cross-compiling** from another architecture:

```bash
cd gelu/

# Build with cross-compiler (auto-detected)
./build_riscv.sh

# Or use make cross-compilation targets
make -f Makefile.gelu cross-riscv-m1
make -f Makefile.gelu cross-riscv-m2
```

## Prerequisites

### For Native Builds (on RISC-V machine)

You just need:
- Standard `gcc` with RVV support (GCC 12+ recommended)
- SLEEF library installed

```bash
# Check GCC version
gcc --version

# Check if RVV is supported
gcc -march=rv64gcv -dM -E - < /dev/null | grep __riscv_vector
```

### For Cross-Compilation (from other architectures)

You need a **RISC-V cross-compiler toolchain**:

### Option 1: Install Pre-built Toolchain (Recommended)

#### macOS (Homebrew)

```bash
# Install RISC-V toolchain via Homebrew
brew tap riscv-software-src/riscv
brew install riscv-gnu-toolchain

# Or download from releases
# https://github.com/riscv-collab/riscv-gnu-toolchain/releases
```

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

#### Manual Download

Download pre-built toolchains from:
- **Official releases**: https://github.com/riscv-collab/riscv-gnu-toolchain/releases
- **SiFive**: https://www.sifive.com/software

Extract and add to PATH:
```bash
export PATH=/path/to/riscv/bin:$PATH
```

### Option 2: Build Toolchain from Source

```bash
# Clone the toolchain repository
git clone https://github.com/riscv-collab/riscv-gnu-toolchain
cd riscv-gnu-toolchain

# Configure (this takes a while)
./configure --prefix=/opt/riscv --with-arch=rv64gcv

# Build (this takes even longer - 1-2 hours)
make linux

# Add to PATH
export PATH=/opt/riscv/bin:$PATH
```

## Building the Binaries

### Method 1: Using the Build Script (Easiest)

```bash
cd gelu/
./build_riscv.sh
```

This will:
- Check for RISC-V toolchain
- Build `gelu_rvvm1` (LMUL=1)
- Build `gelu_rvvm2` (LMUL=2)  
- Build shared libraries `libgelu_rvvm1.so` and `libgelu_rvvm2.so`

### Method 2: Using Make

```bash
cd gelu/

# Build RVVM1 (LMUL=1) - uses less registers, more flexible
make -f Makefile.gelu cross-riscv-m1

# Build RVVM2 (LMUL=2) - 2x throughput, uses more registers
make -f Makefile.gelu cross-riscv-m2

# Build both
make -f Makefile.gelu cross-riscv-m1 cross-riscv-m2
```

### Method 3: Manual Compilation

```bash
cd gelu/

# RVVM1
riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv -DENABLE_RVVM1 \
    gelu_kernel.c -lsleef -lm -o gelu_rvvm1

# RVVM2
riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv -DENABLE_RVVM2 \
    gelu_kernel.c -lsleef -lm -o gelu_rvvm2

# Shared library
riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv -DENABLE_RVVM1 \
    -fPIC -shared gelu_kernel.c -lsleef -lm -o libgelu_rvvm1.so
```

## About RVVM1 vs RVVM2

**RVVM1 (LMUL=1)**:
- Uses vector registers at base multiplier
- More flexible register allocation
- Good for most use cases
- Recommended default

**RVVM2 (LMUL=2)**:
- Uses twice as many registers per vector
- ~2x throughput for larger datasets
- Fewer available registers for compiler
- Better for compute-bound workloads

Both support scalable vector lengths (128-bit to 2048-bit).

## Testing on RISC-V Hardware

### Option 1: Native RISC-V System

Copy the binaries to your RISC-V system:

```bash
# On your build machine
scp gelu_rvvm1 gelu_rvvm2 user@riscv-host:~

# On the RISC-V system
./gelu_rvvm1
```

### Option 2: QEMU Emulation

Install QEMU with RISC-V support:

```bash
# macOS
brew install qemu

# Ubuntu/Debian
sudo apt-get install qemu-user qemu-user-static
```

Run with QEMU:

```bash
# Test RVVM1
qemu-riscv64 -cpu rv64,v=true,vlen=256 ./gelu_rvvm1

# Test RVVM2
qemu-riscv64 -cpu rv64,v=true,vlen=256 ./gelu_rvvm2
```

**Note**: QEMU emulation is slow. For benchmarking, use real RISC-V hardware.

### Option 3: RISC-V Development Board

Test on actual hardware:
- **SiFive boards**: HiFive Unmatched, etc.
- **StarFive**: VisionFive 2 (supports RVV 1.0)
- **Milk-V**: Pioneer (supports RVV)
- **Sophgo**: SG2042 boards

## Troubleshooting

### "riscv64-unknown-linux-gnu-gcc: command not found"

You need to install the RISC-V toolchain. See "Prerequisites" above.

### "cannot find -lsleef"

You need SLEEF built for RISC-V. Either:

1. **Cross-compile SLEEF for RISC-V**:
```bash
git clone https://github.com/shibatch/sleef
cd sleef
mkdir build-riscv && cd build-riscv
cmake .. -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc \
         -DCMAKE_SYSTEM_NAME=Linux \
         -DCMAKE_SYSTEM_PROCESSOR=riscv64
make
sudo make install
```

2. **Use static linking** (if SLEEF is built):
```bash
riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv -DENABLE_RVVM1 \
    gelu_kernel.c /path/to/libsleef.a -lm -o gelu_rvvm1
```

3. **Build on the RISC-V target** instead of cross-compiling

### "undefined reference to __riscv_vsetvl_e64m1"

The toolchain doesn't support vector intrinsics. You need:
- GCC 12+ or Clang 14+
- Configured with `--with-arch=rv64gcv`

### Different toolchain prefix

If your toolchain uses a different prefix (e.g., `riscv64-linux-gnu-gcc`), edit the script:

```bash
# In build_riscv.sh, change:
RV_GCC="riscv64-linux-gnu-gcc"
```

Or set it in the Makefile:

```bash
# In Makefile.gelu, line 75:
RV_CC = riscv64-linux-gnu-gcc
```

## Performance Notes

Expected performance on RISC-V hardware (VLEN=256):

| Version | Elements/Vector | Typical Throughput |
|---------|----------------|-------------------|
| RVVM1 FP64 | 4 doubles | ~200-400M elem/sec |
| RVVM1 FP32 | 8 floats | ~400-800M elem/sec |
| RVVM2 FP64 | 8 doubles | ~400-800M elem/sec |
| RVVM2 FP32 | 16 floats | ~800M-1.6B elem/sec |

*Actual performance depends on CPU frequency and implementation*

## Further Reading

- **RISC-V Vector Spec**: https://github.com/riscv/riscv-v-spec
- **RISC-V Vector Intrinsics**: https://github.com/riscv-non-isa/rvv-intrinsic-doc
- **SLEEF Documentation**: https://sleef.org/
- **RISC-V Toolchain**: https://github.com/riscv-collab/riscv-gnu-toolchain

## Need Help?

1. Check you have the RISC-V toolchain: `riscv64-unknown-linux-gnu-gcc --version`
2. Verify SLEEF is installed for RISC-V
3. Try building natively on RISC-V hardware if cross-compilation fails
4. See the main [README.md](README.md) for general help

---

For a working RISC-V system, you can also build directly without cross-compilation:

```bash
# On the RISC-V machine itself
cd gelu/
make    # Will auto-detect RISC-V and build both variants
```

