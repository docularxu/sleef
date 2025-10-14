#!/bin/bash
#   Copyright Naoki Shibata and contributors 2010 - 2025.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

# Script to build and run SLEEF benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLEEF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${SLEEF_ROOT}/build"

usage() {
    echo "Usage: $0 [options] [benchmark_name]"
    echo ""
    echo "Options:"
    echo "  -b, --build-only     Only build, don't run benchmarks"
    echo "  -r, --run-only       Only run benchmarks (skip build)"
    echo "  -c, --clean          Clean build directory first"
    echo "  -t, --toolchain FILE Use specific toolchain file"
    echo "  -i, --iterations N   Number of iterations (default: varies by benchmark)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Benchmark names:"
    echo "  scalar               Run scalar benchmark only"
    echo "  simd                 Run all SIMD benchmarks"
    echo "  all                  Run all benchmarks (default)"
    echo "  <specific>           Run specific SIMD benchmark (e.g., sse2, avx2, rvvm1)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Build and run all benchmarks"
    echo "  $0 -c                                 # Clean build and run all"
    echo "  $0 scalar                             # Run scalar benchmark only"
    echo "  $0 rvvm1                              # Run RVV LMUL=1 benchmark only"
    echo "  $0 -t ../toolchains/riscv64-gcc.cmake # Cross-compile for RISC-V"
    echo "  $0 -i 50000000 simd                   # Run SIMD benchmarks with 50M iterations"
}

# Default options
BUILD_ONLY=0
RUN_ONLY=0
CLEAN=0
TOOLCHAIN=""
BENCHMARK_TARGET="all"
ITERATIONS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--build-only)
            BUILD_ONLY=1
            shift
            ;;
        -r|--run-only)
            RUN_ONLY=1
            shift
            ;;
        -c|--clean)
            CLEAN=1
            shift
            ;;
        -t|--toolchain)
            TOOLCHAIN="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        scalar|simd|all|sse2|avx2|avx512f|advsimd|sve|rvvm1|rvvm2|vsx|vxe)
            BENCHMARK_TARGET="$1"
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Check if we should build
if [[ $RUN_ONLY -eq 0 ]]; then
    echo "=========================================="
    echo "Building SLEEF with benchmarks"
    echo "=========================================="
    
    if [[ $CLEAN -eq 1 && -d "$BUILD_DIR" ]]; then
        echo "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    CMAKE_ARGS="-DSLEEF_BUILD_BENCH=ON"
    if [[ -n "$TOOLCHAIN" ]]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN"
        echo "Using toolchain: $TOOLCHAIN"
    fi
    
    echo "Configuring CMake..."
    cmake $CMAKE_ARGS ..
    
    echo "Building..."
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    echo "Build complete!"
    echo ""
fi

# Check if we should run
if [[ $BUILD_ONLY -eq 1 ]]; then
    echo "Build-only mode. Exiting."
    exit 0
fi

# Prepare iteration arguments
ITER_ARGS=""
if [[ -n "$ITERATIONS" ]]; then
    ITER_ARGS="-i $ITERATIONS"
fi

echo "=========================================="
echo "Running benchmarks"
echo "=========================================="
echo ""

BIN_DIR="${BUILD_DIR}/bin"

if [[ ! -d "$BIN_DIR" ]]; then
    echo "Error: Build directory not found. Please build first."
    exit 1
fi

# Function to run a benchmark if it exists
run_benchmark() {
    local bench_name="$1"
    local bench_path="${BIN_DIR}/${bench_name}"
    
    if [[ -f "$bench_path" ]]; then
        echo "------------------------------------------"
        echo "Running: $bench_name"
        echo "------------------------------------------"
        "$bench_path" $ITER_ARGS
        echo ""
        return 0
    else
        return 1
    fi
}

# Run benchmarks based on target
case $BENCHMARK_TARGET in
    scalar)
        if ! run_benchmark "benchmark"; then
            echo "Error: scalar benchmark not found"
            exit 1
        fi
        ;;
    
    simd)
        found=0
        for bench in benchmark_sse2 benchmark_avx2 benchmark_avx512f \
                     benchmark_advsimd benchmark_sve \
                     benchmark_rvvm1 benchmark_rvvm2 \
                     benchmark_vsx benchmark_vxe; do
            if run_benchmark "$bench"; then
                found=1
            fi
        done
        if [[ $found -eq 0 ]]; then
            echo "Error: No SIMD benchmarks found"
            exit 1
        fi
        ;;
    
    all)
        run_benchmark "benchmark" || echo "Scalar benchmark not available"
        for bench in benchmark_sse2 benchmark_avx2 benchmark_avx512f \
                     benchmark_advsimd benchmark_sve \
                     benchmark_rvvm1 benchmark_rvvm2 \
                     benchmark_vsx benchmark_vxe; do
            run_benchmark "$bench" || true
        done
        ;;
    
    *)
        # Try to run specific benchmark
        if ! run_benchmark "benchmark_${BENCHMARK_TARGET}"; then
            echo "Error: Benchmark 'benchmark_${BENCHMARK_TARGET}' not found"
            echo "Available benchmarks in ${BIN_DIR}:"
            ls -1 "${BIN_DIR}"/benchmark* 2>/dev/null || echo "  (none)"
            exit 1
        fi
        ;;
esac

echo "=========================================="
echo "Benchmark run complete!"
echo "=========================================="

