# SLEEF Math Library Benchmark Tool

This directory contains benchmarking tools for measuring the performance of SLEEF's mathematical functions.

## Overview

Two benchmark programs are provided:

1. **`benchmark`** - Benchmarks scalar (non-SIMD) functions
2. **`benchmark_<simd>`** - Benchmarks SIMD vectorized functions for specific architectures

## Building

### Enable Benchmarks in CMake

To build the benchmark tools, enable the `SLEEF_BUILD_BENCH` option when configuring with CMake:

```bash
mkdir build
cd build
cmake -DSLEEF_BUILD_BENCH=ON ..
make
```

The benchmark executables will be placed in `build/bin/`.

### Cross-compilation

When cross-compiling (e.g., for RISC-V), use the appropriate toolchain file:

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-gcc.cmake \
      -DSLEEF_BUILD_BENCH=ON \
      ..
make
```

## Usage

### Scalar Benchmark

The scalar benchmark tests the performance of non-SIMD math functions:

```bash
./bin/benchmark
```

#### Options:
- `-i <iterations>` - Number of iterations per function (default: 100,000,000)
- `-h` - Show help message

#### Benchmark specific categories:
```bash
./bin/benchmark trig      # Trigonometric functions only
./bin/benchmark exp       # Exponential and logarithm functions
./bin/benchmark pow       # Power functions
./bin/benchmark invtrig   # Inverse trigonometric functions
./bin/benchmark hyp       # Hyperbolic functions
./bin/benchmark all       # All benchmarks (default)
```

#### Example:
```bash
# Run with 10 million iterations for trig functions only
./bin/benchmark -i 10000000 trig
```

### SIMD Benchmarks

SIMD-specific benchmarks are built for each supported SIMD extension on your platform:

**x86/x86_64:**
```bash
./bin/benchmark_sse2      # SSE2 (128-bit)
./bin/benchmark_avx2      # AVX2 (256-bit)
./bin/benchmark_avx512f   # AVX-512F (512-bit)
```

**ARM:**
```bash
./bin/benchmark_advsimd   # NEON/AdvSIMD (128-bit)
./bin/benchmark_sve       # SVE (scalable)
```

**RISC-V:**
```bash
./bin/benchmark_rvvm1     # RVV with LMUL=1
./bin/benchmark_rvvm2     # RVV with LMUL=2
```

**PowerPC:**
```bash
./bin/benchmark_vsx       # VSX
```

**s390x:**
```bash
./bin/benchmark_vxe       # VXE
```

#### Options:
- `-i <iterations>` - Number of iterations (default: 10,000,000)
- `-s <size>` - Vector size in elements (default: 1,000,000)
- `-h` - Show help message

#### Example:
```bash
# Benchmark RVV functions with 5M iterations on 500K element vectors
./bin/benchmark_rvvm1 -i 5000000 -s 500000
```

## Output Format

### Scalar Benchmark Output:
```
Sleef_sin_u10                 :     15.234 ns/call  (sum=1.23456e+08)
sin (reference)               :     25.678 ns/call  (sum=1.23456e+08)
```

- **Function name**: SLEEF function being tested
- **ns/call**: Nanoseconds per function call (lower is better)
- **sum**: Checksum to prevent compiler optimization
- **reference**: Standard libm function for comparison

### SIMD Benchmark Output:
```
SIMD Benchmark - RVV (LMUL=1)
Vector length (double): runtime determined
Iterations: 10000000
Vector size: 1000000 elements

Sleef_sindx_u10rvvm1           :      1.234 ns/element  (sum=1.23456e+06)
```

- **ns/element**: Nanoseconds per vector element processed (lower is better)
- Shows throughput when processing large arrays

## Performance Tips

1. **Fix CPU frequency**: Disable CPU frequency scaling for consistent results:
   ```bash
   # Linux example
   sudo cpupower frequency-set -g performance
   ```

2. **Disable turbo boost**: For more consistent measurements:
   ```bash
   # Intel
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   # AMD
   echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
   ```

3. **Run multiple times**: Performance can vary, run several times and take the median

4. **Close other applications**: Minimize background processes

5. **Check thermal throttling**: Ensure CPU doesn't throttle during benchmarks

## Understanding Results

### Accuracy vs Performance Trade-offs

SLEEF provides multiple accuracy levels for some functions:

- **u10**: Max error of 1.0 ULP (Unit in the Last Place) - highest accuracy
- **u35**: Max error of 3.5 ULP - faster, lower accuracy
- **u05**: Max error of 0.5 ULP - ultra-high accuracy (for sqrt)

Generally: u05 < u10 < u35 in terms of speed (u35 is fastest)

### SIMD Speedup

SIMD speedup can be estimated by comparing:
- Scalar benchmark: ns/call
- SIMD benchmark: ns/element × vector_width

For example, if:
- Scalar sin: 15 ns/call
- SIMD sin (256-bit AVX2, 4 doubles): 2 ns/element

Effective SIMD time per call: 2 × 4 = 8 ns for 4 elements
Speedup: 15 × 4 / 8 = 7.5×

## Benchmark Results Format

### CSV Export (future enhancement)

You can redirect output and parse it for further analysis:

```bash
./bin/benchmark > results.txt
```

## Troubleshooting

### Benchmark not building

Make sure you enabled the benchmark option:
```bash
cmake -DSLEEF_BUILD_BENCH=ON ..
```

### SIMD benchmark not available

Check that your compiler supports the SIMD extension. CMake will only build benchmarks for detected SIMD capabilities.

### Results inconsistent

1. Check CPU governor is set to "performance"
2. Verify no thermal throttling is occurring
3. Close background applications
4. Run benchmarks multiple times

### Low performance on emulators

If cross-compiling and running under QEMU, performance will be much slower than native execution. Benchmark numbers are only meaningful on real hardware.

## Adding Custom Benchmarks

To add benchmarks for additional functions, edit `benchmark.c` or `benchmark_simd.c`:

1. Add function test using `BENCHMARK_SCALAR_1ARG` or `BENCHMARK_SIMD_1ARG_D` macros
2. Specify appropriate input ranges for your function
3. Recompile

Example:
```c
BENCHMARK_SCALAR_1ARG(Sleef_erf_u10, erf, -5.0, 5.0, iterations);
```

## License

Copyright © 2010-2025 SLEEF Project, Naoki Shibata and contributors.

Distributed under the Boost Software License, Version 1.0.

