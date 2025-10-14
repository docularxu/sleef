# Quick Start Guide - SLEEF Benchmarks

## TL;DR

```bash
# From the sleef.git root directory:
cd src/libm-benchmark
./build_and_run.sh
```

That's it! The script will automatically build SLEEF with benchmarks enabled and run all available benchmarks for your platform.

## Quick Commands

### Build and run all benchmarks
```bash
./build_and_run.sh
```

### Run only scalar benchmarks
```bash
./build_and_run.sh scalar
```

### Run with custom iterations
```bash
./build_and_run.sh -i 50000000
```

### Clean build
```bash
./build_and_run.sh -c
```

### Cross-compile for RISC-V
```bash
./build_and_run.sh -t ../../toolchains/riscv64-gcc.cmake
```

## Manual Build (if you prefer)

```bash
# From sleef.git root
mkdir build
cd build
cmake -DSLEEF_BUILD_BENCH=ON ..
make -j$(nproc)

# Run benchmarks
./bin/benchmark                # Scalar
./bin/benchmark_rvvm1          # RISC-V RVV (LMUL=1)
./bin/benchmark_avx2           # x86 AVX2
# ... etc for your platform
```

## Understanding Output

```
Sleef_sin_u10                 :     15.234 ns/call
sin (reference)               :     25.678 ns/call
```

- **Lower is better**
- SLEEF function vs standard libm reference
- In this example, SLEEF is ~1.7× faster than standard libm

For SIMD benchmarks:
```
Sleef_sindx_u10rvvm1          :      1.234 ns/element
```

- Time per element processed
- Multiply by vector width to compare with scalar

## Performance Tips

### For accurate results:

1. **Fix CPU frequency:**
   ```bash
   # Linux
   sudo cpupower frequency-set -g performance
   ```

2. **Close other apps** - minimize background processes

3. **Run multiple times** - take median result

4. **Watch temperature** - ensure no thermal throttling

## What Gets Benchmarked?

### Scalar benchmark tests:
- Trigonometric: sin, cos, tan (u10 and u35 accuracy)
- Exponential: exp, log, log10, log2, exp2, exp10
- Power: pow, sqrt, cbrt
- Inverse trig: asin, acos, atan, atan2
- Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
- Both double and single precision

### SIMD benchmarks test:
- sin, cos, tan, exp, log, sqrt
- Double and single precision
- Platform-specific vectorization (SSE2/AVX2/AVX512/NEON/SVE/RVV/VSX/VXE)

## Platform Support

Benchmarks will automatically build for:

| Platform | SIMD Extensions |
|----------|-----------------|
| x86_64   | SSE2, AVX2, AVX-512F |
| ARM      | NEON (AdvSIMD), SVE |
| RISC-V   | RVV (LMUL=1,2) |
| PowerPC  | VSX |
| s390x    | VXE |

## Troubleshooting

**"benchmark not found"** → Build with `-DSLEEF_BUILD_BENCH=ON`

**Inconsistent results** → Fix CPU frequency and close apps

**Want to benchmark specific functions?** → Edit benchmark.c and recompile

## Next Steps

See [README.md](README.md) for detailed documentation.

---

Happy benchmarking! 🚀

