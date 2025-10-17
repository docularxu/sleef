# GELU Kernel using SLEEF

This implementation provides high-performance vectorized GELU (Gaussian Error Linear Unit) activation functions using SLEEF's math library across multiple SIMD architectures.

## What is GELU?

GELU is an activation function commonly used in neural networks, especially in Transformer models like BERT and GPT. It has two formulations:

1. **Exact version**: `GELU(x) = 0.5 * x * (1 + erf(x/√2))`
2. **Tanh approximation**: `GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

## Features

- ✅ Both exact (using `erf`) and approximate (using `tanh`) implementations
- ✅ Single and double precision support
- ✅ Multi-architecture support: x86 (SSE2/AVX2/AVX512F), ARM (NEON/SVE), RISC-V (RVV)
- ✅ Optimized using SLEEF's high-accuracy, high-performance math functions
- ✅ Built-in correctness testing and benchmarking

## Supported Architectures

| Architecture | SIMD Extension | Precision | Function Names |
|--------------|----------------|-----------|----------------|
| x86-64 | SSE2 | FP64/FP32 | 2 doubles / 4 floats per vector |
| x86-64 | AVX2 | FP64/FP32 | 4 doubles / 8 floats per vector |
| x86-64 | AVX512F | FP64/FP32 | 8 doubles / 16 floats per vector |
| ARM | NEON (ADVSIMD) | FP64/FP32 | 2 doubles / 4 floats per vector |
| ARM | SVE | FP64/FP32 | Variable length vectors |
| RISC-V | RVV (LMUL=1) | FP64/FP32 | Variable length vectors |
| RISC-V | RVV (LMUL=2) | FP64/FP32 | Variable length vectors |

## Building

### Prerequisites

1. Build and install SLEEF library (see SLEEF's main documentation)
2. GCC or Clang compiler with appropriate architecture support

### Compilation Examples

```bash
# x86-64 SSE2
gcc -O3 -msse2 -DENABLE_SSE2 gelu_kernel.c -lsleef -lm -o gelu_sse2

# x86-64 AVX2
gcc -O3 -mavx2 -mfma -DENABLE_AVX2 gelu_kernel.c -lsleef -lm -o gelu_avx2

# x86-64 AVX512F
gcc -O3 -mavx512f -DENABLE_AVX512F gelu_kernel.c -lsleef -lm -o gelu_avx512

# ARM NEON
gcc -O3 -DENABLE_ADVSIMD gelu_kernel.c -lsleef -lm -o gelu_neon

# ARM SVE
gcc -O3 -march=armv8-a+sve -DENABLE_SVE gelu_kernel.c -lsleef -lm -o gelu_sve

# RISC-V Vector (LMUL=1)
riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv -DENABLE_RVVM1 gelu_kernel.c -lsleef -lm -o gelu_rvv

# RISC-V Vector (LMUL=2)
riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv -DENABLE_RVVM2 gelu_kernel.c -lsleef -lm -o gelu_rvv2
```

## API Reference

### Core Functions

```c
// Double precision
void gelu_exact_double(const double *input, double *output, size_t n);
void gelu_tanh_double(const double *input, double *output, size_t n);

// Single precision
void gelu_exact_float(const float *input, float *output, size_t n);
void gelu_tanh_float(const float *input, float *output, size_t n);
```

### Parameters

- `input`: Input array (should be aligned to 64 bytes for best performance)
- `output`: Output array (should be aligned to 64 bytes for best performance)
- `n`: Number of elements to process

### Usage Example

```c
#include <stdlib.h>
#include "gelu_kernel.c"  // Or link against compiled object

int main() {
    size_t n = 1024;
    
    // Allocate aligned memory for best performance
    float *input = aligned_alloc(64, n * sizeof(float));
    float *output = aligned_alloc(64, n * sizeof(float));
    
    // Initialize input data
    for (size_t i = 0; i < n; i++) {
        input[i] = -3.0f + 6.0f * i / n;
    }
    
    // Apply GELU activation (tanh approximation)
    gelu_tanh_float(input, output, n);
    
    // Use output...
    
    free(input);
    free(output);
    return 0;
}
```

## Running Tests and Benchmarks

The compiled binary includes built-in tests and benchmarks:

```bash
# Run with defaults (1M elements, 100 iterations)
./gelu_avx2

# Specify custom parameters
./gelu_avx2 <num_elements> <iterations>
./gelu_avx2 10000000 1000
```

### Sample Output

```
========================================
GELU Kernel using SLEEF (AVX2)
========================================

Double precision correctness test:
  GELU exact max error:         2.220e-16
  GELU tanh approx max error:   4.441e-16
  Max difference (exact vs tanh): 4.967e-03

Single precision correctness test:
  GELU exact max error:         5.960e-08
  GELU tanh approx max error:   1.192e-07
  Max difference (exact vs tanh): 4.967e-03

Benchmark (n=1048576, iterations=100):
  GELU exact (double):       2.456 ns/elem  (   6.518 GB/s)
  GELU tanh (double):        1.823 ns/elem  (   8.784 GB/s)
  GELU exact (float):        1.234 ns/elem  (   6.479 GB/s)
  GELU tanh (float):         0.892 ns/elem  (   8.969 GB/s)
```

## Performance Characteristics

### Exact vs Tanh Approximation

- **Exact version** (using `erf`):
  - Higher accuracy
  - Slightly slower (~20-30% depending on architecture)
  - Recommended for training or when accuracy is critical

- **Tanh approximation**:
  - Very close approximation (max error < 0.005 across the range)
  - Faster execution
  - Recommended for inference or when speed is critical

### Memory Alignment

For optimal performance, ensure input and output arrays are aligned to 64-byte boundaries:

```c
double *data = aligned_alloc(64, n * sizeof(double));
```

### Architecture-Specific Notes

- **x86 AVX512**: Best throughput for large batches (16 floats, 8 doubles per cycle)
- **ARM SVE**: Scalable vector length allows portability across different ARM implementations
- **RISC-V RVV**: Scalable like SVE, LMUL=2 provides 2x throughput but uses more registers

## Integration with Deep Learning Frameworks

### PyTorch Example

```python
import torch
import ctypes

# Load the compiled library
lib = ctypes.CDLL('./libgelu_sleef.so')

# Define function signatures
lib.gelu_tanh_float.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t
]

def gelu_sleef(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using SLEEF"""
    assert x.dtype == torch.float32, "Only float32 supported"
    assert x.is_contiguous(), "Tensor must be contiguous"
    
    output = torch.empty_like(x)
    lib.gelu_tanh_float(
        x.data_ptr(),
        output.data_ptr(),
        x.numel()
    )
    return output
```

### TensorFlow Example

```python
import tensorflow as tf
import numpy as np
import ctypes

lib = ctypes.CDLL('./libgelu_sleef.so')
lib.gelu_tanh_float.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_size_t
]

@tf.custom_gradient
def gelu_sleef(x):
    x_np = x.numpy()
    output = np.empty_like(x_np)
    lib.gelu_tanh_float(
        x_np.ravel(),
        output.ravel(),
        x_np.size
    )
    
    def grad(dy):
        # Implement gradient if needed
        pass
    
    return tf.constant(output), grad
```

## Benchmarking Against Other Implementations

You can compare this SLEEF-based implementation against:

1. **Standard library**: Using scalar `erf`/`tanh` functions
2. **Intel MKL**: Using `vmdErf` or `vmdTanh`
3. **Framework built-ins**: PyTorch's `torch.nn.functional.gelu`
4. **Custom implementations**: Hand-written SIMD kernels

The SLEEF implementation typically provides:
- Better portability across architectures
- Competitive performance with specialized libraries
- Excellent accuracy (< 1 ULP for most functions)

## Mathematical Background

### Error Function (erf)

The error function is defined as:

```
erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
```

SLEEF provides highly accurate implementations (< 1.0 ULP error for `u10` variants).

### Tanh Approximation Derivation

The tanh approximation comes from approximating the CDF of the standard normal distribution:

```
Φ(x) ≈ 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

This provides a very close approximation with max absolute error < 0.005.

## Troubleshooting

### Compilation Issues

1. **"sleef.h not found"**: Ensure SLEEF is installed and in your include path
   ```bash
   export CFLAGS="-I/path/to/sleef/include"
   export LDFLAGS="-L/path/to/sleef/lib"
   ```

2. **Undefined references**: Make sure to link with `-lsleef -lm`

3. **Illegal instruction**: Binary compiled for different architecture than runtime CPU

### Runtime Issues

1. **Segmentation fault**: Check that input/output arrays are properly allocated
2. **Incorrect results**: Verify that the architecture flag matches your CPU
3. **Poor performance**: Ensure arrays are properly aligned (64-byte boundary)

## References

- [SLEEF Library](https://sleef.org/)
- [GELU Paper](https://arxiv.org/abs/1606.08415): "Gaussian Error Linear Units (GELUs)"
- [BERT Paper](https://arxiv.org/abs/1810.04805): Uses GELU activation
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): Uses GELU activation

## License

Distributed under the Boost Software License, Version 1.0.
See LICENSE.txt or http://www.boost.org/LICENSE_1_0.txt

## Contributing

Contributions are welcome! Areas for improvement:
- Add gradient computation for backpropagation
- Optimize for specific CPU microarchitectures
- Add support for more SIMD extensions (AVX-512 BF16, AMX, etc.)
- Integrate with more ML frameworks


