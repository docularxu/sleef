# GELU Kernel Quick Start Guide

This is a quick reference for getting started with the SLEEF GELU kernel implementation.

## Files Included

| File | Description |
|------|-------------|
| `gelu_kernel.c` | Main implementation with support for multiple architectures |
| `gelu_kernel.h` | C header file with API documentation |
| `gelu_usage_example.c` | C usage examples |
| `gelu_example.py` | Python usage examples and benchmarks |
| `Makefile.gelu` | Makefile for building on different platforms |
| `GELU_KERNEL_README.md` | Complete documentation |
| `GELU_QUICKSTART.md` | This file |

## 5-Minute Quick Start

### Step 1: Build the kernel (x86-64)

```bash
# For AVX2 (most modern x86-64 CPUs)
make -f Makefile.gelu avx2

# Run the built-in test
./gelu_avx2
```

### Step 2: Try the C example

```bash
# Build and run the usage example
gcc -O3 -mavx2 -mfma -DENABLE_AVX2 gelu_kernel.c gelu_usage_example.c -lsleef -lm -o example
./example
```

### Step 3: Build as a shared library

```bash
# Build shared library
gcc -O3 -mavx2 -mfma -DENABLE_AVX2 -fPIC -shared gelu_kernel.c -lsleef -lm -o libgelu_avx2.so

# Test from Python
python gelu_example.py
```

## Platform-Specific Quick Start

### x86-64 (Intel/AMD)

```bash
# Check your CPU features
lscpu | grep -i avx

# Build for your architecture
make -f Makefile.gelu avx2      # Most modern CPUs
make -f Makefile.gelu avx512    # High-end servers
make -f Makefile.gelu sse2      # Older CPUs
```

### ARM (including Apple Silicon)

```bash
# Build for NEON (all ARM64)
make -f Makefile.gelu neon

# Build for SVE (if available)
make -f Makefile.gelu sve
```

### RISC-V

```bash
# Build for RISC-V Vector (requires toolchain)
make -f Makefile.gelu rvvm1
```

## Using in Your Code

### C/C++

```c
#include "gelu_kernel.h"

// Allocate aligned memory
double *input = aligned_alloc(64, n * sizeof(double));
double *output = aligned_alloc(64, n * sizeof(double));

// Apply GELU
gelu_tanh_double(input, output, n);

free(input);
free(output);
```

Compile with:
```bash
gcc -O3 -mavx2 -mfma -DENABLE_AVX2 your_code.c gelu_kernel.c -lsleef -lm -o your_app
```

### Python

```python
from gelu_example import SleefGELU
import numpy as np

# Initialize (automatically finds compiled library)
gelu = SleefGELU('./libgelu_avx2.so')

# Use it
x = np.random.randn(1000).astype(np.float32)
y = gelu.tanh(x)  # Fast approximation
y = gelu.exact(x) # Exact version
```

## Common Use Cases

### 1. Neural Network Inference

```c
// In your forward pass
void forward_layer(double *input, double *output, size_t n) {
    // ... your linear transformation ...
    
    // Apply GELU activation (fast)
    gelu_tanh_double(input, output, n);
}
```

### 2. Batch Processing

```c
// Process large batches efficiently
const size_t batch_size = 128;
const size_t hidden_dim = 768;  // e.g., BERT base

double *activations = aligned_alloc(64, batch_size * hidden_dim * sizeof(double));

// Apply GELU to all activations at once
gelu_tanh_double(activations, activations, batch_size * hidden_dim);
```

### 3. Mixed Precision

```c
// Use single precision for speed
float *input_f = aligned_alloc(64, n * sizeof(float));
float *output_f = aligned_alloc(64, n * sizeof(float));

gelu_tanh_float(input_f, output_f, n);  // ~2x faster than double
```

## Performance Tips

1. **Use aligned memory**: `aligned_alloc(64, size)` for best performance
2. **Choose the right variant**:
   - `gelu_tanh_*`: Faster, good for inference
   - `gelu_exact_*`: More accurate, better for training
3. **Batch operations**: Process multiple elements at once
4. **Use single precision** when possible (2x faster, still accurate)

## Benchmarking Your System

```bash
# Run comprehensive benchmarks
make -f Makefile.gelu all
make -f Makefile.gelu benchmark
```

Expected performance (elements per second):
- AVX512: ~2-4 billion elements/sec (float)
- AVX2: ~1-2 billion elements/sec (float)
- NEON: ~500M-1B elements/sec (float)
- SVE: Varies by vector length

## Troubleshooting

### "sleef.h not found"

```bash
# Install SLEEF first
sudo apt-get install libsleef-dev  # Ubuntu/Debian
brew install sleef                  # macOS

# Or build from source
git clone https://github.com/shibatch/sleef
cd sleef && mkdir build && cd build
cmake .. && make && sudo make install
```

### "Illegal instruction"

You compiled for a newer instruction set than your CPU supports. Try:
```bash
make -f Makefile.gelu sse2  # Most compatible x86-64
```

### Python: "Library not found"

```bash
# Build the shared library first
make -f Makefile.gelu libs

# Or specify the path
python -c "from gelu_example import SleefGELU; g = SleefGELU('./libgelu_avx2.so')"
```

## Next Steps

- Read `GELU_KERNEL_README.md` for complete documentation
- Check `gelu_usage_example.c` for more examples
- Run `gelu_example.py` for Python integration examples
- Integrate into your ML framework

## Quick Reference

| Function | Precision | Method | Speed | Accuracy |
|----------|-----------|--------|-------|----------|
| `gelu_exact_double` | FP64 | erf | Slower | Highest |
| `gelu_tanh_double` | FP64 | tanh | Fast | Very good |
| `gelu_exact_float` | FP32 | erf | Medium | High |
| `gelu_tanh_float` | FP32 | tanh | Fastest | Good |

## Support

For issues or questions:
1. Check the full documentation: `GELU_KERNEL_README.md`
2. Review the examples: `gelu_usage_example.c` and `gelu_example.py`
3. Ensure SLEEF is properly installed
4. Verify your CPU supports the chosen instruction set

---

**Happy computing with SLEEF GELU!** 🚀


