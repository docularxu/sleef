# GELU Activation Kernel using SLEEF

High-performance, vectorized GELU (Gaussian Error Linear Unit) activation functions using SLEEF's math library.

## 📚 Documentation

| File | Description |
|------|-------------|
| **[GELU_QUICKSTART.md](GELU_QUICKSTART.md)** | **Start here!** 5-minute quick start guide |
| [GELU_KERNEL_README.md](GELU_KERNEL_README.md) | Complete technical documentation |
| [GELU_SUMMARY.txt](GELU_SUMMARY.txt) | Overview and reference |

## 🚀 Quick Start

```bash
# Build for your architecture
make -f Makefile.gelu avx2      # x86-64 (Intel/AMD)
make -f Makefile.gelu neon      # ARM (NEON)
make -f Makefile.gelu sve       # ARM (SVE)
make -f Makefile.gelu rvvm1     # RISC-V

# Run tests
./gelu_avx2

# Build shared library for Python
make -f Makefile.gelu libs

# Test from Python
python gelu_example.py
```

## 📁 Files

### Core Implementation
- **`gelu_kernel.c`** - Main SIMD implementation (~900 lines)
- **`gelu_kernel.h`** - C API header

### Examples
- **`gelu_usage_example.c`** - C usage examples
- **`gelu_example.py`** - Python integration and benchmarks

### Build
- **`Makefile.gelu`** - Multi-architecture build system

## 🎯 Features

- ✅ Multi-architecture SIMD: SSE2, AVX2, AVX512, NEON, SVE, RVV
- ✅ Exact (erf-based) and fast (tanh-based) implementations
- ✅ Single and double precision
- ✅ ~1-2 billion elements/second on AVX2
- ✅ Easy C/C++ and Python integration
- ✅ Built-in tests and benchmarks

## 💡 Example Usage

**C:**
```c
#include "gelu_kernel.h"

double *input = aligned_alloc(64, n * sizeof(double));
double *output = aligned_alloc(64, n * sizeof(double));

// Fast tanh approximation
gelu_tanh_double(input, output, n);

// Or exact erf-based version
gelu_exact_double(input, output, n);
```

**Python:**
```python
from gelu_example import SleefGELU

gelu = SleefGELU('./libgelu_avx2.so')
output = gelu.tanh(input_array)  # Fast
output = gelu.exact(input_array) # Exact
```

## 🔧 Requirements

- **SLEEF library** (3.5+): https://sleef.org/
- **GCC** 7+ or **Clang** 8+
- **C99** or later

Install SLEEF:
```bash
# Ubuntu/Debian
sudo apt-get install libsleef-dev

# macOS
brew install sleef

# Or build from source
git clone https://github.com/shibatch/sleef
```

## 📊 Performance

Typical throughput on modern CPUs:

| Architecture | Float (GB/s) | Double (GB/s) | Elements/sec |
|--------------|--------------|---------------|--------------|
| AVX-512 | ~16 | ~12 | 2-4 billion |
| AVX2 | ~9 | ~7 | 1-2 billion |
| NEON | ~6 | ~4 | 500M-1B |

*Tanh approximation, ~20-30% faster than exact version*

## 📖 API

```c
void gelu_exact_double(const double *input, double *output, size_t n);
void gelu_tanh_double(const double *input, double *output, size_t n);
void gelu_exact_float(const float *input, float *output, size_t n);
void gelu_tanh_float(const float *input, float *output, size_t n);
```

All functions are thread-safe and support in-place operation (`input == output`).

## 🧪 Testing

```bash
# Run all tests
make -f Makefile.gelu test

# Run benchmarks
make -f Makefile.gelu benchmark

# Build everything
make -f Makefile.gelu all
```

## 📚 Learn More

- **Quick Start**: [GELU_QUICKSTART.md](GELU_QUICKSTART.md) - Get up and running in 5 minutes
- **Full Docs**: [GELU_KERNEL_README.md](GELU_KERNEL_README.md) - Complete technical documentation
- **C Examples**: [gelu_usage_example.c](gelu_usage_example.c) - Practical C code examples
- **Python Examples**: [gelu_example.py](gelu_example.py) - Python integration guide

## 📄 License

Distributed under the Boost Software License, Version 1.0.

---

**Questions?** Check the documentation files above or visit https://sleef.org/

