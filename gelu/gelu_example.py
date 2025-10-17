#!/usr/bin/env python3
"""
Example of using the SLEEF GELU kernel from Python

This script demonstrates how to call the compiled GELU kernel from Python
using ctypes and compare it with standard implementations.

Requirements:
  - numpy
  - scipy (optional, for reference implementation)
  
Usage:
  python gelu_example.py
"""

import ctypes
import numpy as np
import time
from pathlib import Path

# Try to import scipy for reference implementation
try:
    from scipy.special import erf
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using numpy approximation")


class SleefGELU:
    """Wrapper for SLEEF GELU kernel"""
    
    def __init__(self, lib_path='./libgelu_avx2.so'):
        """
        Initialize the SLEEF GELU wrapper
        
        Args:
            lib_path: Path to the compiled SLEEF GELU shared library
        """
        lib_path = Path(lib_path)
        if not lib_path.exists():
            raise FileNotFoundError(
                f"SLEEF GELU library not found at {lib_path}\n"
                f"Please compile it first:\n"
                f"  gcc -O3 -mavx2 -mfma -DENABLE_AVX2 -fPIC -shared "
                f"gelu_kernel.c -lsleef -lm -o {lib_path}"
            )
        
        self.lib = ctypes.CDLL(str(lib_path))
        
        # Define function signatures for double precision
        self.lib.gelu_exact_double.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self.lib.gelu_exact_double.restype = None
        
        self.lib.gelu_tanh_double.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self.lib.gelu_tanh_double.restype = None
        
        # Define function signatures for single precision
        self.lib.gelu_exact_float.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self.lib.gelu_exact_float.restype = None
        
        self.lib.gelu_tanh_float.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]
        self.lib.gelu_tanh_float.restype = None
    
    def exact(self, x, out=None):
        """
        Apply exact GELU using error function
        
        Args:
            x: Input array (numpy array, float32 or float64)
            out: Optional output array
            
        Returns:
            GELU(x) = 0.5 * x * (1 + erf(x/√2))
        """
        x = np.ascontiguousarray(x)
        if out is None:
            out = np.empty_like(x)
        else:
            out = np.ascontiguousarray(out)
        
        if x.dtype == np.float64:
            self.lib.gelu_exact_double(x.ravel(), out.ravel(), x.size)
        elif x.dtype == np.float32:
            self.lib.gelu_exact_float(x.ravel(), out.ravel(), x.size)
        else:
            raise ValueError(f"Unsupported dtype: {x.dtype}")
        
        return out.reshape(x.shape)
    
    def tanh(self, x, out=None):
        """
        Apply GELU using tanh approximation
        
        Args:
            x: Input array (numpy array, float32 or float64)
            out: Optional output array
            
        Returns:
            GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        """
        x = np.ascontiguousarray(x)
        if out is None:
            out = np.empty_like(x)
        else:
            out = np.ascontiguousarray(out)
        
        if x.dtype == np.float64:
            self.lib.gelu_tanh_double(x.ravel(), out.ravel(), x.size)
        elif x.dtype == np.float32:
            self.lib.gelu_tanh_float(x.ravel(), out.ravel(), x.size)
        else:
            raise ValueError(f"Unsupported dtype: {x.dtype}")
        
        return out.reshape(x.shape)
    
    def __call__(self, x, method='tanh', out=None):
        """
        Apply GELU activation
        
        Args:
            x: Input array
            method: 'exact' or 'tanh' (default: 'tanh')
            out: Optional output array
        """
        if method == 'exact':
            return self.exact(x, out)
        elif method == 'tanh':
            return self.tanh(x, out)
        else:
            raise ValueError(f"Unknown method: {method}")


def gelu_numpy_exact(x):
    """Reference implementation using numpy"""
    if HAS_SCIPY:
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    else:
        # Approximation using tanh
        return gelu_numpy_tanh(x)


def gelu_numpy_tanh(x):
    """Reference tanh approximation using numpy"""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))


def benchmark(func, x, iterations=100, warmup=10):
    """Benchmark a function"""
    # Warmup
    for _ in range(warmup):
        _ = func(x)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(x)
    end = time.perf_counter()
    
    elapsed = end - start
    return elapsed / iterations, result


def test_correctness(gelu, dtype=np.float32):
    """Test correctness of SLEEF GELU against numpy reference"""
    print(f"\n{'='*60}")
    print(f"Testing correctness ({dtype.__name__})")
    print('='*60)
    
    # Test data
    x = np.linspace(-4, 4, 1000, dtype=dtype)
    
    # Compute results
    sleef_exact = gelu.exact(x)
    sleef_tanh = gelu.tanh(x)
    numpy_exact = gelu_numpy_exact(x.astype(np.float64)).astype(dtype)
    numpy_tanh = gelu_numpy_tanh(x.astype(np.float64)).astype(dtype)
    
    # Compare
    error_exact = np.abs(sleef_exact - numpy_exact)
    error_tanh = np.abs(sleef_tanh - numpy_tanh)
    diff_exact_tanh = np.abs(sleef_exact - sleef_tanh)
    
    print(f"SLEEF exact vs NumPy exact:")
    print(f"  Max error:  {np.max(error_exact):.3e}")
    print(f"  Mean error: {np.mean(error_exact):.3e}")
    print(f"  RMS error:  {np.sqrt(np.mean(error_exact**2)):.3e}")
    
    print(f"\nSLEEF tanh vs NumPy tanh:")
    print(f"  Max error:  {np.max(error_tanh):.3e}")
    print(f"  Mean error: {np.mean(error_tanh):.3e}")
    print(f"  RMS error:  {np.sqrt(np.mean(error_tanh**2)):.3e}")
    
    print(f"\nSLEEF exact vs SLEEF tanh:")
    print(f"  Max diff:   {np.max(diff_exact_tanh):.3e}")
    print(f"  Mean diff:  {np.mean(diff_exact_tanh):.3e}")
    
    # Visual check at key points
    print(f"\nSpot checks:")
    test_points = [-3.0, -1.0, 0.0, 1.0, 3.0]
    for x_val in test_points:
        x_test = np.array([x_val], dtype=dtype)
        sleef_val = gelu.exact(x_test)[0]
        numpy_val = gelu_numpy_exact(np.array([x_val], dtype=np.float64))[0]
        print(f"  x={x_val:5.1f}: SLEEF={sleef_val:.6f}, NumPy={numpy_val:.6f}, "
              f"diff={abs(sleef_val - numpy_val):.3e}")


def benchmark_comparison(gelu, n=1000000, iterations=100, dtype=np.float32):
    """Compare performance of different implementations"""
    print(f"\n{'='*60}")
    print(f"Performance Benchmark (n={n:,}, dtype={dtype.__name__})")
    print('='*60)
    
    x = np.random.randn(n).astype(dtype) * 2.0
    
    # Benchmark SLEEF exact
    time_sleef_exact, result = benchmark(gelu.exact, x, iterations)
    throughput_sleef_exact = (n * dtype().itemsize * 2) / time_sleef_exact / 1e9
    
    print(f"\nSLEEF exact:")
    print(f"  Time:       {time_sleef_exact*1e3:.3f} ms")
    print(f"  Per element: {time_sleef_exact*1e9/n:.3f} ns")
    print(f"  Throughput: {throughput_sleef_exact:.2f} GB/s")
    
    # Benchmark SLEEF tanh
    time_sleef_tanh, result = benchmark(gelu.tanh, x, iterations)
    throughput_sleef_tanh = (n * dtype().itemsize * 2) / time_sleef_tanh / 1e9
    
    print(f"\nSLEEF tanh:")
    print(f"  Time:       {time_sleef_tanh*1e3:.3f} ms")
    print(f"  Per element: {time_sleef_tanh*1e9/n:.3f} ns")
    print(f"  Throughput: {throughput_sleef_tanh:.2f} GB/s")
    print(f"  Speedup:    {time_sleef_exact/time_sleef_tanh:.2f}x vs exact")
    
    # Benchmark NumPy exact
    time_numpy_exact, result = benchmark(gelu_numpy_exact, x, iterations)
    throughput_numpy_exact = (n * dtype().itemsize * 2) / time_numpy_exact / 1e9
    
    print(f"\nNumPy exact:")
    print(f"  Time:       {time_numpy_exact*1e3:.3f} ms")
    print(f"  Per element: {time_numpy_exact*1e9/n:.3f} ns")
    print(f"  Throughput: {throughput_numpy_exact:.2f} GB/s")
    print(f"  SLEEF speedup: {time_numpy_exact/time_sleef_exact:.2f}x")
    
    # Benchmark NumPy tanh
    time_numpy_tanh, result = benchmark(gelu_numpy_tanh, x, iterations)
    throughput_numpy_tanh = (n * dtype().itemsize * 2) / time_numpy_tanh / 1e9
    
    print(f"\nNumPy tanh:")
    print(f"  Time:       {time_numpy_tanh*1e3:.3f} ms")
    print(f"  Per element: {time_numpy_tanh*1e9/n:.3f} ns")
    print(f"  Throughput: {throughput_numpy_tanh:.2f} GB/s")
    print(f"  SLEEF speedup: {time_numpy_tanh/time_sleef_tanh:.2f}x")


def plot_gelu(gelu):
    """Plot GELU functions"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nWarning: matplotlib not available, skipping plot")
        return
    
    x = np.linspace(-4, 4, 1000, dtype=np.float32)
    
    sleef_exact = gelu.exact(x)
    sleef_tanh = gelu.tanh(x)
    numpy_exact = gelu_numpy_exact(x.astype(np.float64)).astype(np.float32)
    
    plt.figure(figsize=(12, 8))
    
    # Plot functions
    plt.subplot(2, 1, 1)
    plt.plot(x, sleef_exact, 'b-', label='SLEEF exact', linewidth=2)
    plt.plot(x, sleef_tanh, 'r--', label='SLEEF tanh', linewidth=2)
    plt.plot(x, numpy_exact, 'g:', label='NumPy exact', linewidth=1, alpha=0.7)
    plt.plot(x, x, 'k:', alpha=0.3, label='y=x')
    plt.xlabel('x')
    plt.ylabel('GELU(x)')
    plt.title('GELU Activation Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error
    plt.subplot(2, 1, 2)
    error_exact = np.abs(sleef_exact - numpy_exact)
    error_tanh = np.abs(sleef_tanh - numpy_exact)
    plt.semilogy(x, error_exact, 'b-', label='SLEEF exact error', linewidth=2)
    plt.semilogy(x, error_tanh, 'r--', label='SLEEF tanh error', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Error vs NumPy Reference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gelu_plot.png', dpi=150)
    print("\nPlot saved to gelu_plot.png")


def main():
    """Main function"""
    print("SLEEF GELU Kernel - Python Example")
    print("="*60)
    
    # Try to find the library
    lib_paths = [
        './libgelu_avx2.so',
        './libgelu_sse2.so',
        './libgelu_avx512.so',
        './libgelu_neon.so',
        './libgelu_sve.so',
    ]
    
    lib_path = None
    for path in lib_paths:
        if Path(path).exists():
            lib_path = path
            break
    
    if lib_path is None:
        print("\nError: No SLEEF GELU library found!")
        print("Please compile one of the following:")
        for path in lib_paths:
            print(f"  - {path}")
        print("\nExample:")
        print("  make -f Makefile.gelu libgelu_avx2.so")
        return 1
    
    print(f"Using library: {lib_path}\n")
    
    try:
        gelu = SleefGELU(lib_path)
    except Exception as e:
        print(f"Error loading library: {e}")
        return 1
    
    # Test correctness
    test_correctness(gelu, dtype=np.float32)
    test_correctness(gelu, dtype=np.float64)
    
    # Benchmark
    benchmark_comparison(gelu, n=1_000_000, iterations=100, dtype=np.float32)
    benchmark_comparison(gelu, n=1_000_000, iterations=100, dtype=np.float64)
    
    # Plot
    plot_gelu(gelu)
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())


