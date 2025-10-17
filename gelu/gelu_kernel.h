//   Copyright 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef GELU_KERNEL_H
#define GELU_KERNEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apply GELU activation using exact error function (double precision)
 * 
 * Computes: GELU(x) = 0.5 * x * (1 + erf(x/√2))
 * 
 * This is the exact mathematical definition of GELU and provides
 * the highest accuracy. Uses SLEEF's erf implementation which has
 * < 1 ULP error.
 * 
 * @param input  Input array (must be contiguous, 64-byte aligned recommended)
 * @param output Output array (must be contiguous, 64-byte aligned recommended)
 * @param n      Number of elements to process
 * 
 * @note For best performance, ensure arrays are 64-byte aligned:
 *       double *data = aligned_alloc(64, n * sizeof(double));
 */
void gelu_exact_double(const double *input, double *output, size_t n);

/**
 * @brief Apply GELU activation using tanh approximation (double precision)
 * 
 * Computes: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * 
 * This is a fast approximation that is very close to the exact version
 * (max error < 0.005). Typically 20-30% faster than exact version.
 * 
 * @param input  Input array (must be contiguous, 64-byte aligned recommended)
 * @param output Output array (must be contiguous, 64-byte aligned recommended)
 * @param n      Number of elements to process
 */
void gelu_tanh_double(const double *input, double *output, size_t n);

/**
 * @brief Apply GELU activation using exact error function (single precision)
 * 
 * Computes: GELU(x) = 0.5 * x * (1 + erf(x/√2))
 * 
 * Single precision version of gelu_exact_double().
 * 
 * @param input  Input array (must be contiguous, 64-byte aligned recommended)
 * @param output Output array (must be contiguous, 64-byte aligned recommended)
 * @param n      Number of elements to process
 */
void gelu_exact_float(const float *input, float *output, size_t n);

/**
 * @brief Apply GELU activation using tanh approximation (single precision)
 * 
 * Computes: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * 
 * Single precision version of gelu_tanh_double().
 * 
 * @param input  Input array (must be contiguous, 64-byte aligned recommended)
 * @param output Output array (must be contiguous, 64-byte aligned recommended)
 * @param n      Number of elements to process
 */
void gelu_tanh_float(const float *input, float *output, size_t n);

#ifdef __cplusplus
}
#endif

#endif // GELU_KERNEL_H


