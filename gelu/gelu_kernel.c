//   Copyright 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// GELU Activation Function Implementation using SLEEF
// 
// This file provides vectorized GELU (Gaussian Error Linear Unit) kernels
// using SLEEF's high-performance math library for various SIMD architectures.
//
// GELU has two common formulations:
// 1. Exact:  GELU(x) = x * Φ(x) = 0.5 * x * (1 + erf(x/√2))
// 2. Approx: GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// Compilation examples:
// - SSE2:    gcc -O3 -msse2 -DENABLE_SSE2 gelu_kernel.c -lsleef -lm -o gelu_sse2
// - AVX2:    gcc -O3 -mavx2 -mfma -DENABLE_AVX2 gelu_kernel.c -lsleef -lm -o gelu_avx2
// - AVX512F: gcc -O3 -mavx512f -DENABLE_AVX512F gelu_kernel.c -lsleef -lm -o gelu_avx512
// - NEON:    gcc -O3 -DENABLE_ADVSIMD gelu_kernel.c -lsleef -lm -o gelu_neon
// - SVE:     gcc -O3 -march=armv8-a+sve -DENABLE_SVE gelu_kernel.c -lsleef -lm -o gelu_sve
// - RVV:     gcc -O3 -march=rv64gcv -DENABLE_RVVM1 gelu_kernel.c -lsleef -lm -o gelu_rvv

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "sleef.h"

// Architecture-specific configuration
#if defined(ENABLE_SSE2)
  #define ARCH_NAME "SSE2"
  #include <x86intrin.h>
  typedef __m128d vdouble;
  typedef __m128 vfloat;
  #define VDOUBLE_LANES 2
  #define VFLOAT_LANES 4
  #define VLOAD_D(p) _mm_loadu_pd(p)
  #define VLOAD_F(p) _mm_loadu_ps(p)
  #define VSTORE_D(p, v) _mm_storeu_pd(p, v)
  #define VSTORE_F(p, v) _mm_storeu_ps(p, v)
  #define VSET1_D(x) _mm_set1_pd(x)
  #define VSET1_F(x) _mm_set1_ps(x)
  #define VMUL_D(a, b) _mm_mul_pd(a, b)
  #define VMUL_F(a, b) _mm_mul_ps(a, b)
  #define VADD_D(a, b) _mm_add_pd(a, b)
  #define VADD_F(a, b) _mm_add_ps(a, b)
  #define VFMA_D(a, b, c) _mm_fmadd_pd(a, b, c)
  #define VFMA_F(a, b, c) _mm_fmadd_ps(a, b, c)
  #define SLEEF_ERF_D Sleef_erfd2_u10sse2
  #define SLEEF_ERF_F Sleef_erff4_u10sse2
  #define SLEEF_TANH_D Sleef_tanhd2_u10sse2
  #define SLEEF_TANH_F Sleef_tanhf4_u10sse2

#elif defined(ENABLE_AVX2)
  #define ARCH_NAME "AVX2"
  #include <x86intrin.h>
  typedef __m256d vdouble;
  typedef __m256 vfloat;
  #define VDOUBLE_LANES 4
  #define VFLOAT_LANES 8
  #define VLOAD_D(p) _mm256_loadu_pd(p)
  #define VLOAD_F(p) _mm256_loadu_ps(p)
  #define VSTORE_D(p, v) _mm256_storeu_pd(p, v)
  #define VSTORE_F(p, v) _mm256_storeu_ps(p, v)
  #define VSET1_D(x) _mm256_set1_pd(x)
  #define VSET1_F(x) _mm256_set1_ps(x)
  #define VMUL_D(a, b) _mm256_mul_pd(a, b)
  #define VMUL_F(a, b) _mm256_mul_ps(a, b)
  #define VADD_D(a, b) _mm256_add_pd(a, b)
  #define VADD_F(a, b) _mm256_add_ps(a, b)
  #define VFMA_D(a, b, c) _mm256_fmadd_pd(a, b, c)
  #define VFMA_F(a, b, c) _mm256_fmadd_ps(a, b, c)
  #define SLEEF_ERF_D Sleef_erfd4_u10avx2
  #define SLEEF_ERF_F Sleef_erff8_u10avx2
  #define SLEEF_TANH_D Sleef_tanhd4_u10avx2
  #define SLEEF_TANH_F Sleef_tanhf8_u10avx2

#elif defined(ENABLE_AVX512F)
  #define ARCH_NAME "AVX512F"
  #include <x86intrin.h>
  typedef __m512d vdouble;
  typedef __m512 vfloat;
  #define VDOUBLE_LANES 8
  #define VFLOAT_LANES 16
  #define VLOAD_D(p) _mm512_loadu_pd(p)
  #define VLOAD_F(p) _mm512_loadu_ps(p)
  #define VSTORE_D(p, v) _mm512_storeu_pd(p, v)
  #define VSTORE_F(p, v) _mm512_storeu_ps(p, v)
  #define VSET1_D(x) _mm512_set1_pd(x)
  #define VSET1_F(x) _mm512_set1_ps(x)
  #define VMUL_D(a, b) _mm512_mul_pd(a, b)
  #define VMUL_F(a, b) _mm512_mul_ps(a, b)
  #define VADD_D(a, b) _mm512_add_pd(a, b)
  #define VADD_F(a, b) _mm512_add_ps(a, b)
  #define VFMA_D(a, b, c) _mm512_fmadd_pd(a, b, c)
  #define VFMA_F(a, b, c) _mm512_fmadd_ps(a, b, c)
  #define SLEEF_ERF_D Sleef_erfd8_u10avx512f
  #define SLEEF_ERF_F Sleef_erff16_u10avx512f
  #define SLEEF_TANH_D Sleef_tanhd8_u10avx512f
  #define SLEEF_TANH_F Sleef_tanhf16_u10avx512f

#elif defined(ENABLE_ADVSIMD)
  #define ARCH_NAME "ADVSIMD (NEON)"
  #include <arm_neon.h>
  typedef float64x2_t vdouble;
  typedef float32x4_t vfloat;
  #define VDOUBLE_LANES 2
  #define VFLOAT_LANES 4
  #define VLOAD_D(p) vld1q_f64(p)
  #define VLOAD_F(p) vld1q_f32(p)
  #define VSTORE_D(p, v) vst1q_f64(p, v)
  #define VSTORE_F(p, v) vst1q_f32(p, v)
  #define VSET1_D(x) vdupq_n_f64(x)
  #define VSET1_F(x) vdupq_n_f32(x)
  #define VMUL_D(a, b) vmulq_f64(a, b)
  #define VMUL_F(a, b) vmulq_f32(a, b)
  #define VADD_D(a, b) vaddq_f64(a, b)
  #define VADD_F(a, b) vaddq_f32(a, b)
  #define VFMA_D(a, b, c) vfmaq_f64(c, a, b)
  #define VFMA_F(a, b, c) vfmaq_f32(c, a, b)
  #define SLEEF_ERF_D Sleef_erfd2_u10advsimd
  #define SLEEF_ERF_F Sleef_erff4_u10advsimd
  #define SLEEF_TANH_D Sleef_tanhd2_u10advsimd
  #define SLEEF_TANH_F Sleef_tanhf4_u10advsimd

#elif defined(ENABLE_SVE)
  #define ARCH_NAME "SVE"
  #include <arm_sve.h>
  typedef svfloat64_t vdouble;
  typedef svfloat32_t vfloat;
  #define VDOUBLE_LANES svcntd()
  #define VFLOAT_LANES svcntw()
  #define USE_SVE 1
  #define SLEEF_ERF_D Sleef_erfdx_u10sve
  #define SLEEF_ERF_F Sleef_erffx_u10sve
  #define SLEEF_TANH_D Sleef_tanhdx_u10sve
  #define SLEEF_TANH_F Sleef_tanhfx_u10sve

#elif defined(ENABLE_RVVM1)
  #define ARCH_NAME "RISC-V Vector (LMUL=1)"
  #include <riscv_vector.h>
  typedef vfloat64m1_t vdouble;
  typedef vfloat32m1_t vfloat;
  #define USE_RVV 1
  #define SLEEF_ERF_D Sleef_erfdx_u10rvvm1
  #define SLEEF_ERF_F Sleef_erffx_u10rvvm1
  #define SLEEF_TANH_D Sleef_tanhdx_u10rvvm1
  #define SLEEF_TANH_F Sleef_tanhfx_u10rvvm1

#elif defined(ENABLE_RVVM2)
  #define ARCH_NAME "RISC-V Vector (LMUL=2)"
  #include <riscv_vector.h>
  typedef vfloat64m2_t vdouble;
  typedef vfloat32m2_t vfloat;
  #define USE_RVV 1
  #define SLEEF_ERF_D Sleef_erfdx_u10rvvm2
  #define SLEEF_ERF_F Sleef_erffx_u10rvvm2
  #define SLEEF_TANH_D Sleef_tanhdx_u10rvvm2
  #define SLEEF_TANH_F Sleef_tanhfx_u10rvvm2

#else
  #error "No SIMD architecture defined. Use -DENABLE_SSE2, -DENABLE_AVX2, -DENABLE_AVX512F, -DENABLE_ADVSIMD, -DENABLE_SVE, -DENABLE_RVVM1, or -DENABLE_RVVM2"
#endif

// Mathematical constants
#define SQRT_2_INV 0.7071067811865475244       // 1/√2
#define SQRT_2_INV_F 0.7071067811865475244f
#define SQRT_2_OVER_PI 0.7978845608028653558   // √(2/π)
#define SQRT_2_OVER_PI_F 0.7978845608028653558f
#define GELU_COEFF 0.044715                     // Coefficient in tanh approximation
#define GELU_COEFF_F 0.044715f

// ============================================================================
// GELU Kernels - Double Precision
// ============================================================================

#if defined(USE_SVE)

// GELU exact version using erf - Double precision (SVE)
void gelu_exact_d_sve(const double *input, double *output, size_t n) {
  size_t lanes = VDOUBLE_LANES;
  size_t i;
  for (i = 0; i + lanes <= n; i += lanes) {
    svbool_t pg = svptrue_b64();
    vdouble x = svld1_f64(pg, input + i);
    vdouble half = svdup_f64(0.5);
    vdouble sqrt2_inv = svdup_f64(SQRT_2_INV);
    vdouble one = svdup_f64(1.0);
    
    // GELU(x) = 0.5 * x * (1 + erf(x/√2))
    vdouble x_scaled = svmul_f64_x(pg, x, sqrt2_inv);
    vdouble erf_val = SLEEF_ERF_D(x_scaled);
    vdouble one_plus_erf = svadd_f64_x(pg, one, erf_val);
    vdouble result = svmul_f64_x(pg, svmul_f64_x(pg, half, x), one_plus_erf);
    
    svst1_f64(pg, output + i, result);
  }
  
  // Handle remaining elements
  if (i < n) {
    svbool_t pg = svwhilelt_b64(i, n);
    vdouble x = svld1_f64(pg, input + i);
    vdouble half = svdup_f64(0.5);
    vdouble sqrt2_inv = svdup_f64(SQRT_2_INV);
    vdouble one = svdup_f64(1.0);
    
    vdouble x_scaled = svmul_f64_x(pg, x, sqrt2_inv);
    vdouble erf_val = SLEEF_ERF_D(x_scaled);
    vdouble one_plus_erf = svadd_f64_x(pg, one, erf_val);
    vdouble result = svmul_f64_x(pg, svmul_f64_x(pg, half, x), one_plus_erf);
    
    svst1_f64(pg, output + i, result);
  }
}

// GELU tanh approximation - Double precision (SVE)
void gelu_tanh_d_sve(const double *input, double *output, size_t n) {
  size_t lanes = VDOUBLE_LANES;
  size_t i;
  for (i = 0; i + lanes <= n; i += lanes) {
    svbool_t pg = svptrue_b64();
    vdouble x = svld1_f64(pg, input + i);
    vdouble half = svdup_f64(0.5);
    vdouble one = svdup_f64(1.0);
    vdouble coeff = svdup_f64(GELU_COEFF);
    vdouble sqrt_2_over_pi = svdup_f64(SQRT_2_OVER_PI);
    
    // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    vdouble x2 = svmul_f64_x(pg, x, x);
    vdouble x3 = svmul_f64_x(pg, x2, x);
    vdouble inner = svmla_f64_x(pg, x, x3, coeff);  // x + 0.044715 * x³
    vdouble scaled = svmul_f64_x(pg, sqrt_2_over_pi, inner);
    vdouble tanh_val = SLEEF_TANH_D(scaled);
    vdouble one_plus_tanh = svadd_f64_x(pg, one, tanh_val);
    vdouble result = svmul_f64_x(pg, svmul_f64_x(pg, half, x), one_plus_tanh);
    
    svst1_f64(pg, output + i, result);
  }
  
  // Handle remaining elements
  if (i < n) {
    svbool_t pg = svwhilelt_b64(i, n);
    vdouble x = svld1_f64(pg, input + i);
    vdouble half = svdup_f64(0.5);
    vdouble one = svdup_f64(1.0);
    vdouble coeff = svdup_f64(GELU_COEFF);
    vdouble sqrt_2_over_pi = svdup_f64(SQRT_2_OVER_PI);
    
    vdouble x2 = svmul_f64_x(pg, x, x);
    vdouble x3 = svmul_f64_x(pg, x2, x);
    vdouble inner = svmla_f64_x(pg, x, x3, coeff);
    vdouble scaled = svmul_f64_x(pg, sqrt_2_over_pi, inner);
    vdouble tanh_val = SLEEF_TANH_D(scaled);
    vdouble one_plus_tanh = svadd_f64_x(pg, one, tanh_val);
    vdouble result = svmul_f64_x(pg, svmul_f64_x(pg, half, x), one_plus_tanh);
    
    svst1_f64(pg, output + i, result);
  }
}

#elif defined(USE_RVV)

// Define LMUL-specific macros
#if defined(ENABLE_RVVM1)
  #define RVV_VSETVL_D(n) __riscv_vsetvl_e64m1(n)
  #define RVV_VSETVL_F(n) __riscv_vsetvl_e32m1(n)
  #define RVV_VLE_D(p, vl) __riscv_vle64_v_f64m1(p, vl)
  #define RVV_VLE_F(p, vl) __riscv_vle32_v_f32m1(p, vl)
  #define RVV_VSE_D(p, v, vl) __riscv_vse64_v_f64m1(p, v, vl)
  #define RVV_VSE_F(p, v, vl) __riscv_vse32_v_f32m1(p, v, vl)
  #define RVV_VFMUL_VF_D(v, f, vl) __riscv_vfmul_vf_f64m1(v, f, vl)
  #define RVV_VFMUL_VF_F(v, f, vl) __riscv_vfmul_vf_f32m1(v, f, vl)
  #define RVV_VFMUL_VV_D(v1, v2, vl) __riscv_vfmul_vv_f64m1(v1, v2, vl)
  #define RVV_VFMUL_VV_F(v1, v2, vl) __riscv_vfmul_vv_f32m1(v1, v2, vl)
  #define RVV_VFADD_VF_D(v, f, vl) __riscv_vfadd_vf_f64m1(v, f, vl)
  #define RVV_VFADD_VF_F(v, f, vl) __riscv_vfadd_vf_f32m1(v, f, vl)
  #define RVV_VFMACC_VF_D(vd, f, vs, vl) __riscv_vfmacc_vf_f64m1(vd, f, vs, vl)
  #define RVV_VFMACC_VF_F(vd, f, vs, vl) __riscv_vfmacc_vf_f32m1(vd, f, vs, vl)
#elif defined(ENABLE_RVVM2)
  #define RVV_VSETVL_D(n) __riscv_vsetvl_e64m2(n)
  #define RVV_VSETVL_F(n) __riscv_vsetvl_e32m2(n)
  #define RVV_VLE_D(p, vl) __riscv_vle64_v_f64m2(p, vl)
  #define RVV_VLE_F(p, vl) __riscv_vle32_v_f32m2(p, vl)
  #define RVV_VSE_D(p, v, vl) __riscv_vse64_v_f64m2(p, v, vl)
  #define RVV_VSE_F(p, v, vl) __riscv_vse32_v_f32m2(p, v, vl)
  #define RVV_VFMUL_VF_D(v, f, vl) __riscv_vfmul_vf_f64m2(v, f, vl)
  #define RVV_VFMUL_VF_F(v, f, vl) __riscv_vfmul_vf_f32m2(v, f, vl)
  #define RVV_VFMUL_VV_D(v1, v2, vl) __riscv_vfmul_vv_f64m2(v1, v2, vl)
  #define RVV_VFMUL_VV_F(v1, v2, vl) __riscv_vfmul_vv_f32m2(v1, v2, vl)
  #define RVV_VFADD_VF_D(v, f, vl) __riscv_vfadd_vf_f64m2(v, f, vl)
  #define RVV_VFADD_VF_F(v, f, vl) __riscv_vfadd_vf_f32m2(v, f, vl)
  #define RVV_VFMACC_VF_D(vd, f, vs, vl) __riscv_vfmacc_vf_f64m2(vd, f, vs, vl)
  #define RVV_VFMACC_VF_F(vd, f, vs, vl) __riscv_vfmacc_vf_f32m2(vd, f, vs, vl)
#endif

// GELU exact version using erf - Double precision (RVV)
void gelu_exact_d_rvv(const double *input, double *output, size_t n) {
  size_t i = 0;
  while (i < n) {
    size_t vl = RVV_VSETVL_D(n - i);
    vdouble x = RVV_VLE_D(input + i, vl);
    
    // GELU(x) = 0.5 * x * (1 + erf(x/√2))
    vdouble x_scaled = RVV_VFMUL_VF_D(x, SQRT_2_INV, vl);
    vdouble erf_val = SLEEF_ERF_D(x_scaled, vl);
    vdouble one_plus_erf = RVV_VFADD_VF_D(erf_val, 1.0, vl);
    vdouble half_x = RVV_VFMUL_VF_D(x, 0.5, vl);
    vdouble result = RVV_VFMUL_VV_D(half_x, one_plus_erf, vl);
    
    RVV_VSE_D(output + i, result, vl);
    i += vl;
  }
}

// GELU tanh approximation - Double precision (RVV)
void gelu_tanh_d_rvv(const double *input, double *output, size_t n) {
  size_t i = 0;
  while (i < n) {
    size_t vl = RVV_VSETVL_D(n - i);
    vdouble x = RVV_VLE_D(input + i, vl);
    
    // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    vdouble x2 = RVV_VFMUL_VV_D(x, x, vl);
    vdouble x3 = RVV_VFMUL_VV_D(x2, x, vl);
    vdouble inner = RVV_VFMACC_VF_D(x, GELU_COEFF, x3, vl);  // x + coeff * x³
    vdouble scaled = RVV_VFMUL_VF_D(inner, SQRT_2_OVER_PI, vl);
    vdouble tanh_val = SLEEF_TANH_D(scaled, vl);
    vdouble one_plus_tanh = RVV_VFADD_VF_D(tanh_val, 1.0, vl);
    vdouble half_x = RVV_VFMUL_VF_D(x, 0.5, vl);
    vdouble result = RVV_VFMUL_VV_D(half_x, one_plus_tanh, vl);
    
    RVV_VSE_D(output + i, result, vl);
    i += vl;
  }
}

#else

// GELU exact version using erf - Double precision (x86/ARM fixed-length SIMD)
void gelu_exact_d(const double *input, double *output, size_t n) {
  vdouble half = VSET1_D(0.5);
  vdouble one = VSET1_D(1.0);
  vdouble sqrt2_inv = VSET1_D(SQRT_2_INV);
  
  size_t i;
  for (i = 0; i + VDOUBLE_LANES <= n; i += VDOUBLE_LANES) {
    vdouble x = VLOAD_D(input + i);
    
    // GELU(x) = 0.5 * x * (1 + erf(x/√2))
    vdouble x_scaled = VMUL_D(x, sqrt2_inv);
    vdouble erf_val = SLEEF_ERF_D(x_scaled);
    vdouble one_plus_erf = VADD_D(one, erf_val);
    vdouble result = VMUL_D(VMUL_D(half, x), one_plus_erf);
    
    VSTORE_D(output + i, result);
  }
  
  // Handle remaining elements
  for (; i < n; i++) {
    double x = input[i];
    output[i] = 0.5 * x * (1.0 + erf(x * SQRT_2_INV));
  }
}

// GELU tanh approximation - Double precision (x86/ARM fixed-length SIMD)
void gelu_tanh_d(const double *input, double *output, size_t n) {
  vdouble half = VSET1_D(0.5);
  vdouble one = VSET1_D(1.0);
  vdouble coeff = VSET1_D(GELU_COEFF);
  vdouble sqrt_2_over_pi = VSET1_D(SQRT_2_OVER_PI);
  
  size_t i;
  for (i = 0; i + VDOUBLE_LANES <= n; i += VDOUBLE_LANES) {
    vdouble x = VLOAD_D(input + i);
    
    // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    vdouble x2 = VMUL_D(x, x);
    vdouble x3 = VMUL_D(x2, x);
    vdouble inner = VFMA_D(x3, coeff, x);  // x + 0.044715 * x³
    vdouble scaled = VMUL_D(sqrt_2_over_pi, inner);
    vdouble tanh_val = SLEEF_TANH_D(scaled);
    vdouble one_plus_tanh = VADD_D(one, tanh_val);
    vdouble result = VMUL_D(VMUL_D(half, x), one_plus_tanh);
    
    VSTORE_D(output + i, result);
  }
  
  // Handle remaining elements
  for (; i < n; i++) {
    double x = input[i];
    double inner = x + GELU_COEFF * x * x * x;
    output[i] = 0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * inner));
  }
}

#endif

// ============================================================================
// GELU Kernels - Single Precision
// ============================================================================

#if defined(USE_SVE)

// GELU exact version using erf - Single precision (SVE)
void gelu_exact_f_sve(const float *input, float *output, size_t n) {
  size_t lanes = VFLOAT_LANES;
  size_t i;
  for (i = 0; i + lanes <= n; i += lanes) {
    svbool_t pg = svptrue_b32();
    vfloat x = svld1_f32(pg, input + i);
    vfloat half = svdup_f32(0.5f);
    vfloat sqrt2_inv = svdup_f32(SQRT_2_INV_F);
    vfloat one = svdup_f32(1.0f);
    
    vfloat x_scaled = svmul_f32_x(pg, x, sqrt2_inv);
    vfloat erf_val = SLEEF_ERF_F(x_scaled);
    vfloat one_plus_erf = svadd_f32_x(pg, one, erf_val);
    vfloat result = svmul_f32_x(pg, svmul_f32_x(pg, half, x), one_plus_erf);
    
    svst1_f32(pg, output + i, result);
  }
  
  if (i < n) {
    svbool_t pg = svwhilelt_b32(i, n);
    vfloat x = svld1_f32(pg, input + i);
    vfloat half = svdup_f32(0.5f);
    vfloat sqrt2_inv = svdup_f32(SQRT_2_INV_F);
    vfloat one = svdup_f32(1.0f);
    
    vfloat x_scaled = svmul_f32_x(pg, x, sqrt2_inv);
    vfloat erf_val = SLEEF_ERF_F(x_scaled);
    vfloat one_plus_erf = svadd_f32_x(pg, one, erf_val);
    vfloat result = svmul_f32_x(pg, svmul_f32_x(pg, half, x), one_plus_erf);
    
    svst1_f32(pg, output + i, result);
  }
}

// GELU tanh approximation - Single precision (SVE)
void gelu_tanh_f_sve(const float *input, float *output, size_t n) {
  size_t lanes = VFLOAT_LANES;
  size_t i;
  for (i = 0; i + lanes <= n; i += lanes) {
    svbool_t pg = svptrue_b32();
    vfloat x = svld1_f32(pg, input + i);
    vfloat half = svdup_f32(0.5f);
    vfloat one = svdup_f32(1.0f);
    vfloat coeff = svdup_f32(GELU_COEFF_F);
    vfloat sqrt_2_over_pi = svdup_f32(SQRT_2_OVER_PI_F);
    
    vfloat x2 = svmul_f32_x(pg, x, x);
    vfloat x3 = svmul_f32_x(pg, x2, x);
    vfloat inner = svmla_f32_x(pg, x, x3, coeff);
    vfloat scaled = svmul_f32_x(pg, sqrt_2_over_pi, inner);
    vfloat tanh_val = SLEEF_TANH_F(scaled);
    vfloat one_plus_tanh = svadd_f32_x(pg, one, tanh_val);
    vfloat result = svmul_f32_x(pg, svmul_f32_x(pg, half, x), one_plus_tanh);
    
    svst1_f32(pg, output + i, result);
  }
  
  if (i < n) {
    svbool_t pg = svwhilelt_b32(i, n);
    vfloat x = svld1_f32(pg, input + i);
    vfloat half = svdup_f32(0.5f);
    vfloat one = svdup_f32(1.0f);
    vfloat coeff = svdup_f32(GELU_COEFF_F);
    vfloat sqrt_2_over_pi = svdup_f32(SQRT_2_OVER_PI_F);
    
    vfloat x2 = svmul_f32_x(pg, x, x);
    vfloat x3 = svmul_f32_x(pg, x2, x);
    vfloat inner = svmla_f32_x(pg, x, x3, coeff);
    vfloat scaled = svmul_f32_x(pg, sqrt_2_over_pi, inner);
    vfloat tanh_val = SLEEF_TANH_F(scaled);
    vfloat one_plus_tanh = svadd_f32_x(pg, one, tanh_val);
    vfloat result = svmul_f32_x(pg, svmul_f32_x(pg, half, x), one_plus_tanh);
    
    svst1_f32(pg, output + i, result);
  }
}

#elif defined(USE_RVV)

// GELU exact version using erf - Single precision (RVV)
void gelu_exact_f_rvv(const float *input, float *output, size_t n) {
  size_t i = 0;
  while (i < n) {
    size_t vl = RVV_VSETVL_F(n - i);
    vfloat x = RVV_VLE_F(input + i, vl);
    
    vfloat x_scaled = RVV_VFMUL_VF_F(x, SQRT_2_INV_F, vl);
    vfloat erf_val = SLEEF_ERF_F(x_scaled, vl);
    vfloat one_plus_erf = RVV_VFADD_VF_F(erf_val, 1.0f, vl);
    vfloat half_x = RVV_VFMUL_VF_F(x, 0.5f, vl);
    vfloat result = RVV_VFMUL_VV_F(half_x, one_plus_erf, vl);
    
    RVV_VSE_F(output + i, result, vl);
    i += vl;
  }
}

// GELU tanh approximation - Single precision (RVV)
void gelu_tanh_f_rvv(const float *input, float *output, size_t n) {
  size_t i = 0;
  while (i < n) {
    size_t vl = RVV_VSETVL_F(n - i);
    vfloat x = RVV_VLE_F(input + i, vl);
    
    vfloat x2 = RVV_VFMUL_VV_F(x, x, vl);
    vfloat x3 = RVV_VFMUL_VV_F(x2, x, vl);
    vfloat inner = RVV_VFMACC_VF_F(x, GELU_COEFF_F, x3, vl);
    vfloat scaled = RVV_VFMUL_VF_F(inner, SQRT_2_OVER_PI_F, vl);
    vfloat tanh_val = SLEEF_TANH_F(scaled, vl);
    vfloat one_plus_tanh = RVV_VFADD_VF_F(tanh_val, 1.0f, vl);
    vfloat half_x = RVV_VFMUL_VF_F(x, 0.5f, vl);
    vfloat result = RVV_VFMUL_VV_F(half_x, one_plus_tanh, vl);
    
    RVV_VSE_F(output + i, result, vl);
    i += vl;
  }
}

#else

// GELU exact version using erf - Single precision (x86/ARM fixed-length SIMD)
void gelu_exact_f(const float *input, float *output, size_t n) {
  vfloat half = VSET1_F(0.5f);
  vfloat one = VSET1_F(1.0f);
  vfloat sqrt2_inv = VSET1_F(SQRT_2_INV_F);
  
  size_t i;
  for (i = 0; i + VFLOAT_LANES <= n; i += VFLOAT_LANES) {
    vfloat x = VLOAD_F(input + i);
    
    vfloat x_scaled = VMUL_F(x, sqrt2_inv);
    vfloat erf_val = SLEEF_ERF_F(x_scaled);
    vfloat one_plus_erf = VADD_F(one, erf_val);
    vfloat result = VMUL_F(VMUL_F(half, x), one_plus_erf);
    
    VSTORE_F(output + i, result);
  }
  
  for (; i < n; i++) {
    float x = input[i];
    output[i] = 0.5f * x * (1.0f + erff(x * SQRT_2_INV_F));
  }
}

// GELU tanh approximation - Single precision (x86/ARM fixed-length SIMD)
void gelu_tanh_f(const float *input, float *output, size_t n) {
  vfloat half = VSET1_F(0.5f);
  vfloat one = VSET1_F(1.0f);
  vfloat coeff = VSET1_F(GELU_COEFF_F);
  vfloat sqrt_2_over_pi = VSET1_F(SQRT_2_OVER_PI_F);
  
  size_t i;
  for (i = 0; i + VFLOAT_LANES <= n; i += VFLOAT_LANES) {
    vfloat x = VLOAD_F(input + i);
    
    vfloat x2 = VMUL_F(x, x);
    vfloat x3 = VMUL_F(x2, x);
    vfloat inner = VFMA_F(x3, coeff, x);
    vfloat scaled = VMUL_F(sqrt_2_over_pi, inner);
    vfloat tanh_val = SLEEF_TANH_F(scaled);
    vfloat one_plus_tanh = VADD_F(one, tanh_val);
    vfloat result = VMUL_F(VMUL_F(half, x), one_plus_tanh);
    
    VSTORE_F(output + i, result);
  }
  
  for (; i < n; i++) {
    float x = input[i];
    float inner = x + GELU_COEFF_F * x * x * x;
    output[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI_F * inner));
  }
}

#endif

// ============================================================================
// Unified API
// ============================================================================

void gelu_exact_double(const double *input, double *output, size_t n) {
#if defined(USE_SVE)
  gelu_exact_d_sve(input, output, n);
#elif defined(USE_RVV)
  gelu_exact_d_rvv(input, output, n);
#else
  gelu_exact_d(input, output, n);
#endif
}

void gelu_tanh_double(const double *input, double *output, size_t n) {
#if defined(USE_SVE)
  gelu_tanh_d_sve(input, output, n);
#elif defined(USE_RVV)
  gelu_tanh_d_rvv(input, output, n);
#else
  gelu_tanh_d(input, output, n);
#endif
}

void gelu_exact_float(const float *input, float *output, size_t n) {
#if defined(USE_SVE)
  gelu_exact_f_sve(input, output, n);
#elif defined(USE_RVV)
  gelu_exact_f_rvv(input, output, n);
#else
  gelu_exact_f(input, output, n);
#endif
}

void gelu_tanh_float(const float *input, float *output, size_t n) {
#if defined(USE_SVE)
  gelu_tanh_f_sve(input, output, n);
#elif defined(USE_RVV)
  gelu_tanh_f_rvv(input, output, n);
#else
  gelu_tanh_f(input, output, n);
#endif
}

// ============================================================================
// Testing and Benchmarking
// ============================================================================

static double get_time_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Scalar reference implementations
static double gelu_exact_scalar(double x) {
  return 0.5 * x * (1.0 + erf(x * SQRT_2_INV));
}

static float gelu_exact_scalar_f(float x) {
  return 0.5f * x * (1.0f + erff(x * SQRT_2_INV_F));
}

static double gelu_tanh_scalar(double x) {
  double inner = x + GELU_COEFF * x * x * x;
  return 0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * inner));
}

static float gelu_tanh_scalar_f(float x) {
  float inner = x + GELU_COEFF_F * x * x * x;
  return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI_F * inner));
}

// Test correctness
static void test_correctness(void) {
  const size_t n = 1000;
  double *input_d = aligned_alloc(64, n * sizeof(double));
  double *output_exact_d = aligned_alloc(64, n * sizeof(double));
  double *output_tanh_d = aligned_alloc(64, n * sizeof(double));
  float *input_f = aligned_alloc(64, n * sizeof(float));
  float *output_exact_f = aligned_alloc(64, n * sizeof(float));
  float *output_tanh_f = aligned_alloc(64, n * sizeof(float));
  
  // Initialize test data
  for (size_t i = 0; i < n; i++) {
    input_d[i] = -4.0 + 8.0 * i / n;
    input_f[i] = -4.0f + 8.0f * i / n;
  }
  
  // Test double precision
  gelu_exact_double(input_d, output_exact_d, n);
  gelu_tanh_double(input_d, output_tanh_d, n);
  
  double max_error_exact = 0.0;
  double max_error_tanh = 0.0;
  double max_diff = 0.0;
  
  for (size_t i = 0; i < n; i++) {
    double ref_exact = gelu_exact_scalar(input_d[i]);
    double ref_tanh = gelu_tanh_scalar(input_d[i]);
    double err_exact = fabs(output_exact_d[i] - ref_exact);
    double err_tanh = fabs(output_tanh_d[i] - ref_tanh);
    double diff = fabs(output_exact_d[i] - output_tanh_d[i]);
    
    if (err_exact > max_error_exact) max_error_exact = err_exact;
    if (err_tanh > max_error_tanh) max_error_tanh = err_tanh;
    if (diff > max_diff) max_diff = diff;
  }
  
  printf("Double precision correctness test:\n");
  printf("  GELU exact max error:         %.3e\n", max_error_exact);
  printf("  GELU tanh approx max error:   %.3e\n", max_error_tanh);
  printf("  Max difference (exact vs tanh): %.3e\n\n", max_diff);
  
  // Test single precision
  gelu_exact_float(input_f, output_exact_f, n);
  gelu_tanh_float(input_f, output_tanh_f, n);
  
  float max_error_exact_f = 0.0f;
  float max_error_tanh_f = 0.0f;
  float max_diff_f = 0.0f;
  
  for (size_t i = 0; i < n; i++) {
    float ref_exact = gelu_exact_scalar_f(input_f[i]);
    float ref_tanh = gelu_tanh_scalar_f(input_f[i]);
    float err_exact = fabsf(output_exact_f[i] - ref_exact);
    float err_tanh = fabsf(output_tanh_f[i] - ref_tanh);
    float diff = fabsf(output_exact_f[i] - output_tanh_f[i]);
    
    if (err_exact > max_error_exact_f) max_error_exact_f = err_exact;
    if (err_tanh > max_error_tanh_f) max_error_tanh_f = err_tanh;
    if (diff > max_diff_f) max_diff_f = diff;
  }
  
  printf("Single precision correctness test:\n");
  printf("  GELU exact max error:         %.3e\n", max_error_exact_f);
  printf("  GELU tanh approx max error:   %.3e\n", max_error_tanh_f);
  printf("  Max difference (exact vs tanh): %.3e\n\n", max_diff_f);
  
  free(input_d); free(output_exact_d); free(output_tanh_d);
  free(input_f); free(output_exact_f); free(output_tanh_f);
}

// Benchmark
static void benchmark(size_t n, size_t iterations) {
  double *input_d = aligned_alloc(64, n * sizeof(double));
  double *output_d = aligned_alloc(64, n * sizeof(double));
  float *input_f = aligned_alloc(64, n * sizeof(float));
  float *output_f = aligned_alloc(64, n * sizeof(float));
  
  for (size_t i = 0; i < n; i++) {
    input_d[i] = -3.0 + 6.0 * i / n;
    input_f[i] = -3.0f + 6.0f * i / n;
  }
  
  printf("Benchmark (n=%zu, iterations=%zu):\n", n, iterations);
  
  // Double precision - exact
  double start = get_time_sec();
  for (size_t iter = 0; iter < iterations; iter++) {
    gelu_exact_double(input_d, output_d, n);
  }
  double elapsed = get_time_sec() - start;
  printf("  GELU exact (double):    %8.3f ns/elem  (%8.3f GB/s)\n",
         elapsed * 1e9 / (iterations * n),
         (iterations * n * 2 * sizeof(double)) / (elapsed * 1e9));
  
  // Double precision - tanh
  start = get_time_sec();
  for (size_t iter = 0; iter < iterations; iter++) {
    gelu_tanh_double(input_d, output_d, n);
  }
  elapsed = get_time_sec() - start;
  printf("  GELU tanh (double):     %8.3f ns/elem  (%8.3f GB/s)\n",
         elapsed * 1e9 / (iterations * n),
         (iterations * n * 2 * sizeof(double)) / (elapsed * 1e9));
  
  // Single precision - exact
  start = get_time_sec();
  for (size_t iter = 0; iter < iterations; iter++) {
    gelu_exact_float(input_f, output_f, n);
  }
  elapsed = get_time_sec() - start;
  printf("  GELU exact (float):     %8.3f ns/elem  (%8.3f GB/s)\n",
         elapsed * 1e9 / (iterations * n),
         (iterations * n * 2 * sizeof(float)) / (elapsed * 1e9));
  
  // Single precision - tanh
  start = get_time_sec();
  for (size_t iter = 0; iter < iterations; iter++) {
    gelu_tanh_float(input_f, output_f, n);
  }
  elapsed = get_time_sec() - start;
  printf("  GELU tanh (float):      %8.3f ns/elem  (%8.3f GB/s)\n\n",
         elapsed * 1e9 / (iterations * n),
         (iterations * n * 2 * sizeof(float)) / (elapsed * 1e9));
  
  free(input_d); free(output_d);
  free(input_f); free(output_f);
}

int main(int argc, char **argv) {
  printf("========================================\n");
  printf("GELU Kernel using SLEEF (%s)\n", ARCH_NAME);
  printf("========================================\n\n");
  
  test_correctness();
  
  size_t n = 1024 * 1024;  // 1M elements
  size_t iterations = 100;
  
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) iterations = atoi(argv[2]);
  
  benchmark(n, iterations);
  
  return 0;
}


