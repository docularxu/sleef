#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define _ISOC11_SOURCE
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "sleef.h"

#if defined(ENABLE_SSE2)
  #define SIMD_NAME "SSE2"
  #define VDOUBLE Sleef___m128d_2
  #define VFLOAT Sleef___m128_2
  typedef __m128d vdouble;
  typedef __m128 vfloat;
  #define VDOUBLE_LANES 2
  #define VFLOAT_LANES 4
  #define VLOAD_DOUBLE(p) _mm_loadu_pd(p)
  #define VLOAD_FLOAT(p) _mm_loadu_ps(p)
  #define VSTORE_DOUBLE(p, v) _mm_storeu_pd(p, v)
  #define VSTORE_FLOAT(p, v) _mm_storeu_ps(p, v)
  #define SLEEF_SIN Sleef_sind2_u10sse2
  #define SLEEF_SIN_STR STRINGIFY(Sleef_sind2_u10sse2)
  #define SLEEF_COS Sleef_cosd2_u10sse2
  #define SLEEF_COS_STR STRINGIFY(Sleef_cosd2_u10sse2)
  #define SLEEF_TAN Sleef_tand2_u10sse2
  #define SLEEF_TAN_STR STRINGIFY(Sleef_tand2_u10sse2)
  #define SLEEF_EXP Sleef_expd2_u10sse2
  #define SLEEF_EXP_STR STRINGIFY(Sleef_expd2_u10sse2)
  #define SLEEF_LOG Sleef_logd2_u10sse2
  #define SLEEF_LOG_STR STRINGIFY(Sleef_logd2_u10sse2)
  #define SLEEF_POW Sleef_powd2_u10sse2
  #define SLEEF_SQRT Sleef_sqrtd2_u05sse2
  #define SLEEF_SQRT_STR STRINGIFY(Sleef_sqrtd2_u05sse2)
  #define SLEEF_TANH Sleef_tanhd2_u10sse2
  #define SLEEF_TANH_STR STRINGIFY(Sleef_tanhd2_u10sse2)
  #define SLEEF_SINF Sleef_sinf4_u10sse2
  #define SLEEF_SINF_STR STRINGIFY(Sleef_sinf4_u10sse2)
  #define SLEEF_COSF Sleef_cosf4_u10sse2
  #define SLEEF_COSF_STR STRINGIFY(Sleef_cosf4_u10sse2)
  #define SLEEF_TANF Sleef_tanf4_u10sse2
  #define SLEEF_TANF_STR STRINGIFY(Sleef_tanf4_u10sse2)
  #define SLEEF_EXPF Sleef_expf4_u10sse2
  #define SLEEF_EXPF_STR STRINGIFY(Sleef_expf4_u10sse2)
  #define SLEEF_LOGF Sleef_logf4_u10sse2
  #define SLEEF_LOGF_STR STRINGIFY(Sleef_logf4_u10sse2)
  #define SLEEF_TANHF Sleef_tanhf4_u10sse2
  #define SLEEF_TANHF_STR STRINGIFY(Sleef_tanhf4_u10sse2)

#elif defined(ENABLE_AVX2)
  #define SIMD_NAME "AVX2"
  #define VDOUBLE Sleef___m256d_2
  #define VFLOAT Sleef___m256_2
  typedef __m256d vdouble;
  typedef __m256 vfloat;
  #define VDOUBLE_LANES 4
  #define VFLOAT_LANES 8
  #define VLOAD_DOUBLE(p) _mm256_loadu_pd(p)
  #define VLOAD_FLOAT(p) _mm256_loadu_ps(p)
  #define VSTORE_DOUBLE(p, v) _mm256_storeu_pd(p, v)
  #define VSTORE_FLOAT(p, v) _mm256_storeu_ps(p, v)
  #define SLEEF_SIN Sleef_sind4_u10avx2
  #define SLEEF_SIN_STR STRINGIFY(Sleef_sind4_u10avx2)
  #define SLEEF_COS Sleef_cosd4_u10avx2
  #define SLEEF_COS_STR STRINGIFY(Sleef_cosd4_u10avx2)
  #define SLEEF_TAN Sleef_tand4_u10avx2
  #define SLEEF_TAN_STR STRINGIFY(Sleef_tand4_u10avx2)
  #define SLEEF_EXP Sleef_expd4_u10avx2
  #define SLEEF_EXP_STR STRINGIFY(Sleef_expd4_u10avx2)
  #define SLEEF_LOG Sleef_logd4_u10avx2
  #define SLEEF_LOG_STR STRINGIFY(Sleef_logd4_u10avx2)
  #define SLEEF_POW Sleef_powd4_u10avx2
  #define SLEEF_SQRT Sleef_sqrtd4_u05avx2
  #define SLEEF_SQRT_STR STRINGIFY(Sleef_sqrtd4_u05avx2)
  #define SLEEF_TANH Sleef_tanhd4_u10avx2
  #define SLEEF_TANH_STR STRINGIFY(Sleef_tanhd4_u10avx2)
  #define SLEEF_SINF Sleef_sinf8_u10avx2
  #define SLEEF_SINF_STR STRINGIFY(Sleef_sinf8_u10avx2)
  #define SLEEF_COSF Sleef_cosf8_u10avx2
  #define SLEEF_COSF_STR STRINGIFY(Sleef_cosf8_u10avx2)
  #define SLEEF_TANF Sleef_tanf8_u10avx2
  #define SLEEF_TANF_STR STRINGIFY(Sleef_tanf8_u10avx2)
  #define SLEEF_EXPF Sleef_expf8_u10avx2
  #define SLEEF_EXPF_STR STRINGIFY(Sleef_expf8_u10avx2)
  #define SLEEF_LOGF Sleef_logf8_u10avx2
  #define SLEEF_LOGF_STR STRINGIFY(Sleef_logf8_u10avx2)
  #define SLEEF_TANHF Sleef_tanhf8_u10avx2
  #define SLEEF_TANHF_STR STRINGIFY(Sleef_tanhf8_u10avx2)

#elif defined(ENABLE_AVX512F)
  #define SIMD_NAME "AVX512F"
  #define VDOUBLE Sleef___m512d_2
  #define VFLOAT Sleef___m512_2
  typedef __m512d vdouble;
  typedef __m512 vfloat;
  #define VDOUBLE_LANES 8
  #define VFLOAT_LANES 16
  #define VLOAD_DOUBLE(p) _mm512_loadu_pd(p)
  #define VLOAD_FLOAT(p) _mm512_loadu_ps(p)
  #define VSTORE_DOUBLE(p, v) _mm512_storeu_pd(p, v)
  #define VSTORE_FLOAT(p, v) _mm512_storeu_ps(p, v)
  #define SLEEF_SIN Sleef_sind8_u10avx512f
  #define SLEEF_COS Sleef_cosd8_u10avx512f
  #define SLEEF_TAN Sleef_tand8_u10avx512f
  #define SLEEF_EXP Sleef_expd8_u10avx512f
  #define SLEEF_LOG Sleef_logd8_u10avx512f
  #define SLEEF_POW Sleef_powd8_u10avx512f
  #define SLEEF_SQRT Sleef_sqrtd8_u05avx512f
  #define SLEEF_TANH Sleef_tanhd8_u10avx512f
  #define SLEEF_SINF Sleef_sinf16_u10avx512f
  #define SLEEF_COSF Sleef_cosf16_u10avx512f
  #define SLEEF_TANF Sleef_tanf16_u10avx512f
  #define SLEEF_EXPF Sleef_expf16_u10avx512f
  #define SLEEF_LOGF Sleef_logf16_u10avx512f
  #define SLEEF_TANHF Sleef_tanhf16_u10avx512f

#elif defined(ENABLE_ADVSIMD)
  #define SIMD_NAME "ADVSIMD"
  #define VDOUBLE Sleef_float64x2_t_2
  #define VFLOAT Sleef_float32x4_t_2
  typedef float64x2_t vdouble;
  typedef float32x4_t vfloat;
  #define VDOUBLE_LANES 2
  #define VFLOAT_LANES 4
  #define VLOAD_DOUBLE(p) vld1q_f64(p)
  #define VLOAD_FLOAT(p) vld1q_f32(p)
  #define VSTORE_DOUBLE(p, v) vst1q_f64(p, v)
  #define VSTORE_FLOAT(p, v) vst1q_f32(p, v)
  #define SLEEF_SIN Sleef_sind2_u10advsimd
  #define SLEEF_COS Sleef_cosd2_u10advsimd
  #define SLEEF_TAN Sleef_tand2_u10advsimd
  #define SLEEF_EXP Sleef_expd2_u10advsimd
  #define SLEEF_LOG Sleef_logd2_u10advsimd
  #define SLEEF_POW Sleef_powd2_u10advsimd
  #define SLEEF_SQRT Sleef_sqrtd2_u05advsimd
  #define SLEEF_TANH Sleef_tanhd2_u10advsimd
  #define SLEEF_SINF Sleef_sinf4_u10advsimd
  #define SLEEF_COSF Sleef_cosf4_u10advsimd
  #define SLEEF_TANF Sleef_tanf4_u10advsimd
  #define SLEEF_EXPF Sleef_expf4_u10advsimd
  #define SLEEF_LOGF Sleef_logf4_u10advsimd
  #define SLEEF_TANHF Sleef_tanhf4_u10advsimd

#elif defined(ENABLE_SVE)
  #define SIMD_NAME "SVE"
  typedef svfloat64_t vdouble;
  typedef svfloat32_t vfloat;
  #define VDOUBLE_LANES svcntd()
  #define VFLOAT_LANES svcntw()
  #define SLEEF_SIN Sleef_sindx_u10sve
  #define SLEEF_COS Sleef_cosdx_u10sve
  #define SLEEF_TAN Sleef_tandx_u10sve
  #define SLEEF_EXP Sleef_expdx_u10sve
  #define SLEEF_LOG Sleef_logdx_u10sve
  #define SLEEF_POW Sleef_powdx_u10sve
  #define SLEEF_SQRT Sleef_sqrtdx_u05sve
  #define SLEEF_SINF Sleef_sinfx_u10sve
  #define SLEEF_COSF Sleef_cosfx_u10sve
  #define SLEEF_TANF Sleef_tanfx_u10sve
  #define SLEEF_EXPF Sleef_expfx_u10sve
  #define SLEEF_LOGF Sleef_logfx_u10sve
  #define SLEEF_TANH Sleef_tanhdx_u10sve
  #define SLEEF_TANHF Sleef_tanhfx_u10sve
  #define USE_SVE 1

#elif defined(ENABLE_RVVM1)
  #define SIMD_NAME "RVV (LMUL=1)"
  #define USE_RVV 1
  #include <riscv_vector.h>
  typedef vfloat64m1_t vdouble;
  typedef vfloat32m1_t vfloat;
  #define SLEEF_SIN Sleef_sindx_u10rvvm1
  #define SLEEF_COS Sleef_cosdx_u10rvvm1
  #define SLEEF_TAN Sleef_tandx_u10rvvm1
  #define SLEEF_EXP Sleef_expdx_u10rvvm1
  #define SLEEF_LOG Sleef_logdx_u10rvvm1
  #define SLEEF_POW Sleef_powdx_u10rvvm1
  #define SLEEF_SQRT Sleef_sqrtdx_u05rvvm1
  #define SLEEF_SINF Sleef_sinfx_u10rvvm1
  #define SLEEF_COSF Sleef_cosfx_u10rvvm1
  #define SLEEF_TANF Sleef_tanfx_u10rvvm1
  #define SLEEF_EXPF Sleef_expfx_u10rvvm1
  #define SLEEF_LOGF Sleef_logfx_u10rvvm1
  #define SLEEF_TANH Sleef_tanhdx_u10rvvm1
  #define SLEEF_TANHF Sleef_tanhfx_u10rvvm1

#elif defined(ENABLE_RVVM2)
  #define SIMD_NAME "RVV (LMUL=2)"
  #define USE_RVV 1
  #include <riscv_vector.h>
  typedef vfloat64m2_t vdouble;
  typedef vfloat32m2_t vfloat;
  #define SLEEF_SIN Sleef_sindx_u10rvvm2
  #define SLEEF_COS Sleef_cosdx_u10rvvm2
  #define SLEEF_TAN Sleef_tandx_u10rvvm2
  #define SLEEF_EXP Sleef_expdx_u10rvvm2
  #define SLEEF_LOG Sleef_logdx_u10rvvm2
  #define SLEEF_POW Sleef_powdx_u10rvvm2
  #define SLEEF_SQRT Sleef_sqrtdx_u05rvvm2
  #define SLEEF_SINF Sleef_sinfx_u10rvvm2
  #define SLEEF_COSF Sleef_cosfx_u10rvvm2
  #define SLEEF_TANF Sleef_tanfx_u10rvvm2
  #define SLEEF_EXPF Sleef_expfx_u10rvvm2
  #define SLEEF_LOGF Sleef_logfx_u10rvvm2
  #define SLEEF_TANH Sleef_tanhdx_u10rvvm2
  #define SLEEF_TANHF Sleef_tanhfx_u10rvvm2

#elif defined(ENABLE_VSX)
  #define SIMD_NAME "VSX"
  typedef SLEEF_VECTOR_DOUBLE vdouble;
  typedef SLEEF_VECTOR_FLOAT vfloat;
  #define VDOUBLE_LANES 2
  #define VFLOAT_LANES 4
  #define SLEEF_SIN Sleef_sind2_u10vsx
  #define SLEEF_COS Sleef_cosd2_u10vsx
  #define SLEEF_TAN Sleef_tand2_u10vsx
  #define SLEEF_EXP Sleef_expd2_u10vsx
  #define SLEEF_LOG Sleef_logd2_u10vsx
  #define SLEEF_POW Sleef_powd2_u10vsx
  #define SLEEF_SQRT Sleef_sqrtd2_u05vsx
  #define SLEEF_SINF Sleef_sinf4_u10vsx
  #define SLEEF_COSF Sleef_cosf4_u10vsx
  #define SLEEF_TANF Sleef_tanf4_u10vsx
  #define SLEEF_EXPF Sleef_expf4_u10vsx
  #define SLEEF_LOGF Sleef_logf4_u10vsx
  #define SLEEF_TANH Sleef_tanhd2_u10vsx
  #define SLEEF_TANHF Sleef_tanhf4_u10vsx

#elif defined(ENABLE_VXE)
  #define SIMD_NAME "VXE"
  typedef SLEEF_VECTOR_DOUBLE vdouble;
  typedef SLEEF_VECTOR_FLOAT vfloat;
  #define VDOUBLE_LANES 2
  #define VFLOAT_LANES 4
  #define SLEEF_SIN Sleef_sind2_u10vxe
  #define SLEEF_COS Sleef_cosd2_u10vxe
  #define SLEEF_TAN Sleef_tand2_u10vxe
  #define SLEEF_EXP Sleef_expd2_u10vxe
  #define SLEEF_LOG Sleef_logd2_u10vxe
  #define SLEEF_POW Sleef_powd2_u10vxe
  #define SLEEF_SQRT Sleef_sqrtd2_u05vxe
  #define SLEEF_SINF Sleef_sinf4_u10vxe
  #define SLEEF_COSF Sleef_cosf4_u10vxe
  #define SLEEF_TANF Sleef_tanf4_u10vxe
  #define SLEEF_EXPF Sleef_expf4_u10vxe
  #define SLEEF_LOGF Sleef_logf4_u10vxe
  #define SLEEF_TANH Sleef_tanhd2_u10vxe
  #define SLEEF_TANHF Sleef_tanhf4_u10vxe

#else
  #error "No SIMD extension defined"
#endif

#ifndef BENCHMARK_ITERATIONS
#define BENCHMARK_ITERATIONS 10000000
#endif

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 1000000
#endif

// ============================================================================
// Timing utilities
// ============================================================================

static double get_time_sec(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ============================================================================
// Random number generation
// ============================================================================

static uint64_t xorshift_state = 0x123456789abcdefULL;

static double rand_double(double min, double max) {
  xorshift_state ^= xorshift_state << 13;
  xorshift_state ^= xorshift_state >> 7;
  xorshift_state ^= xorshift_state << 17;
  double t = (double)(xorshift_state >> 11) / (double)(1ULL << 53);
  return min + t * (max - min);
}

static float rand_float(float min, float max) {
  xorshift_state ^= xorshift_state << 13;
  xorshift_state ^= xorshift_state >> 7;
  xorshift_state ^= xorshift_state << 17;
  float t = (float)(xorshift_state >> 11) / (float)(1ULL << 53);
  return min + t * (max - min);
}

// ============================================================================
// Benchmark macros
// ============================================================================

#if defined(USE_RVV)
  #if defined(ENABLE_RVVM1)
    #define VSETVL_D(n) __riscv_vsetvl_e64m1(n)
    #define VSETVL_F(n) __riscv_vsetvl_e32m1(n)
    #define VLE_D(p, vl) __riscv_vle64_v_f64m1(p, vl)
    #define VLE_F(p, vl) __riscv_vle32_v_f32m1(p, vl)
    #define VSE_D(p, v, vl) __riscv_vse64_v_f64m1(p, v, vl)
    #define VSE_F(p, v, vl) __riscv_vse32_v_f32m1(p, v, vl)
  #elif defined(ENABLE_RVVM2)
    #define VSETVL_D(n) __riscv_vsetvl_e64m2(n)
    #define VSETVL_F(n) __riscv_vsetvl_e32m2(n)
    #define VLE_D(p, vl) __riscv_vle64_v_f64m2(p, vl)
    #define VLE_F(p, vl) __riscv_vle32_v_f32m2(p, vl)
    #define VSE_D(p, v, vl) __riscv_vse64_v_f64m2(p, v, vl)
    #define VSE_F(p, v, vl) __riscv_vse32_v_f32m2(p, v, vl)
  #endif
  
  #define BENCHMARK_SIMD_1ARG_D(func, func_str, min_val, max_val, iterations, vector_size) do { \
    double *input = (double *)aligned_alloc(64, vector_size * sizeof(double)); \
    double *output = (double *)aligned_alloc(64, vector_size * sizeof(double)); \
    for (size_t i = 0; i < vector_size; i++) { \
      input[i] = rand_double(min_val, max_val); \
    } \
    double start = get_time_sec(); \
    for (uint64_t iter = 0; iter < iterations; iter++) { \
      size_t vl; \
      for (size_t i = 0; i < vector_size; ) { \
        vl = VSETVL_D(vector_size - i); \
        vdouble v = VLE_D(input + i, vl); \
        vdouble r = func(v); \
        VSE_D(output + i, r, vl); \
        i += vl; \
      } \
    } \
    double elapsed = get_time_sec() - start; \
    double sum = 0; \
    for (size_t i = 0; i < vector_size; i++) sum += output[i]; \
    printf("%-35s: %10.3f ns/element  (sum=%g)\n", func_str, \
           elapsed * 1e9 / (iterations * vector_size), sum); \
    free(input); \
    free(output); \
  } while(0)

  #define BENCHMARK_SIMD_1ARG_F(func, func_str, min_val, max_val, iterations, vector_size) do { \
    float *input = (float *)aligned_alloc(64, vector_size * sizeof(float)); \
    float *output = (float *)aligned_alloc(64, vector_size * sizeof(float)); \
    for (size_t i = 0; i < vector_size; i++) { \
      input[i] = rand_float(min_val, max_val); \
    } \
    double start = get_time_sec(); \
    for (uint64_t iter = 0; iter < iterations; iter++) { \
      size_t vl; \
      for (size_t i = 0; i < vector_size; ) { \
        vl = VSETVL_F(vector_size - i); \
        vfloat v = VLE_F(input + i, vl); \
        vfloat r = func(v); \
        VSE_F(output + i, r, vl); \
        i += vl; \
      } \
    } \
    double elapsed = get_time_sec() - start; \
    float sum = 0; \
    for (size_t i = 0; i < vector_size; i++) sum += output[i]; \
    printf("%-35s: %10.3f ns/element  (sum=%g)\n", func_str, \
           elapsed * 1e9 / (iterations * vector_size), sum); \
    free(input); \
    free(output); \
  } while(0)

#elif defined(USE_SVE)
  #include <arm_sve.h>
  
  #define BENCHMARK_SIMD_1ARG_D(func, min_val, max_val, iterations, vector_size) do { \
    size_t lanes = VDOUBLE_LANES; \
    double *input = (double *)aligned_alloc(64, vector_size * sizeof(double)); \
    double *output = (double *)aligned_alloc(64, vector_size * sizeof(double)); \
    for (size_t i = 0; i < vector_size; i++) { \
      input[i] = rand_double(min_val, max_val); \
    } \
    double start = get_time_sec(); \
    for (uint64_t iter = 0; iter < iterations; iter++) { \
      for (size_t i = 0; i < vector_size; i += lanes) { \
        svbool_t pg = svwhilelt_b64(i, vector_size); \
        svfloat64_t v = svld1(pg, input + i); \
        svfloat64_t r = func(v); \
        svst1(pg, output + i, r); \
      } \
    } \
    double elapsed = get_time_sec() - start; \
    double sum = 0; \
    for (size_t i = 0; i < vector_size; i++) sum += output[i]; \
    printf("%-35s: %10.3f ns/element  (sum=%g)\n", #func, \
           elapsed * 1e9 / (iterations * vector_size), sum); \
    free(input); \
    free(output); \
  } while(0)

  #define BENCHMARK_SIMD_1ARG_F(func, min_val, max_val, iterations, vector_size) do { \
    size_t lanes = VFLOAT_LANES; \
    float *input = (float *)aligned_alloc(64, vector_size * sizeof(float)); \
    float *output = (float *)aligned_alloc(64, vector_size * sizeof(float)); \
    for (size_t i = 0; i < vector_size; i++) { \
      input[i] = rand_float(min_val, max_val); \
    } \
    double start = get_time_sec(); \
    for (uint64_t iter = 0; iter < iterations; iter++) { \
      for (size_t i = 0; i < vector_size; i += lanes) { \
        svbool_t pg = svwhilelt_b32(i, vector_size); \
        svfloat32_t v = svld1(pg, input + i); \
        svfloat32_t r = func(v); \
        svst1(pg, output + i, r); \
      } \
    } \
    double elapsed = get_time_sec() - start; \
    float sum = 0; \
    for (size_t i = 0; i < vector_size; i++) sum += output[i]; \
    printf("%-35s: %10.3f ns/element  (sum=%g)\n", #func, \
           elapsed * 1e9 / (iterations * vector_size), sum); \
    free(input); \
    free(output); \
  } while(0)

#else
  #define BENCHMARK_SIMD_1ARG_D(func, min_val, max_val, iterations, vector_size) do { \
    double *input = (double *)aligned_alloc(64, vector_size * sizeof(double)); \
    double *output = (double *)aligned_alloc(64, vector_size * sizeof(double)); \
    for (size_t i = 0; i < vector_size; i++) { \
      input[i] = rand_double(min_val, max_val); \
    } \
    double start = get_time_sec(); \
    for (uint64_t iter = 0; iter < iterations; iter++) { \
      for (size_t i = 0; i < vector_size; i += VDOUBLE_LANES) { \
        vdouble v = VLOAD_DOUBLE(input + i); \
        vdouble r = func(v); \
        VSTORE_DOUBLE(output + i, r); \
      } \
    } \
    double elapsed = get_time_sec() - start; \
    double sum = 0; \
    for (size_t i = 0; i < vector_size; i++) sum += output[i]; \
    printf("%-35s: %10.3f ns/element  (sum=%g)\n", #func, \
           elapsed * 1e9 / (iterations * vector_size), sum); \
    free(input); \
    free(output); \
  } while(0)

  #define BENCHMARK_SIMD_1ARG_F(func, min_val, max_val, iterations, vector_size) do { \
    float *input = (float *)aligned_alloc(64, vector_size * sizeof(float)); \
    float *output = (float *)aligned_alloc(64, vector_size * sizeof(float)); \
    for (size_t i = 0; i < vector_size; i++) { \
      input[i] = rand_float(min_val, max_val); \
    } \
    double start = get_time_sec(); \
    for (uint64_t iter = 0; iter < iterations; iter++) { \
      for (size_t i = 0; i < vector_size; i += VFLOAT_LANES) { \
        vfloat v = VLOAD_FLOAT(input + i); \
        vfloat r = func(v); \
        VSTORE_FLOAT(output + i, r); \
      } \
    } \
    double elapsed = get_time_sec(); \
    float sum = 0; \
    for (size_t i = 0; i < vector_size; i++) sum += output[i]; \
    printf("%-35s: %10.3f ns/element  (sum=%g)\n", #func, \
           elapsed * 1e9 / (iterations * vector_size), sum); \
    free(input); \
    free(output); \
  } while(0)
#endif

// ============================================================================
// Benchmark functions
// ============================================================================

void benchmark_simd_functions(uint64_t iterations, size_t vector_size) {
  printf("=================================================================\n");
  printf("SIMD Benchmark - %s\n", SIMD_NAME);
  printf("=================================================================\n");
#if defined(USE_RVV)
  printf("Vector length (double): runtime determined\n");
  printf("Vector length (float):  runtime determined\n");
#elif defined(USE_SVE)
  printf("Vector length (double): %zu\n", (size_t)VDOUBLE_LANES);
  printf("Vector length (float):  %zu\n", (size_t)VFLOAT_LANES);
#else
  printf("Vector length (double): %d\n", VDOUBLE_LANES);
  printf("Vector length (float):  %d\n", VFLOAT_LANES);
#endif
  printf("Iterations: %llu\n", (unsigned long long)iterations);
  printf("Vector size: %zu elements\n", vector_size);
  printf("=================================================================\n\n");

  printf("Double Precision Functions:\n");
  printf("-----------------------------------------------------------------\n");
  BENCHMARK_SIMD_1ARG_D(SLEEF_SIN, SLEEF_SIN_STR, 0.0, 6.28, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_D(SLEEF_COS, SLEEF_COS_STR, 0.0, 6.28, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_D(SLEEF_TAN, SLEEF_TAN_STR, 0.0, 6.28, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_D(SLEEF_EXP, SLEEF_EXP_STR, -700.0, 700.0, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_D(SLEEF_LOG, SLEEF_LOG_STR, 1.0, 1e300, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_D(SLEEF_SQRT, SLEEF_SQRT_STR, 0.0, 1e300, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_D(SLEEF_TANH, SLEEF_TANH_STR, -10.0, 10.0, iterations, vector_size);
  
  printf("\nSingle Precision Functions:\n");
  printf("-----------------------------------------------------------------\n");
  BENCHMARK_SIMD_1ARG_F(SLEEF_SINF, SLEEF_SINF_STR, 0.0f, 6.28f, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_F(SLEEF_COSF, SLEEF_COSF_STR, 0.0f, 6.28f, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_F(SLEEF_TANF, SLEEF_TANF_STR, 0.0f, 6.28f, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_F(SLEEF_EXPF, SLEEF_EXPF_STR, -100.0f, 100.0f, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_F(SLEEF_LOGF, SLEEF_LOGF_STR, 1.0f, 1e38f, iterations, vector_size);
  BENCHMARK_SIMD_1ARG_F(SLEEF_TANHF, SLEEF_TANHF_STR, -10.0f, 10.0f, iterations, vector_size);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  uint64_t iterations = BENCHMARK_ITERATIONS;
  size_t vector_size = VECTOR_SIZE;
  
  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      iterations = strtoull(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
      vector_size = strtoull(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "-h") == 0) {
      printf("Usage: %s [options]\n", argv[0]);
      printf("Options:\n");
      printf("  -i <iterations>  Number of iterations (default: %d)\n", BENCHMARK_ITERATIONS);
      printf("  -s <size>        Vector size (default: %d elements)\n", VECTOR_SIZE);
      printf("  -h               Show this help message\n");
      return 0;
    }
  }
  
  printf("\n");
  printf("=================================================================\n");
  printf("SLEEF SIMD Math Library Benchmark\n");
  printf("=================================================================\n\n");
  
  benchmark_simd_functions(iterations, vector_size);
  
  printf("\n=================================================================\n");
  printf("Benchmark Complete\n");
  printf("=================================================================\n\n");
  
  return 0;
}

