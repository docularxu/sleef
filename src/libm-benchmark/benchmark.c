//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "sleef.h"

// Global toggle to control inclusion of u35 variants at runtime.
// Default: include u35. Can be disabled via --no-u35 / --match-simd.
static int g_include_u35 = 1;
static size_t g_pool_size = 1000000; // default input pool size for scalar
static uint64_t g_seed = 0;          // 0 = default seed

// Number of iterations for each benchmark
#ifndef BENCHMARK_ITERATIONS
#define BENCHMARK_ITERATIONS 100000000
#endif

// Vector size for SIMD operations
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

static void srand_xorshift(uint64_t seed) {
  if (seed != 0) xorshift_state = seed;
}

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
// Input pool helpers (allocate, fill, free)
// ============================================================================

static double *alloc_fill_double(size_t n, double minv, double maxv) {
  double *buf = (double *)malloc(sizeof(double) * n);
  if (!buf) return NULL;
  for (size_t i = 0; i < n; i++) buf[i] = rand_double(minv, maxv);
  return buf;
}

static float *alloc_fill_float(size_t n, float minv, float maxv) {
  float *buf = (float *)malloc(sizeof(float) * n);
  if (!buf) return NULL;
  for (size_t i = 0; i < n; i++) buf[i] = rand_float(minv, maxv);
  return buf;
}

// ============================================================================
// Benchmark functions (scalar) - use pre-generated pools
// ============================================================================

typedef struct {
  const char *name;
  double min_arg;
  double max_arg;
} BenchmarkConfig;

#define RUN_SCALAR_1ARG_D(func, libm_func, pool, pooln, iterations) do { \
  double sum = 0; \
  double start = get_time_sec(); \
  for (uint64_t i = 0; i < (iterations); i++) { \
    double x = (pool)[i % (pooln)]; \
    sum += func(x); \
  } \
  double elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #func, elapsed * 1e9 / (iterations), sum); \
  sum = 0; \
  start = get_time_sec(); \
  for (uint64_t i = 0; i < (iterations); i++) { \
    double x = (pool)[i % (pooln)]; \
    sum += libm_func(x); \
  } \
  elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #libm_func " (reference)", elapsed * 1e9 / (iterations), sum); \
  printf("\n"); \
} while(0)

#define RUN_SCALAR_1ARG_F(func, libm_func, pool, pooln, iterations) do { \
  float sum = 0; \
  double start = get_time_sec(); \
  for (uint64_t i = 0; i < (iterations); i++) { \
    float x = (pool)[i % (pooln)]; \
    sum += func(x); \
  } \
  double elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #func, elapsed * 1e9 / (iterations), sum); \
  sum = 0; \
  start = get_time_sec(); \
  for (uint64_t i = 0; i < (iterations); i++) { \
    float x = (pool)[i % (pooln)]; \
    sum += libm_func(x); \
  } \
  elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #libm_func " (reference)", elapsed * 1e9 / (iterations), sum); \
  printf("\n"); \
} while(0)

#define RUN_SCALAR_2ARG_D(func, libm_func, poolx, pooly, pooln, iterations) do { \
  double sum = 0; \
  double start = get_time_sec(); \
  for (uint64_t i = 0; i < (iterations); i++) { \
    double x = (poolx)[i % (pooln)]; \
    double y = (pooly)[i % (pooln)]; \
    sum += func(x, y); \
  } \
  double elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #func, elapsed * 1e9 / (iterations), sum); \
  sum = 0; \
  start = get_time_sec(); \
  for (uint64_t i = 0; i < (iterations); i++) { \
    double x = (poolx)[i % (pooln)]; \
    double y = (pooly)[i % (pooln)]; \
    sum += libm_func(x, y); \
  } \
  elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #libm_func " (reference)", elapsed * 1e9 / (iterations), sum); \
  printf("\n"); \
} while(0)

void benchmark_trig_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Trigonometric Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  double *pool = alloc_fill_double(g_pool_size, 0.0, 6.28);
  if (!pool) { fprintf(stderr, "alloc failed\n"); return; }
  RUN_SCALAR_1ARG_D(Sleef_sin_u10, sin, pool, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_sin_u35, sin, pool, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_cos_u10, cos, pool, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_cos_u35, cos, pool, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_tan_u10, tan, pool, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_tan_u35, tan, pool, g_pool_size, iterations);
  free(pool); pool = NULL;
}

void benchmark_trig_functions_f(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Trigonometric Functions (Single Precision)\n");
  printf("=================================================================\n\n");
  
  float *pool = alloc_fill_float(g_pool_size, 0.0f, 6.28f);
  if (!pool) { fprintf(stderr, "alloc failed\n"); return; }
  RUN_SCALAR_1ARG_F(Sleef_sinf_u10, sinf, pool, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_F(Sleef_sinf_u35, sinf, pool, g_pool_size, iterations);
  RUN_SCALAR_1ARG_F(Sleef_cosf_u10, cosf, pool, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_F(Sleef_cosf_u35, cosf, pool, g_pool_size, iterations);
  RUN_SCALAR_1ARG_F(Sleef_tanf_u10, tanf, pool, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_F(Sleef_tanf_u35, tanf, pool, g_pool_size, iterations);
  free(pool); pool = NULL;
}

void benchmark_exp_log_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Exponential and Logarithm Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  double *pd_exp = alloc_fill_double(g_pool_size, -700.0, 700.0);
  double *pd_ln  = alloc_fill_double(g_pool_size, 1.0, 1e300);
  if (!pd_exp || !pd_ln) { fprintf(stderr, "alloc failed\n"); free(pd_exp); free(pd_ln); return; }
  RUN_SCALAR_1ARG_D(Sleef_exp_u10, exp, pd_exp, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_log_u10, log, pd_ln, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_log_u35, log, pd_ln, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_log10_u10, log10, pd_ln, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_log2_u10, log2, pd_ln, g_pool_size, iterations);
  double *pd_exp2 = alloc_fill_double(g_pool_size, -1000.0, 1000.0);
  if (!pd_exp2) { fprintf(stderr, "alloc failed\n"); free(pd_exp); free(pd_ln); return; }
  RUN_SCALAR_1ARG_D(Sleef_exp2_u10, exp2, pd_exp2, g_pool_size, iterations);
  double *pd_exp10 = alloc_fill_double(g_pool_size, -300.0, 300.0);
  if (!pd_exp10) { fprintf(stderr, "alloc failed\n"); free(pd_exp); free(pd_ln); free(pd_exp2); return; }
  RUN_SCALAR_1ARG_D(Sleef_exp10_u10, exp10, pd_exp10, g_pool_size, iterations);
  free(pd_exp10); free(pd_exp2); free(pd_ln); free(pd_exp);
}

void benchmark_exp_log_functions_f(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Exponential and Logarithm Functions (Single Precision)\n");
  printf("=================================================================\n\n");
  
  float *pf_exp = alloc_fill_float(g_pool_size, -100.0f, 100.0f);
  float *pf_ln  = alloc_fill_float(g_pool_size, 1.0f, 1e38f);
  if (!pf_exp || !pf_ln) { fprintf(stderr, "alloc failed\n"); free(pf_exp); free(pf_ln); return; }
  RUN_SCALAR_1ARG_F(Sleef_expf_u10, expf, pf_exp, g_pool_size, iterations);
  RUN_SCALAR_1ARG_F(Sleef_logf_u10, logf, pf_ln, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_F(Sleef_logf_u35, logf, pf_ln, g_pool_size, iterations);
  RUN_SCALAR_1ARG_F(Sleef_log10f_u10, log10f, pf_ln, g_pool_size, iterations);
  RUN_SCALAR_1ARG_F(Sleef_log2f_u10, log2f, pf_ln, g_pool_size, iterations);
  float *pf_exp2 = alloc_fill_float(g_pool_size, -100.0f, 100.0f);
  if (!pf_exp2) { fprintf(stderr, "alloc failed\n"); free(pf_exp); free(pf_ln); return; }
  RUN_SCALAR_1ARG_F(Sleef_exp2f_u10, exp2f, pf_exp2, g_pool_size, iterations);
  float *pf_exp10 = alloc_fill_float(g_pool_size, -38.0f, 38.0f);
  if (!pf_exp10) { fprintf(stderr, "alloc failed\n"); free(pf_exp); free(pf_ln); free(pf_exp2); return; }
  RUN_SCALAR_1ARG_F(Sleef_exp10f_u10, exp10f, pf_exp10, g_pool_size, iterations);
  free(pf_exp10); free(pf_exp2); free(pf_ln); free(pf_exp);
}

void benchmark_power_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Power Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  double *pd_powx = alloc_fill_double(g_pool_size, -30.0, 30.0);
  double *pd_powy = alloc_fill_double(g_pool_size, -30.0, 30.0);
  double *pd_sqrt = alloc_fill_double(g_pool_size, 0.0, 1e300);
  double *pd_cbrt = alloc_fill_double(g_pool_size, -1e100, 1e100);
  if (!pd_powx || !pd_powy || !pd_sqrt || !pd_cbrt) { fprintf(stderr, "alloc failed\n"); free(pd_powx); free(pd_powy); free(pd_sqrt); free(pd_cbrt); return; }
  RUN_SCALAR_2ARG_D(Sleef_pow_u10, pow, pd_powx, pd_powy, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_sqrt_u05, sqrt, pd_sqrt, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_sqrt_u35, sqrt, pd_sqrt, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_cbrt_u10, cbrt, pd_cbrt, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_cbrt_u35, cbrt, pd_cbrt, g_pool_size, iterations);
  free(pd_cbrt); free(pd_sqrt); free(pd_powy); free(pd_powx);
}

void benchmark_inverse_trig_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Inverse Trigonometric Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  double *pd_unit = alloc_fill_double(g_pool_size, -1.0, 1.0);
  double *pd_tan  = alloc_fill_double(g_pool_size, -10.0, 10.0);
  if (!pd_unit || !pd_tan) { fprintf(stderr, "alloc failed\n"); free(pd_unit); free(pd_tan); return; }
  RUN_SCALAR_1ARG_D(Sleef_asin_u10, asin, pd_unit, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_asin_u35, asin, pd_unit, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_acos_u10, acos, pd_unit, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_acos_u35, acos, pd_unit, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_atan_u10, atan, pd_tan, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_atan_u35, atan, pd_tan, g_pool_size, iterations);
  RUN_SCALAR_2ARG_D(Sleef_atan2_u10, atan2, pd_tan, pd_tan, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_2ARG_D(Sleef_atan2_u35, atan2, pd_tan, pd_tan, g_pool_size, iterations);
  free(pd_tan); free(pd_unit);
}

void benchmark_hyperbolic_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Hyperbolic Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  double *pd_huge = alloc_fill_double(g_pool_size, -700.0, 700.0);
  double *pd_unit = alloc_fill_double(g_pool_size, -1.0, 1.0);
  double *pd_pos  = alloc_fill_double(g_pool_size, 1.0, 1e300);
  double *pd_tanh = alloc_fill_double(g_pool_size, -10.0, 10.0);
  if (!pd_huge || !pd_unit || !pd_pos || !pd_tanh) {
    fprintf(stderr, "alloc failed\n");
    free(pd_huge); free(pd_unit); free(pd_pos); free(pd_tanh);
    return;
  }
  RUN_SCALAR_1ARG_D(Sleef_sinh_u10, sinh, pd_huge, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_sinh_u35, sinh, pd_huge, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_cosh_u10, cosh, pd_huge, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_cosh_u35, cosh, pd_huge, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_tanh_u10, tanh, pd_tanh, g_pool_size, iterations);
  if (g_include_u35) RUN_SCALAR_1ARG_D(Sleef_tanh_u35, tanh, pd_tanh, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_asinh_u10, asinh, pd_pos, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_acosh_u10, acosh, pd_pos, g_pool_size, iterations);
  RUN_SCALAR_1ARG_D(Sleef_atanh_u10, atanh, pd_unit, g_pool_size, iterations);
  free(pd_tanh); free(pd_pos); free(pd_unit); free(pd_huge);
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  printf("Options:\n");
  printf("  -i <iterations>  Number of iterations (default: %d)\n", BENCHMARK_ITERATIONS);
  printf("  -psz <pool_size> Input pool size for scalar (default: %zu)\n", (size_t)1000000);
  printf("  --seed <value>   RNG seed for reproducible pools\n");
  printf("  -h               Show this help message\n");
  printf("  --no-u35         Disable u35 variant benchmarks (match SIMD variants)\n");
  printf("  --match-simd     Alias of --no-u35; keep only variants used by SIMD\n");
  printf("\n");
  printf("Benchmark categories:\n");
  printf("  trig             Trigonometric functions\n");
  printf("  exp              Exponential and logarithm functions\n");
  printf("  pow              Power functions\n");
  printf("  invtrig          Inverse trigonometric functions\n");
  printf("  hyp              Hyperbolic functions\n");
  printf("  all              Run all benchmarks (default)\n");
}

int main(int argc, char **argv) {
  uint64_t iterations = BENCHMARK_ITERATIONS;
  int run_all = 1;
  int run_trig = 0;
  int run_exp = 0;
  int run_pow = 0;
  int run_invtrig = 0;
  int run_hyp = 0;
  
  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      iterations = strtoull(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "-psz") == 0 && i + 1 < argc) {
      g_pool_size = strtoull(argv[++i], NULL, 10);
      if (g_pool_size == 0) g_pool_size = 1;
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      g_seed = strtoull(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else if (strcmp(argv[i], "--no-u35") == 0 || strcmp(argv[i], "--match-simd") == 0) {
      g_include_u35 = 0;
    } else if (strcmp(argv[i], "trig") == 0) {
      run_trig = 1;
      run_all = 0;
    } else if (strcmp(argv[i], "exp") == 0) {
      run_exp = 1;
      run_all = 0;
    } else if (strcmp(argv[i], "pow") == 0) {
      run_pow = 1;
      run_all = 0;
    } else if (strcmp(argv[i], "invtrig") == 0) {
      run_invtrig = 1;
      run_all = 0;
    } else if (strcmp(argv[i], "hyp") == 0) {
      run_hyp = 1;
      run_all = 0;
    } else if (strcmp(argv[i], "all") == 0) {
      run_all = 1;
    }
  }
  
  printf("\n");
  printf("=================================================================\n");
  printf("SLEEF Math Library Benchmark\n");
  printf("=================================================================\n");
  printf("Iterations per function: %llu\n", (unsigned long long)iterations);
  printf("Scalar input pool size: %zu\n", g_pool_size);
  printf("=================================================================\n\n");

  // Initialize RNG seed if requested
  if (g_seed) srand_xorshift(g_seed);
  
  if (run_all || run_trig) {
    benchmark_trig_functions(iterations);
    benchmark_trig_functions_f(iterations);
  }
  
  if (run_all || run_exp) {
    benchmark_exp_log_functions(iterations);
    benchmark_exp_log_functions_f(iterations);
  }
  
  if (run_all || run_pow) {
    benchmark_power_functions(iterations);
  }
  
  if (run_all || run_invtrig) {
    benchmark_inverse_trig_functions(iterations);
  }
  
  if (run_all || run_hyp) {
    benchmark_hyperbolic_functions(iterations);
  }
  
  printf("=================================================================\n");
  printf("Benchmark Complete\n");
  printf("=================================================================\n\n");
  
  return 0;
}

