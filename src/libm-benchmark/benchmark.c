//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "sleef.h"

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
// Benchmark functions
// ============================================================================

typedef struct {
  const char *name;
  double min_arg;
  double max_arg;
} BenchmarkConfig;

#define BENCHMARK_SCALAR_1ARG(func, libm_func, min_val, max_val, iterations) do { \
  double sum = 0; \
  double start = get_time_sec(); \
  for (uint64_t i = 0; i < iterations; i++) { \
    double x = rand_double(min_val, max_val); \
    sum += func(x); \
  } \
  double elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #func, \
         elapsed * 1e9 / iterations, sum); \
  \
  sum = 0; \
  start = get_time_sec(); \
  for (uint64_t i = 0; i < iterations; i++) { \
    double x = rand_double(min_val, max_val); \
    sum += libm_func(x); \
  } \
  elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #libm_func " (reference)", \
         elapsed * 1e9 / iterations, sum); \
  printf("\n"); \
} while(0)

#define BENCHMARK_SCALAR_1ARG_F(func, libm_func, min_val, max_val, iterations) do { \
  float sum = 0; \
  double start = get_time_sec(); \
  for (uint64_t i = 0; i < iterations; i++) { \
    float x = rand_float(min_val, max_val); \
    sum += func(x); \
  } \
  double elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #func, \
         elapsed * 1e9 / iterations, sum); \
  \
  sum = 0; \
  start = get_time_sec(); \
  for (uint64_t i = 0; i < iterations; i++) { \
    float x = rand_float(min_val, max_val); \
    sum += libm_func(x); \
  } \
  elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #libm_func " (reference)", \
         elapsed * 1e9 / iterations, sum); \
  printf("\n"); \
} while(0)

#define BENCHMARK_SCALAR_2ARG(func, libm_func, min_val, max_val, iterations) do { \
  double sum = 0; \
  double start = get_time_sec(); \
  for (uint64_t i = 0; i < iterations; i++) { \
    double x = rand_double(min_val, max_val); \
    double y = rand_double(min_val, max_val); \
    sum += func(x, y); \
  } \
  double elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #func, \
         elapsed * 1e9 / iterations, sum); \
  \
  sum = 0; \
  start = get_time_sec(); \
  for (uint64_t i = 0; i < iterations; i++) { \
    double x = rand_double(min_val, max_val); \
    double y = rand_double(min_val, max_val); \
    sum += libm_func(x, y); \
  } \
  elapsed = get_time_sec() - start; \
  printf("%-30s: %10.3f ns/call  (sum=%g)\n", #libm_func " (reference)", \
         elapsed * 1e9 / iterations, sum); \
  printf("\n"); \
} while(0)

void benchmark_trig_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Trigonometric Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  BENCHMARK_SCALAR_1ARG(Sleef_sin_u10, sin, 0.0, 6.28, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_sin_u35, sin, 0.0, 6.28, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_cos_u10, cos, 0.0, 6.28, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_cos_u35, cos, 0.0, 6.28, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_tan_u10, tan, 0.0, 6.28, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_tan_u35, tan, 0.0, 6.28, iterations);
}

void benchmark_trig_functions_f(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Trigonometric Functions (Single Precision)\n");
  printf("=================================================================\n\n");
  
  BENCHMARK_SCALAR_1ARG_F(Sleef_sinf_u10, sinf, 0.0f, 6.28f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_sinf_u35, sinf, 0.0f, 6.28f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_cosf_u10, cosf, 0.0f, 6.28f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_cosf_u35, cosf, 0.0f, 6.28f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_tanf_u10, tanf, 0.0f, 6.28f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_tanf_u35, tanf, 0.0f, 6.28f, iterations);
}

void benchmark_exp_log_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Exponential and Logarithm Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  BENCHMARK_SCALAR_1ARG(Sleef_exp_u10, exp, -700.0, 700.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_log_u10, log, 1.0, 1e300, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_log_u35, log, 1.0, 1e300, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_log10_u10, log10, 1.0, 1e300, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_log2_u10, log2, 1.0, 1e300, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_exp2_u10, exp2, -1000.0, 1000.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_exp10_u10, exp10, -300.0, 300.0, iterations);
}

void benchmark_exp_log_functions_f(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Exponential and Logarithm Functions (Single Precision)\n");
  printf("=================================================================\n\n");
  
  BENCHMARK_SCALAR_1ARG_F(Sleef_expf_u10, expf, -100.0f, 100.0f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_logf_u10, logf, 1.0f, 1e38f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_logf_u35, logf, 1.0f, 1e38f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_log10f_u10, log10f, 1.0f, 1e38f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_log2f_u10, log2f, 1.0f, 1e38f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_exp2f_u10, exp2f, -100.0f, 100.0f, iterations);
  BENCHMARK_SCALAR_1ARG_F(Sleef_exp10f_u10, exp10f, -38.0f, 38.0f, iterations);
}

void benchmark_power_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Power Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  BENCHMARK_SCALAR_2ARG(Sleef_pow_u10, pow, -30.0, 30.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_sqrt_u05, sqrt, 0.0, 1e300, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_sqrt_u35, sqrt, 0.0, 1e300, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_cbrt_u10, cbrt, -1e100, 1e100, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_cbrt_u35, cbrt, -1e100, 1e100, iterations);
}

void benchmark_inverse_trig_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Inverse Trigonometric Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  BENCHMARK_SCALAR_1ARG(Sleef_asin_u10, asin, -1.0, 1.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_asin_u35, asin, -1.0, 1.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_acos_u10, acos, -1.0, 1.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_acos_u35, acos, -1.0, 1.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_atan_u10, atan, -10.0, 10.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_atan_u35, atan, -10.0, 10.0, iterations);
  BENCHMARK_SCALAR_2ARG(Sleef_atan2_u10, atan2, -10.0, 10.0, iterations);
  BENCHMARK_SCALAR_2ARG(Sleef_atan2_u35, atan2, -10.0, 10.0, iterations);
}

void benchmark_hyperbolic_functions(uint64_t iterations) {
  printf("=================================================================\n");
  printf("Hyperbolic Functions (Double Precision)\n");
  printf("=================================================================\n\n");
  
  BENCHMARK_SCALAR_1ARG(Sleef_sinh_u10, sinh, -700.0, 700.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_sinh_u35, sinh, -700.0, 700.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_cosh_u10, cosh, -700.0, 700.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_cosh_u35, cosh, -700.0, 700.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_tanh_u10, tanh, -10.0, 10.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_tanh_u35, tanh, -10.0, 10.0, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_asinh_u10, asinh, -1e300, 1e300, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_acosh_u10, acosh, 1.0, 1e300, iterations);
  BENCHMARK_SCALAR_1ARG(Sleef_atanh_u10, atanh, -1.0, 1.0, iterations);
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  printf("Options:\n");
  printf("  -i <iterations>  Number of iterations (default: %d)\n", BENCHMARK_ITERATIONS);
  printf("  -h               Show this help message\n");
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
    } else if (strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
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
  printf("=================================================================\n\n");
  
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

