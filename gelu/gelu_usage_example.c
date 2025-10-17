// Simple example showing how to use the GELU kernel
//
// Compile and run:
//   gcc -O3 -mavx2 -mfma -DENABLE_AVX2 gelu_kernel.c gelu_usage_example.c -lsleef -lm -o example
//   ./example

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gelu_kernel.h"

int main(void) {
    // Example 1: Simple usage
    printf("Example 1: Simple GELU computation\n");
    printf("===================================\n");
    
    const size_t n = 10;
    double *input = aligned_alloc(64, n * sizeof(double));
    double *output = aligned_alloc(64, n * sizeof(double));
    
    // Initialize some test values
    for (size_t i = 0; i < n; i++) {
        input[i] = -4.0 + 8.0 * i / (n - 1);  // Range from -4 to 4
    }
    
    // Apply GELU (exact version)
    gelu_exact_double(input, output, n);
    
    printf("Input -> Output (exact GELU):\n");
    for (size_t i = 0; i < n; i++) {
        printf("  %7.3f -> %7.4f\n", input[i], output[i]);
    }
    
    // Apply GELU (tanh approximation)
    gelu_tanh_double(input, output, n);
    
    printf("\nInput -> Output (tanh approximation):\n");
    for (size_t i = 0; i < n; i++) {
        printf("  %7.3f -> %7.4f\n", input[i], output[i]);
    }
    
    free(input);
    free(output);
    
    // Example 2: In-place computation (input == output)
    printf("\n\nExample 2: In-place computation\n");
    printf("================================\n");
    
    double *data = aligned_alloc(64, n * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        data[i] = -2.0 + 4.0 * i / (n - 1);
    }
    
    printf("Before GELU: ");
    for (size_t i = 0; i < 5; i++) {
        printf("%.3f ", data[i]);
    }
    printf("...\n");
    
    // Apply GELU in-place
    gelu_tanh_double(data, data, n);
    
    printf("After GELU:  ");
    for (size_t i = 0; i < 5; i++) {
        printf("%.3f ", data[i]);
    }
    printf("...\n");
    
    free(data);
    
    // Example 3: Single precision
    printf("\n\nExample 3: Single precision\n");
    printf("===========================\n");
    
    const size_t n_float = 8;
    float *input_f = aligned_alloc(64, n_float * sizeof(float));
    float *output_f = aligned_alloc(64, n_float * sizeof(float));
    
    for (size_t i = 0; i < n_float; i++) {
        input_f[i] = -3.0f + 6.0f * i / (n_float - 1);
    }
    
    gelu_exact_float(input_f, output_f, n_float);
    
    printf("Single precision GELU:\n");
    for (size_t i = 0; i < n_float; i++) {
        printf("  %.3f -> %.4f\n", input_f[i], output_f[i]);
    }
    
    free(input_f);
    free(output_f);
    
    // Example 4: Large batch processing
    printf("\n\nExample 4: Large batch processing\n");
    printf("==================================\n");
    
    const size_t n_large = 1000000;
    double *large_input = aligned_alloc(64, n_large * sizeof(double));
    double *large_output = aligned_alloc(64, n_large * sizeof(double));
    
    // Initialize with random-like values
    for (size_t i = 0; i < n_large; i++) {
        large_input[i] = ((double)i / n_large) * 6.0 - 3.0;
    }
    
    printf("Processing %zu elements...\n", n_large);
    gelu_tanh_double(large_input, large_output, n_large);
    
    // Verify a few values
    printf("Sample results:\n");
    printf("  output[0]       = %.6f\n", large_output[0]);
    printf("  output[%zu] = %.6f\n", n_large/2, large_output[n_large/2]);
    printf("  output[%zu] = %.6f\n", n_large-1, large_output[n_large-1]);
    
    free(large_input);
    free(large_output);
    
    // Example 5: Comparing exact vs tanh approximation
    printf("\n\nExample 5: Exact vs Tanh approximation\n");
    printf("======================================\n");
    
    const size_t n_cmp = 100;
    double *x = aligned_alloc(64, n_cmp * sizeof(double));
    double *y_exact = aligned_alloc(64, n_cmp * sizeof(double));
    double *y_tanh = aligned_alloc(64, n_cmp * sizeof(double));
    
    for (size_t i = 0; i < n_cmp; i++) {
        x[i] = -5.0 + 10.0 * i / (n_cmp - 1);
    }
    
    gelu_exact_double(x, y_exact, n_cmp);
    gelu_tanh_double(x, y_tanh, n_cmp);
    
    // Find maximum difference
    double max_diff = 0.0;
    size_t max_diff_idx = 0;
    for (size_t i = 0; i < n_cmp; i++) {
        double diff = fabs(y_exact[i] - y_tanh[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    
    printf("Maximum difference between exact and tanh:\n");
    printf("  Value:    %.6e\n", max_diff);
    printf("  At x =    %.3f\n", x[max_diff_idx]);
    printf("  Exact:    %.6f\n", y_exact[max_diff_idx]);
    printf("  Tanh:     %.6f\n", y_tanh[max_diff_idx]);
    
    // Sample points
    printf("\nSample comparison:\n");
    printf("     x      |   exact   |   tanh    |   diff\n");
    printf("------------|-----------|-----------|----------\n");
    for (size_t i = 0; i < n_cmp; i += n_cmp/10) {
        printf("  %7.3f   | %9.6f | %9.6f | %.3e\n",
               x[i], y_exact[i], y_tanh[i], fabs(y_exact[i] - y_tanh[i]));
    }
    
    free(x);
    free(y_exact);
    free(y_tanh);
    
    printf("\n\nAll examples completed successfully!\n");
    
    return 0;
}


