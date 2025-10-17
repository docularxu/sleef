# Makefile for GELU Kernel using SLEEF
# This is a simplified wrapper - see Makefile.gelu for full options
# 
# Usage:
#   make              # Build for detected architecture
#   make test         # Build and run tests
#   make libs         # Build shared libraries
#   make clean        # Clean all builds
#   make help         # Show detailed help

# Use the full Makefile
include Makefile.gelu

.DEFAULT_GOAL := all

# Show help by default if running 'make' with no prior builds
.PHONY: info
info:
	@echo "GELU Kernel Build System"
	@echo "========================"
	@echo ""
	@echo "Quick commands:"
	@echo "  make          - Build all available targets"
	@echo "  make test     - Build and run tests"
	@echo "  make benchmark- Build and run benchmarks"
	@echo "  make libs     - Build shared libraries"
	@echo "  make clean    - Clean all builds"
	@echo ""
	@echo "For more options: make -f Makefile.gelu help"
	@echo ""

