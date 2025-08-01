#!/bin/bash

# Copyright (c) 2025 Liquid Edge LLN Kit Contributors
# SPDX-License-Identifier: MIT

set -e

# Build script for Liquid Edge LLN Kit
# Supports multiple build targets and configurations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
BUILD_TARGET="all"
BUILD_TYPE="release"
CLEAN=false
VERBOSE=false
DOCKER_BUILD=false
CROSS_COMPILE=""
OUTPUT_DIR="$PROJECT_ROOT/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Liquid Edge LLN Kit Build Script"
    echo "================================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --target TARGET     Build target: all, python, docker, mcu, docs"
    echo "  -b, --build-type TYPE   Build type: debug, release (default: release)"
    echo "  -c, --clean            Clean build directory before building"
    echo "  -v, --verbose          Verbose output"
    echo "  -d, --docker           Build using Docker container"
    echo "  -x, --cross-compile    Cross-compile target (arm, esp32, etc.)"
    echo "  -o, --output DIR       Output directory (default: build/)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Build everything"
    echo "  $0 --target python --clean           # Clean build Python package"
    echo "  $0 --target mcu --cross-compile arm  # Cross-compile for ARM MCU"
    echo "  $0 --docker                          # Build in Docker container"
}

log() {
    echo -e "${BLUE}[BUILD]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -b|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--docker)
            DOCKER_BUILD=true
            shift
            ;;
        -x|--cross-compile)
            CROSS_COMPILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate build target
case $BUILD_TARGET in
    all|python|docker|mcu|docs)
        ;;
    *)
        error "Invalid build target: $BUILD_TARGET"
        usage
        exit 1
        ;;
esac

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Clean build directory if requested
if [[ "$CLEAN" == "true" ]]; then
    log "Cleaning build directory..."
    rm -rf "$OUTPUT_DIR"
    rm -rf "$PROJECT_ROOT/dist"
    rm -rf "$PROJECT_ROOT"/*.egg-info
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build functions
build_python() {
    log "Building Python package..."
    
    cd "$PROJECT_ROOT"
    
    # Install build dependencies
    python -m pip install --upgrade pip build wheel
    
    # Build package
    if [[ "$BUILD_TYPE" == "debug" ]]; then
        python -m build --wheel
    else
        python -m build
    fi
    
    # Copy artifacts
    cp -r dist/* "$OUTPUT_DIR/"
    
    success "Python package built successfully"
}

build_docker() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build multi-stage Docker image
    docker build -t liquid-edge-lln:latest \
                 -t liquid-edge-lln:runtime \
                 --target runtime .
    
    docker build -t liquid-edge-lln:dev \
                 --target development .
    
    docker build -t liquid-edge-lln:testing \
                 --target testing .
    
    # Save images
    docker save liquid-edge-lln:latest | gzip > "$OUTPUT_DIR/liquid-edge-lln.tar.gz"
    
    success "Docker images built successfully"
}

build_mcu() {
    log "Building MCU firmware..."
    
    if [[ -z "$CROSS_COMPILE" ]]; then
        error "Cross-compile target required for MCU build"
        exit 1
    fi
    
    case $CROSS_COMPILE in
        arm)
            build_arm_mcu
            ;;
        esp32)
            build_esp32_mcu
            ;;
        *)
            error "Unsupported cross-compile target: $CROSS_COMPILE"
            exit 1
            ;;
    esac
}

build_arm_mcu() {
    log "Building for ARM Cortex-M..."
    
    # Check for ARM toolchain
    if ! command -v arm-none-eabi-gcc &> /dev/null; then
        error "ARM toolchain not found. Please install arm-none-eabi-gcc"
        exit 1
    fi
    
    # Create CMake build directory
    mkdir -p "$OUTPUT_DIR/arm"
    cd "$OUTPUT_DIR/arm"
    
    # Configure CMake for ARM
    cmake "$PROJECT_ROOT/firmware/arm" \
          -DCMAKE_TOOLCHAIN_FILE="$PROJECT_ROOT/cmake/arm-none-eabi.cmake" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    
    # Build
    make -j$(nproc)
    
    success "ARM firmware built successfully"
}

build_esp32_mcu() {
    log "Building for ESP32..."
    
    # Check for ESP-IDF
    if [[ -z "$IDF_PATH" ]]; then
        error "ESP-IDF not found. Please set IDF_PATH environment variable"
        exit 1
    fi
    
    # Source ESP-IDF environment
    source "$IDF_PATH/export.sh"
    
    # Create build directory
    mkdir -p "$OUTPUT_DIR/esp32"
    cd "$PROJECT_ROOT/firmware/esp32"
    
    # Set target
    idf.py set-target esp32s3
    
    # Build
    idf.py build
    
    # Copy artifacts
    cp build/*.bin "$OUTPUT_DIR/esp32/"
    
    success "ESP32 firmware built successfully"
}

build_docs() {
    log "Building documentation..."
    
    cd "$PROJECT_ROOT"
    
    # Install documentation dependencies
    python -m pip install -e ".[docs]"
    
    # Build documentation
    mkdir -p "$OUTPUT_DIR/docs"
    
    if command -v sphinx-build &> /dev/null; then
        sphinx-build -b html docs/ "$OUTPUT_DIR/docs/html"
        sphinx-build -b pdf docs/ "$OUTPUT_DIR/docs/pdf"
    else
        warning "Sphinx not found, using simple HTTP server documentation"
        cp -r docs/* "$OUTPUT_DIR/docs/"
    fi
    
    success "Documentation built successfully"
}

# Run quality checks
run_quality_checks() {
    log "Running quality checks..."
    
    cd "$PROJECT_ROOT"
    
    # Linting
    if command -v ruff &> /dev/null; then
        ruff check src/ tests/
    else
        warning "Ruff not found, skipping linting"
    fi
    
    # Type checking
    if command -v mypy &> /dev/null; then
        mypy src/liquid_edge/
    else
        warning "MyPy not found, skipping type checking"
    fi
    
    # Security scanning
    if [[ -f "scripts/security_scan.py" ]]; then
        python scripts/security_scan.py
    fi
    
    success "Quality checks completed"
}

# Docker build wrapper
build_in_docker() {
    log "Building in Docker container..."
    
    docker run --rm \
               -v "$PROJECT_ROOT:/workspace" \
               -v "$OUTPUT_DIR:/output" \
               liquid-edge-dev:latest \
               /workspace/scripts/build.sh --target "$BUILD_TARGET" \
                                          --build-type "$BUILD_TYPE" \
                                          --output /output
}

# Main build logic
main() {
    log "Starting build process..."
    log "Target: $BUILD_TARGET"
    log "Build type: $BUILD_TYPE"
    log "Output directory: $OUTPUT_DIR"
    
    if [[ "$DOCKER_BUILD" == "true" ]]; then
        build_in_docker
        return
    fi
    
    case $BUILD_TARGET in
        all)
            run_quality_checks
            build_python
            build_docker
            build_docs
            ;;
        python)
            run_quality_checks
            build_python
            ;;
        docker)
            build_docker
            ;;
        mcu)
            build_mcu
            ;;
        docs)
            build_docs
            ;;
    esac
    
    success "Build completed successfully!"
    log "Output available in: $OUTPUT_DIR"
}

# Run main function
main