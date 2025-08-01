#!/bin/bash

set -e

echo "🚀 Setting up Liquid Edge LLN Kit development environment..."

# Update system packages
sudo apt-get update

# Install system dependencies for embedded development
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libusb-1.0-0-dev \
    pkg-config \
    curl \
    wget \
    unzip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev,deployment,ros2,monitoring,docs]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Install ESP-IDF (for ESP32 development)
echo "📱 Installing ESP-IDF..."
mkdir -p /opt/esp
cd /opt/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh
echo '. /opt/esp/esp-idf/export.sh' >> ~/.bashrc

# Install PlatformIO for embedded development
echo "🔌 Installing PlatformIO..."
pip install platformio

# Install additional MCU tools
echo "⚙️ Installing MCU development tools..."
# OpenOCD for debugging
sudo apt-get install -y openocd

# Create workspace directories
echo "📁 Creating workspace structure..."
mkdir -p /workspaces/liquid-edge-lln-kit/{models,firmware,datasets,experiments}

# Set up git configuration template
git config --global init.defaultBranch main
git config --global pull.rebase false

# Install JetBrains Mono font for better code readability
echo "🔤 Installing developer fonts..."
sudo apt-get install -y fonts-jetbrains-mono

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Run 'liquid-lln doctor' to verify installation"
echo "  2. Check out the examples/ directory"
echo "  3. Read docs/DEVELOPMENT.md for contribution guidelines"
echo ""
echo "💡 Useful commands:"
echo "  - make test: Run all tests"
echo "  - make lint: Check code quality"
echo "  - make docs: Build documentation"
echo "  - liquid-lln --help: CLI help"