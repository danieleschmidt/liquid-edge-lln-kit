# Liquid Edge LLN Kit Development Container
FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    # Version control
    git \
    git-lfs \
    # Network tools
    curl \
    wget \
    netcat-openbsd \
    # Development tools
    vim \
    nano \
    htop \
    tree \
    jq \
    # Hardware simulation tools
    qemu-system-arm \
    qemu-user-static \
    # ARM cross-compilation
    gcc-arm-none-eabi \
    binutils-arm-none-eabi \
    gdb-multiarch \
    openocd \
    # ESP32 dependencies
    libffi-dev \
    libssl-dev \
    libudev-dev \
    # Additional utilities
    zip \
    unzip \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install ESP-IDF for ESP32 development
ENV ESP_IDF_VERSION=v5.1.2
RUN git clone --recursive --branch ${ESP_IDF_VERSION} --single-branch \
    https://github.com/espressif/esp-idf.git /opt/esp-idf && \
    cd /opt/esp-idf && \
    ./install.sh all && \
    chmod +x /opt/esp-idf/export.sh
ENV IDF_PATH=/opt/esp-idf

# Install PlatformIO for embedded development
RUN pip install --upgrade pip && \
    pip install platformio && \
    pio platform install espressif32 && \
    pio platform install ststm32

# Create conda environment for development
COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Set up conda environment activation
ENV PATH=/opt/conda/envs/liquid-edge-dev/bin:$PATH
ENV CONDA_DEFAULT_ENV=liquid-edge-dev

# Install additional Python packages not in conda
RUN pip install \
    # Development tools
    pre-commit \
    commitizen \
    bpython \
    # Testing enhancements
    pytest-benchmark \
    pytest-mock \
    pytest-asyncio \
    pytest-timeout \
    # Documentation
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    # Monitoring and profiling
    py-spy \
    memory-profiler \
    line-profiler \
    # Security scanning
    bandit \
    safety \
    pip-audit \
    # Hardware simulation
    renode \
    # ROS2 (if available)
    colcon-common-extensions || echo "ROS2 packages not available"

# Create workspace directories
RUN mkdir -p /workspace/{data,models,experiments,logs,artifacts,hardware_tests}

# Setup user permissions
RUN chown -R vscode:vscode /workspace && \
    chown -R vscode:vscode /opt/conda

# Switch to vscode user
USER vscode

# Setup git configuration
RUN git config --global init.defaultBranch main && \
    git config --global pull.rebase false && \
    git config --global user.email "developer@liquid-edge.com" && \
    git config --global user.name "Liquid Edge Developer"

# Setup shell environment
RUN echo 'export PATH="/opt/conda/envs/liquid-edge-dev/bin:$PATH"' >> ~/.bashrc && \
    echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> ~/.bashrc && \
    echo 'export LIQUID_EDGE_ENV="development"' >> ~/.bashrc && \
    echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias la="ls -A"' >> ~/.bashrc && \
    echo 'alias l="ls -CF"' >> ~/.bashrc

# Setup zsh if available
RUN if [ -f ~/.zshrc ]; then \
        echo 'export PATH="/opt/conda/envs/liquid-edge-dev/bin:$PATH"' >> ~/.zshrc && \
        echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> ~/.zshrc && \
        echo 'export LIQUID_EDGE_ENV="development"' >> ~/.zshrc && \
        echo 'alias ll="ls -la"' >> ~/.zshrc && \
        echo 'alias la="ls -A"' >> ~/.zshrc && \
        echo 'alias l="ls -CF"' >> ~/.zshrc; \
    fi

# Create Jupyter config
RUN mkdir -p ~/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Set working directory
WORKDIR /workspace

# Expose common development ports
EXPOSE 8000 8080 8081 8888 3000 9090 5000

# Default command
CMD ["/bin/bash"]