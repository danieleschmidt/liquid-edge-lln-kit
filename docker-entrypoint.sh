#!/bin/bash
set -e

# Initialize ESP-IDF environment if not already done
if [ -f "$HOME/esp-idf/export.sh" ] && [ -z "$IDF_PATH" ]; then
    source "$HOME/esp-idf/export.sh"
fi

# Install package in development mode if pyproject.toml exists
if [ -f "/workspace/pyproject.toml" ] && [ ! -f "/workspace/.dev-installed" ]; then
    echo "Installing package in development mode..."
    cd /workspace
    python3.10 -m pip install --user -e ".[dev]"
    touch .dev-installed
fi

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f "/workspace/.pre-commit-config.yaml" ] && [ ! -f "/workspace/.pre-commit-installed" ]; then
    echo "Installing pre-commit hooks..."
    cd /workspace
    pre-commit install
    touch .pre-commit-installed
fi

# Set up Jupyter configuration
if [ ! -d "$HOME/.jupyter" ]; then
    echo "Setting up Jupyter configuration..."
    mkdir -p "$HOME/.jupyter"
    cat > "$HOME/.jupyter/jupyter_notebook_config.py" << EOF
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.allow_root = True
EOF
fi

# Create useful aliases
cat > "$HOME/.bash_aliases" << 'EOF'
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Project specific aliases
alias test='python -m pytest'
alias test-cov='python -m pytest --cov=liquid_edge --cov-report=html'
alias test-fast='python -m pytest -x -vvs'
alias lint='ruff check .'
alias fmt='black . && ruff check --fix .'
alias typecheck='mypy src/'
alias clean='find . -type f -name "*.pyc" -delete && find . -type d -name "__pycache__" -delete'

# Hardware development
alias esp-build='idf.py build'
alias esp-flash='idf.py flash'
alias esp-monitor='idf.py monitor'
alias pio='platformio'

# JAX utilities
alias jax-info='python -c "import jax; print(f\"JAX version: {jax.__version__}\"); print(f\"Devices: {jax.devices()}\"); print(f\"Platform: {jax.default_backend()}\")"'
EOF

# Source aliases
echo "source ~/.bash_aliases" >> "$HOME/.bashrc"

# Display welcome message
cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                    Liquid Edge LLN Kit                       ║
║                  Development Container                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Available commands:                                         ║
║    test         - Run pytest test suite                     ║
║    test-cov     - Run tests with coverage                   ║
║    lint         - Run code linting                          ║
║    fmt          - Format code with black/ruff               ║
║    typecheck    - Type checking with mypy                   ║
║    jax-info     - Display JAX configuration                 ║
║                                                              ║
║  Hardware development:                                       ║
║    esp-build    - Build ESP32 firmware                      ║
║    esp-flash    - Flash firmware to device                  ║
║    pio          - PlatformIO CLI                            ║
║                                                              ║
║  Jupyter notebook available on port 8888                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF

# Execute the main command
exec "$@"