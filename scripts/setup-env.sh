#!/bin/bash
# Environment setup script for different platforms with CUDA auto-detection

set -e

# Detect platform and architecture
OS=$(uname -s)
ARCH=$(uname -m)

echo "Setting up environment for: $OS-$ARCH"

# Function to detect CUDA version
detect_cuda_version() {
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/')
        echo "Detected CUDA version: $CUDA_VERSION"
        
        # Map CUDA versions to PyTorch wheel indices
        case $CUDA_VERSION in
            12.0*|12.1*)
                export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
                echo "Using PyTorch CUDA 12.1 wheels"
                ;;
            12.2*|12.3*|12.4*|12.5*|12.6*|12.7*|12.8*)
                export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"
                echo "Using PyTorch CUDA 12.8 wheels"
                ;;
            *)
                echo "Unsupported CUDA version: $CUDA_VERSION, falling back to CPU"
                export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
                ;;
        esac
    else
        echo "No CUDA detected, using CPU-only PyTorch"
        export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
    fi
}

# Set Python version and install packages based on environment
if [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    # Ubuntu with potential GPU
    echo "3.13" > .python-version
    echo "Setting up Linux GPU environment (Python 3.13)..."
    
    detect_cuda_version
    uv sync --extra gpu-dev
    
    # Verify CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU information:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi
else
    # Local development (Windows/macOS)
    echo "3.12" > .python-version
    echo "Setting up local development environment (Python 3.12)..."
    uv sync --extra dev
fi

# Verify installation
echo "Verifying installation..."
uv run python -c "
import sys, platform, torch
print(f'Python: {sys.version.split()[0]}')
print(f'Platform: {platform.platform()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
else:
    print('Running on CPU')
"

echo "Environment setup complete!"
echo "To activate: source your shell or run commands with 'uv run'"
