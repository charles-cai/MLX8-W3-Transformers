# UV Python Environment Management

## Course Materials


| Document | Description |
|----------|-------------|
| [bes-shared-code.py](bes-shared-code.py) | Shared code for Braille/Einops/MNIST dataset composition |

This project uses [UV](https://docs.astral.sh/uv/) for fast, reliable Python package and environment management across multiple platforms and Python versions.

## Environment Overview

### Supported Platforms & Python Versions

| Platform | Architecture | Python Version | GPU Support | Environment |
|----------|-------------|----------------|-------------|-------------|
| Windows 11 | x64 | 3.12 | No | Local Development |
| macOS Intel | x64 | 3.12 | No | Local Development |
| macOS Apple Silicon | arm64 | 3.12 | No | Local Development |
| Ubuntu 22.04 | x64 | 3.13 | CUDA 12.06 | Remote/Production |
| Ubuntu 24.04 | x64 | 3.13 | CUDA 12.8 | Remote/Production |

### Python Version Priority

UV resolves Python versions in this order:
1. **`.python-version`** - Project default (3.13 for remote, 3.12 for local dev)
2. **`pyproject.toml`** - Version constraints (`>=3.12,<3.14`)
3. **`uv.lock`** - Locked specific versions per platform

## Quick Start

### Initial Setup
```bash
# Clone repository
git clone <repo-url>
cd MLX8-W3-Transformers

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and sync environment
uv sync
```

### Daily Workflow Commands

#### After `git pull`
```bash
# Sync dependencies (updates lockfile if needed)
uv sync

# If lockfile conflicts, regenerate
uv lock --upgrade
```

#### Before `git commit`
```bash
# Lock dependencies to ensure reproducible builds
uv lock

# Run tests
uv run pytest

# Format code
uv run black .
uv run ruff check --fix .
```

#### Running Applications
```bash
# Run main application
uv run python main.py

# Run with FastAPI server
uv run uvicorn app:app --reload

# Run Jupyter notebook
uv run jupyter lab
```

## Platform-Specific Configurations

### Local Development (Windows/macOS - Python 3.12)
```bash
# Set local development Python version
echo "3.12" > .python-version

# Standard CPU-only environment
uv sync

# With development tools
uv sync --extra dev
```

### Remote GPU Environments (Ubuntu - Python 3.13)

#### Auto-detect CUDA version and configure
```bash
# Set Python version for GPU environment
echo "3.13" > .python-version

# Check CUDA version
nvidia-smi | grep "CUDA Version"

# Install with GPU support (auto-detects CUDA version)
uv sync --extra gpu

# Verify CUDA setup
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
"
```

#### Manual CUDA version configuration
```bash
# For Ubuntu 22.04 with CUDA 12.06
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
uv sync --extra gpu

# For Ubuntu 24.04 with CUDA 12.8
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"
uv sync --extra gpu
```

## Environment Management

### Switch Between Environments
```bash
# Local development (Python 3.12, CPU)
echo "3.12" > .python-version
uv sync --extra dev

# Remote GPU Ubuntu 22.04 (Python 3.13, CUDA 12.06)
echo "3.13" > .python-version
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
uv sync --extra gpu

# Remote GPU Ubuntu 24.04 (Python 3.13, CUDA 12.8)
echo "3.13" > .python-version
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"
uv sync --extra gpu
```

### Clean Environment
```bash
# Remove virtual environment
uv clean

# Recreate from scratch
uv sync --reinstall
```

### Add New Dependencies
```bash
# Add runtime dependency
uv add numpy pandas

# Add development dependency
uv add --dev pytest black

# Add GPU-specific dependency
uv add --optional gpu nvidia-ml-py
```

## Environment Variables

### Platform Detection & CUDA Configuration
```bash
# Ubuntu 22.04 with CUDA 12.06
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
export CUDA_VISIBLE_DEVICES=0

# Ubuntu 24.04 with CUDA 12.8
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"
export CUDA_VISIBLE_DEVICES=0

# macOS/Windows (CPU only)
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
```

### Performance Optimization
```bash
# Faster dependency resolution
export UV_RESOLUTION=highest

# Parallel installs
export UV_CONCURRENT_DOWNLOADS=10
```

## Troubleshooting

### Common Issues

#### CUDA Version Detection
```bash
# Check system CUDA version
nvidia-smi

# Check PyTorch CUDA compatibility
uv run python -c "
import torch
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'System CUDA compatible: {torch.cuda.is_available()}')
"

# Check available PyTorch CUDA versions
curl -s https://download.pytorch.org/whl/torch/ | grep -o 'cu[0-9]\{3\}' | sort -u
```

#### Dependency Conflicts
```bash
# Clear cache and reinstall
uv cache clean
uv sync --reinstall
```

#### Platform-Specific Package Issues
```bash
# Force platform-specific resolution
uv lock --python-platform linux
uv lock --python-platform darwin
uv lock --python-platform win32
```

#### CUDA Version Mismatch
```bash
# Ubuntu 22.04 - Force CUDA 12.1 compatible PyTorch
uv add torch --index-url https://download.pytorch.org/whl/cu121

# Ubuntu 24.04 - Force CUDA 12.8 compatible PyTorch  
uv add torch --index-url https://download.pytorch.org/whl/cu128

# Fallback to CPU version
uv add torch --index-url https://download.pytorch.org/whl/cpu
```

### Environment Validation
```bash
# Check Python version
uv run python --version

# Comprehensive environment check
uv run python -c "
import sys, platform, torch
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
else:
    print('Running on CPU')
"
```

## CI/CD Integration

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
strategy:
  matrix:
    os: [ubuntu-22.04, ubuntu-24.04]
    python-version: ["3.13"]
    include:
      - os: windows-latest
        python-version: "3.12"
      - os: macos-latest  
        python-version: "3.12"

steps:
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: ${{ matrix.python-version }}

- name: Install UV
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Configure CUDA (Ubuntu)
  if: startsWith(matrix.os, 'ubuntu')
  run: |
    if [[ "${{ matrix.os }}" == "ubuntu-22.04" ]]; then
      echo "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121" >> $GITHUB_ENV
    else
      echo "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128" >> $GITHUB_ENV  
    fi

- name: Install dependencies
  run: |
    if [[ "${{ matrix.os }}" == *"ubuntu"* ]]; then
      uv sync --extra gpu-dev
    else
      uv sync --extra dev
    fi

- name: Run tests
  run: uv run pytest
```

## Performance Tips

1. **Use `uv sync` instead of `uv install`** - Faster dependency resolution
2. **Pin exact versions in production** - Use `uv lock` before deployment
3. **Platform-specific locks** - Generate separate lockfiles for different platforms
4. **Cache dependencies** - UV automatically caches for faster subsequent installs
5. **CUDA version matching** - Ensure PyTorch CUDA version matches system CUDA

## Best Practices

1. **Always run `uv lock` before committing** - Ensures reproducible builds
2. **Use optional dependencies** - Keep core lightweight, add features as needed
3. **Platform-specific testing** - Test on each target platform before release
4. **Regular updates** - Run `uv lock --upgrade` periodically to get security updates
5. **CUDA compatibility** - Verify CUDA versions match between system and PyTorch

For more advanced configurations and troubleshooting, see the [UV documentation](https://docs.astral.sh/uv/).
