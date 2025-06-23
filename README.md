<div align="center">

# 🤖 MLX8-W3-Transformers
### Week 3: Advanced Transformer Architectures & Implementation

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-red.svg)](https://pytorch.org)
[![UV](https://img.shields.io/badge/UV-Package%20Manager-green.svg)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Building state-of-the-art transformer models from scratch with modern MLOps practices*

</div>

---

## 📺 Educational Resources

### 🎥 Neural Networks Fundamentals - 3Blue1Brown Series

<div align="center">

[![Neural Networks Playlist](https://img.youtube.com/vi/aircAruvnKk/maxresdefault.jpg)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

**🧠 [Neural Networks Fundamentals Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3Blue1Brown**

*Click the image above to watch the complete series* 📹

</div>

#### 📚 What you'll learn from this series:

<table>
<tr>
<td width="30%">

**🎯 Core Concepts**
- Neural network basics
- Gradient descent intuition
- Backpropagation explained
- Mathematical foundations

</td>
<td width="35%">

**🔬 Visual Understanding**
- Interactive visualizations
- Mathematical animations
- Intuitive explanations
- Beautiful graphics

</td>
<td width="35%">

**🚀 Foundation for Transformers**
- Building blocks of deep learning
- Optimization principles
- Network architecture design
- Mathematical rigor

</td>
</tr>
</table>

#### 🎬 Series Breakdown:

| Episode | Topic | Duration | Key Concepts |
|---------|-------|----------|--------------|
| **1** | [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | 19 min | Neurons, layers, MNIST |
| **2** | [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) | 21 min | Cost functions, optimization |
| **3** | [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) | 14 min | Chain rule, derivatives |
| **4** | [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) | 10 min | Mathematical details |

---

### 🎓 Course Content - Advanced Transformers

#### 🎥 Week 3 Main Lecture: From Neural Networks to Transformers

<div align="center">

[![Advanced Transformer Architectures](https://img.youtube.com/vi/aircAruvnKk/maxresdefault.jpg)](https://discord.com/channels/1213520333026500719/1381608466103140453/1386679136172638238)

**📋 Advanced Transformer Implementation Workshop**


</div>

**📋 What you'll learn in this video:**
- 🔧 **Transformer Architecture Deep Dive**: Understanding attention mechanisms, positional encoding, and layer normalization
- 🚀 **Implementation from Scratch**: Building transformers with PyTorch, including multi-head attention and feed-forward networks
- 📊 **Training Strategies**: Advanced techniques for training large transformer models efficiently
- 🎯 **Fine-tuning & Transfer Learning**: Adapting pre-trained models for specific tasks
- 🛠️ **MLOps Integration**: Using modern tools like UV for dependency management and reproducible environments
- 📈 **Performance Optimization**: Memory management, gradient checkpointing, and distributed training

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ (3.13 for GPU environments)
- CUDA 12.06+ (for GPU training)
- [UV Package Manager](https://docs.astral.sh/uv/)
- **Recommended**: Watch the 3Blue1Brown series first! 🎥

### 🏃‍♂️ Get Running in 60 Seconds

```bash
# 1. Clone the repository
git clone https://github.com/your-username/MLX8-W3-Transformers.git
cd MLX8-W3-Transformers

# 2. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Setup environment (auto-detects your platform)
uv sync

# 4. Run your first transformer!
uv run python examples/basic_transformer.py
```

### 🖥️ Platform-Specific Setup

<details>
<summary><strong>🪟 Windows 11 Development</strong></summary>

```bash
echo "3.12" > .python-version
uv sync --extra dev
uv run python examples/cpu_training.py
```
</details>

<details>
<summary><strong>🍎 macOS (Intel & Apple Silicon)</strong></summary>

```bash
echo "3.12" > .python-version  
uv sync --extra dev
uv run python examples/cpu_training.py
```
</details>

<details>
<summary><strong>🐧 Ubuntu 22.04 + CUDA 12.06</strong></summary>

```bash
echo "3.13" > .python-version
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
uv sync --extra gpu-dev
uv run python examples/gpu_training.py
```
</details>

<details>
<summary><strong>🐧 Ubuntu 24.04 + CUDA 12.8</strong></summary>

```bash
echo "3.13" > .python-version
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"  
uv sync --extra gpu-dev
uv run python examples/gpu_training.py
```
</details>

---

## 📚 Project Structure

