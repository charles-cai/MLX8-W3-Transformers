# Encoder Only Transformer & Encoder-Decoder for MNIST 4-Digit Recognition

This project implements Transformer-based models for recognizing sequences of digits in stacked MNIST images. The main focus is on two architectures:
- **Encoder Only Transformer** (ViT-style) for single-digit and multi-digit classification.
- **Encoder-Decoder Transformer** for sequence-to-sequence recognition of 4-digit numbers from 2x2 stacked MNIST images.

---

## Project Overview

- **Goal:** Recognize 4-digit numbers from images created by stacking four MNIST digits in a 2x2 grid (56x56 pixels).
- **Approach:** Use Vision Transformer (ViT) patch embedding and Transformer blocks to encode images, and a Transformer decoder to autoregressively generate digit sequences.
- **Dataset:** Based on the [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist) dataset, with custom code to generate 4-digit stacked samples.

---

## Dataset

- **Source:** [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)
- **Format:** Parquet files with columns:
  - `image`: 28x28 grayscale digit (as bytes/blob)
  - `label`: integer digit (0-9)
- **4-digit dataset:** Each sample is a 2x2 grid of random MNIST digits, resulting in a 56x56 image and a label vector of length 4.

---

## Model Architectures

### Encoder Only Transformer

- **Input:** 28x28 MNIST image
- **Patch Embedding:** Conv2d to split into 7x7 patches (49 tokens)
- **CLS Token:** Added for classification
- **Positional Embedding:** Learnable
- **Transformer Blocks:** Stack of self-attention + feedforward layers
- **Output:** Classification logits for 10 classes

### Encoder-Decoder Transformer

- **Input:** 56x56 image (2x2 grid of digits)
- **Patch Embedding:** Conv2d to 8x8 patches (64 tokens)
- **Encoder:** Stack of Transformer blocks, pooling to 4 quadrant representations
- **Decoder:** Autoregressive Transformer, generates 4-digit sequence
- **Output:** Sequence of 4 digits (0-9)

---

## Model Diagram

Encoder Only Transformer Architecture:

```mermaid
graph TD
    A[Input Image 28×28] --> B[PatchEmbed Conv2d]
    B --> C[Patches 7×7 = 49 tokens]
    C --> D[Add CLS Token]
    D --> E[Sequence: CLS + 49 patches = 50 tokens]
    E --> F[Positional Embedding]
    F --> G[TransformerBlock 1]
    G --> H[TransformerBlock 2]
    H --> I[TransformerBlock 3]
    I --> J[TransformerBlock 4]
    J --> K[Extract CLS Token]
    K --> L[Classification Head]
    L --> M[Output Logits 10 classes]

    subgraph TB ["TransformerBlock Structure"]
        TB1[Input x] --> TB2[MultiHeadAttention]
        TB2 --> TB3[+ Residual]
        TB3 --> TB4[LayerNorm]
        TB4 --> TB5[FeedForward]
        TB5 --> TB6[+ Residual]
        TB6 --> TB7[LayerNorm]
        TB7 --> TB8[Output x]
    end

    subgraph MHA ["MultiHeadAttention"]
        MHA1[Input] --> MHA2[Linear Q]
        MHA1 --> MHA3[Linear K]  
        MHA1 --> MHA4[Linear V]
        MHA2 --> MHA5[Scaled Dot-Product]
        MHA3 --> MHA5
        MHA4 --> MHA5
        MHA5 --> MHA6[Concat Heads]
        MHA6 --> MHA7[Linear Output]
    end

    subgraph FF ["FeedForward"]
        FF1[Input] --> FF2[Linear 128→512]
        FF2 --> FF3[ReLU]
        FF3 --> FF4[Dropout]
        FF4 --> FF5[Linear 512→128]
    end

    style A fill:#e1f5fe
    style M fill:#f3e5f5
    style TB fill:#fff3e0
    style MHA fill:#e8f5e8
    style FF fill:#fce4ec
```

Encoder-Decoder Transformer Architecture:

```mermaid
graph TD
    A[Input Image 56×56<br/>2×2 Grid of Digits] --> B[PatchEmbed Conv2d<br/>patch=7]
    B --> C[Patches 8×8 = 64 tokens]
    C --> D[Positional Embedding]
    D --> E[Encoder Stack<br/>6 TransformerBlocks]
    E --> F[Pool to 4 Quadrants<br/>4 digit representations]
    
    G[Start Token: 10] --> H[Token Embedding]
    H --> I[Decoder Input<br/>+ Positional Embedding]
    I --> J[Decoder Stack<br/>6 TransformerBlocks]
    
    F --> K[Cross-Attention<br/>Encoder → Decoder]
    J --> K
    K --> L[Output Projection]
    L --> M[4-Digit Sequence<br/>0-9 each position]
    
    N[Causal Mask] --> J
    
    subgraph ENC ["Encoder Architecture"]
        E1[Input Patches] --> E2[Self-Attention]
        E2 --> E3[+ Residual & LayerNorm]
        E3 --> E4[FeedForward]
        E4 --> E5[+ Residual & LayerNorm]
        E5 --> E6[Quadrant Pooling]
        E6 --> E7[4 Digit Encodings]
    end
    
    subgraph DEC ["Decoder Architecture"]
        D1[Target Tokens] --> D2[Masked Self-Attention]
        D2 --> D3[+ Residual & LayerNorm]
        D3 --> D4[Cross-Attention<br/>with Encoder]
        D4 --> D5[+ Residual & LayerNorm]
        D5 --> D6[FeedForward]
        D6 --> D7[+ Residual & LayerNorm]
    end
    
    subgraph GEN ["Autoregressive Generation"]
        G1[START] --> G2[Predict Digit 1]
        G2 --> G3[Predict Digit 2]
        G3 --> G4[Predict Digit 3]
        G4 --> G5[Predict Digit 4]
        G5 --> G6[Complete Sequence]
    end
    
    subgraph QUAD ["Quadrant Pooling"]
        Q1[64 Patches<br/>8×8 Grid] --> Q2[Top-Left 4×4<br/>→ Digit 1 Encoding]
        Q1 --> Q3[Top-Right 4×4<br/>→ Digit 2 Encoding]
        Q1 --> Q4[Bottom-Left 4×4<br/>→ Digit 3 Encoding]
        Q1 --> Q5[Bottom-Right 4×4<br/>→ Digit 4 Encoding]
    end
    
    style A fill:#e1f5fe
    style M fill:#f3e5f5
    style ENC fill:#fff3e0
    style DEC fill:#e8f5e8
    style GEN fill:#fce4ec
    style QUAD fill:#f1f8e9
```

---

## Usage

### 1. Clone the Dataset

```bash
cd .charles/data
bash clone-mnist.sh
```

### 2. Install Requirements

Inside the repo folder,

```bash
uv sync
```

### 3. Configure Environment

Copy `.env.example` to `.env` and set your parameters (especially for Weights & Biases logging).

### 4. Train the Model

```bash
cd .charles

# Encoder only model
uv run encoder_only_models.py

# Encoder + Decoder model
uv run encoder_decode_models.py

```

- Training and validation progress will be logged to Weights & Biases (wandb).
- Model checkpoints are saved to `.data/models`.

---

## Results

- The encoder only model achieves high accuracy on a single digit recognition on par with CNN (Foundation Project), 97.57%, before sweeping.
- The encoder-decoder model achieves high accuracy on 4-digit recognition, valuation accuracy 87.5% before sweeping.
- Validation includes visualizations of predictions using a Braille-style display for easy inspection.

---

## References

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [HuggingFace Datasets: ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---