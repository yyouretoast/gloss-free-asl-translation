# Gloss-Free ASL-to-English Translation

A deep learning framework to translate American Sign Language (ASL) video coordinates directly into fluent English sentences without using intermediate gloss representations. The project connects a custom **Conformer Encoder** directly to a pretrained **Hugging Face T5 Decoder** to perform end-to-end visual-to-text sequence generation.

---

## 🚀 Key Achievements & Architectural Features

### 1. Robust Coordinates Processing Pipeline
* **Targeted Landmark Extraction (`src/data_pipeline.py`)**: Uses MediaPipe to extract keypoints across Pose, Face, and Hands. Slices face landmarks to a optimized **92-point facial subset** (eyebrows, eyes, lips) instead of the full 468 points, reducing coordinate dimensionality by **80%** (from 1404 to 276 dimensions) while preserving critical grammatical facial expressions.
* **Scale & Distance Invariance (`src/dataset.py`)**: Coordinates are geometrically normalized to ensure model invariance to camera distance, signer height, and camera movement.
* **How2Sign & OpenPose Support (`src/dataset.py`)**: Custom parser to dynamically load OpenPose JSON outputs frame-by-frame. Auto-aligns keypoint layouts (e.g. mapping BODY_25 shoulders vs. MediaPipe landmarks) to handle multiple dataset schemas seamlessly.

### 2. High-Performance Model Architecture
* **Hybrid Conformer Encoder (`src/models/manual_encoder.py`)**: Combines multi-head self-attention (global context) with convolutional blocks (local movement details) in 4 Conformer blocks (`d_model=512`, `heads=4`, `kernel_size=31`).
* **Learnable Temporal Downsampling**: Uses a strided convolutional layer (`stride=2`) rather than max-pooling, allowing the encoder to learn downsampling while retaining continuous movement trajectories.
* **Rigorous Padding Fixes**: Implements `LayerNorm` (with transpose) instead of standard `BatchNorm1d` within Conformer blocks. This prevents padded zero-tokens in variable-length sequences from biasing batch normalization statistics.
* **Seamless T5 Cross-Attention Wrapper (`src/train.py`)**: Routes Conformer visual outputs directly to the cross-attention layers of the T5 decoder, bypassing the T5 text encoder completely.

### 3. Leak-Free Evaluation and Dataset Auditing
* **Signer-Based Splits**: Automatically parses metadata manifests to separate training, validation, and test splits by uploader/channel ID. This guarantees that validation metrics are evaluated on unseen signers, preventing signer data leakage.
* **Fast Profiler (`src/validate_dataset.py`)**: Evaluates landmark quality (sequence length distribution, hand/face tracking dropout percentages, and frame-to-frame wrist jitter noise) on large directories in seconds using non-recursive directory lists.

---

## 📐 Mathematical Formulation of Coordinate Normalization

To achieve scale and camera distance invariance, frame-level landmarks are normalized as follows:

1. **Mid-Shoulder Centering ($p_{\text{mid-shoulder}}$)**:
   $$p_{\text{mid-shoulder}} = \frac{p_{\text{shoulder1}} + p_{\text{shoulder2}}}{2}$$
   $$w_{\text{shoulder}} = \max\left(0.01, \|p_{\text{shoulder1}} - p_{\text{shoulder2}}\|_2\right)$$
   
   *For MediaPipe inputs, shoulders are indices `11` and `12`.*
   *For OpenPose BODY_25 inputs, shoulders are indices `2` and `5`.*

2. **Pose Coordinate Normalization ($p_{\text{norm}}$)**:
   $$p_{\text{norm}} = \frac{p - p_{\text{mid-shoulder}}}{w_{\text{shoulder}}}$$

3. **Hand Coordinate Normalization ($h_{\text{norm}}$)**:
   Hands are centered relative to their respective wrist joint ($w_{\text{wrist}}$, index `0` of the hand stream):
   $$h_{\text{norm}} = \frac{h - w_{\text{wrist}}}{w_{\text{shoulder}}}$$

4. **Face Coordinate Normalization ($f_{\text{norm}}$)**:
   Face coordinates are centered relative to the facial bounding centroid ($c_{\text{face-centroid}}$):
   $$f_{\text{norm}} = \frac{f - c_{\text{face-centroid}}}{w_{\text{shoulder}}}$$

---

## 📊 Landmark File & Dimension Specifications

### 1. NPZ File Format Structure
If using pre-extracted `.npz` files, each file contains the following compressed NumPy arrays:
* **`pose`**: Shape `(num_frames, 33, 4)` representing $(x, y, z, \text{visibility})$.
* **`left_hand`**: Shape `(num_frames, 21, 3)` representing $(x, y, z)$.
* **`right_hand`**: Shape `(num_frames, 21, 3)` representing $(x, y, z)$.
* **`face`**: Shape `(num_frames, 92, 3)` representing $(x, y, z)$.

### 2. Feature Dimension Breakdown
The feature vectors are concatenated per frame into a single 1D tensor:
* **Face Enabled (Default)**: **534 dimensions**
  $$\text{Pose (} 33 \times 4 = 132\text{)} + \text{Left Hand (} 21 \times 3 = 63\text{)} + \text{Right Hand (} 21 \times 3 = 63\text{)} + \text{Face (} 92 \times 3 = 276\text{)} = 534\text{ dims}$$
* **Face Disabled (`--no_face`)**: **258 dimensions**
  $$\text{Pose (} 33 \times 4 = 132\text{)} + \text{Left Hand (} 21 \times 3 = 63\text{)} + \text{Right Hand (} 21 \times 3 = 63\text{)} = 258\text{ dims}$$

---

## 📋 Metadata CSV/TSV Schema Mapping

The metadata parser dynamically maps columns by matching keywords (case-insensitive):
* **Video/File Key**: Automatically maps the first column containing `id`, `file`, `video`, `key`, or `name`.
* **Target English Text**: Automatically maps the first column containing `text`, `trans`, `gloss`, `sentence`, or `caption`.
* **Signer/Uploader ID**: Automatically maps the first column containing `signer`, `channel`, `uploader`, `author`, or `subject` to enforce independent split isolation.

*Note: If the metadata filepath points to a file containing split patterns (e.g. `_train.csv`), the training loop will automatically search the directory and dynamically merge all splits (`_train`, `_val`, `_test`) before partition.*

---

## 📈 Evaluation & Metrics

The translation model is evaluated at the end of each training epoch using two primary sequence-to-sequence evaluation metrics:

1. **Word Error Rate (WER)** (calculated using `jiwer`):
   $$\text{WER} = \frac{S + D + I}{N}$$
   where:
   * $S$ is the number of word substitutions.
   * $D$ is the number of word deletions.
   * $I$ is the number of word insertions.
   * $N$ is the total number of words in the ground-truth target translation.
   
   WER evaluates the edit distance between the model's prediction and the target translation. A lower WER is better ($0.0$ indicates a perfect match).

2. **BLEU Score** (calculated using `sacrebleu`):
   Computes $N$-gram precision overlap (up to 4-grams) between the generated translations and the ground-truth target sentences. A higher BLEU score is better ($100.0$ indicates a perfect n-gram match).

---

## 📂 Project Structure

```bash
├── data/                  # Local directory for mock/active landmarks (.npz files)
├── results/               # Training checkpoints and output logs
├── scripts/               # Testing and profiling utilities
│   ├── test_gpu.py        # CUDA capability checker
│   ├── test_mediapipe.py  # MediaPipe installation check
│   ├── test_model.py      # End-to-end forward/backward shape check script
│   ├── test_pipeline.py   # Synthesized pipeline simulator
│   └── profile_memory.py  # GPU memory estimator for small/base T5 decoders
├── src/                   # Main source code
│   ├── __init__.py
│   ├── data_pipeline.py   # MediaPipe extractor and formatter
│   ├── dataset.py         # LandmarkDataset & collators (NPZ + JSON OpenPose)
│   ├── train.py           # Seq2SeqTrainer setup and wrapper
│   └── validate_dataset.py # Statistics and noise validation script
├── kaggle_training.ipynb  # Interactive Kaggle GPU/TPU training notebook
├── requirements.txt       # Environment specifications
└── README.md              # Project overview
```

---

## 🛠️ Local Setup & Installation

### 1. Environment Setup
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/yyouretoast/gloss-free-asl-translation.git
cd gloss-free-asl-translation
python -m venv .venv
```
Activate the environment:
* **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
* **Linux/Mac**: `source .venv/bin/activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Verify Compilation
Run the local compilation and forward pass test:
```bash
python scripts/test_model.py
```
Run a local dry-run training pass (executes 1 epoch on mock data):
```bash
python -m src.train --epochs 1 --batch_size 2
```

---

## 📈 Running on Kaggle

To train the model on the full **How2Sign** or **YouTube-ASL** datasets, use the provided `kaggle_training.ipynb` notebook:

1. Create a new notebook on Kaggle.
2. Add the dataset: **How2Sign Keypoints** (e.g., `nazarboholii/how2sign-keypoints`).
3. Import the `kaggle_training.ipynb` file.
4. Execute the cells sequentially. The first cell will automatically pull down all the latest updates directly from this repository:
   ```python
   # Automatically clones or updates the repo to pull latest fixes
   !git pull
   ```
5. Choose your training config:
   * **Full Model (Face Enabled - 411 dimensions)**: Includes critical facial expressions for grammar.
   * **Ablation Model (No Face - 201 dimensions)**: Ignores facial expressions and trains strictly on body/hand trajectories.

---

## 📊 Training Options

The training entry point (`src/train.py`) supports the following customized configuration flags:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | `str` | `data/landmarks` | Path to coordinate folder (.npz or OpenPose JSONs) |
| `--metadata_file` | `str` | `None` | Path to csv uploader metadata for signer-leak proof splits |
| `--epochs` | `int` | `10` | Total training epochs |
| `--batch_size` | `int` | `8` | Size of training batches |
| `--lr` | `float` | `1e-4` | Learning rate |
| `--no_face` | `bool` | `False` | Disables face landmarks to study coordinate ablation (reduces dims) |
| `--t5_model` | `str` | `t5-small` | Hugging Face T5 decoder checkpoint (`t5-small` or `t5-base`) |
