# Gloss-Free ASL-to-English Translation

A deep learning framework to translate American Sign Language (ASL) video coordinates directly into fluent English sentences without using intermediate gloss representations. The project connects a custom **Conformer Encoder** directly to a pretrained **Hugging Face T5 Decoder** to perform end-to-end visual-to-text sequence generation.

---

## 🚀 Key Achievements & Architectural Features

### 1. Robust Coordinates Processing Pipeline
* **Targeted Landmark Extraction (`src/data_pipeline.py`)**: Uses MediaPipe to extract keypoints across Pose, Face, and Hands. Slices face landmarks to a optimized **92-point facial subset** (eyebrows, eyes, lips) instead of the full 468 points, reducing coordinate dimensionality by **80%** (from 1404 to 276 dimensions) while preserving critical grammatical facial expressions.
* **Scale & Distance Invariance (`src/dataset.py`)**: Centers coordinate frames around the mid-shoulder point and scales landmarks based on shoulder width. This ensures model invariance to camera distance, signer height, and camera movement.
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
│   └── validate_dataset.py# Statistics and noise validation script
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
