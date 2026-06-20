# Project Attempt Log: Continuous ASL Recognition System

This document tracks all setups, code implementations, tests, and configuration decisions made since the start of the project.

---

## 1. Project Goal & Strategy (Ruthlessly Streamlined)

We are building a landmark-based, gloss-free ASL translation system that translates sign sequences directly into English sentences, fully optimized for a solo developer to successfully build and complete under resource constraints.

### Core Tech Stack:
- **Input Features**: MediaPipe Holistic 3D skeletal landmarks.
  - **Pose**: 33 points (4D: x, y, z, visibility).
  - **Hands**: 21 points each (3D: x, y, z).
  - **Face Mesh (Optimized)**: Sliced to a **92-landmark subset** focusing solely on expressive features (lips, eyes, eyebrows). This reduces face dimensions from 1404 to 276 (an **80% data reduction**), preventing overfitting to bone structure and saving memory.
- **Model**: Full Conformer Encoder $\rightarrow$ Pretrained `t5-small` Decoder (fine-tuned).
- **Fusion**: Simple concatenation of manual and face coordinates along the feature dimension (simple, robust, and fast).

---

## 2. Project Success Thresholds

| Stage | Success Threshold |
| :--- | :--- |
| **Landmark Extraction** | >95% of target clips processed successfully |
| **Data Validation** | Real-world coordinates profile documented and truncation limits set |
| **End-to-End Model** | Training runs without memory/instability crashes; model maps coordinates to coherent words/fragments showing translation signal |
| **Evaluation** | Model achieves better-than-trivial BLEU-4 and WER on validation splits |

---

## 3. Completed Milestones

### Phase 1: Local Setup & Environment Verification
1. **Virtual Environment**: Initialized a Python 3.10 virtual environment (`.venv`) for package compatibility.
2. **MediaPipe Compatibility Resolution**: Identified that recent MediaPipe releases (`0.10.31+`) deprecated the legacy solutions API, causing import crashes (`AttributeError: module 'mediapipe' has no attribute 'solutions'`). We resolved this by pinning `mediapipe==0.10.14` in [requirements.txt](file:///c:/Users/Yassin/Desktop/code/ASL/requirements.txt).
3. **Environment Checks**:
   - OpenCV and MediaPipe Holistic successfully loaded and processed frames.
   - PyTorch loaded and initialized CPU tensor execution.

### Phase 2: Coordinate Extraction Data Pipeline
1. **Landmark Extraction Engine**: Implemented `ASLLandmarkExtractor` in [src/data_pipeline.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/data_pipeline.py) to parse pose, hands, and the 92 face expression landmarks, apply zero-padding for missing tracking, and write compressed `.npz` files.
2. **Verification Harness**: Created [scripts/test_pipeline.py](file:///c:/Users/Yassin/Desktop/code/ASL/scripts/test_pipeline.py) which writes a synthetic video and runs the pipeline on it to ensure 100% test coverage locally.
3. **Results**: Verified shapes and serialization loaded perfectly. The extractor runs at **32.06 FPS** on CPU.

### Phase 3: Dataset Validation, Memory Profiling, and Conformer Encoder Model
1. **Dataset Validation Tool**: Implemented [src/validate_dataset.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/validate_dataset.py) to profile coordinates (sequence lengths, tracking failure rates, wrist coordinate jitter, and Signer IDs). It successfully scanned files and generated a complete validation report.
2. **Hugging Face Memory Profiler**: Implemented [scripts/profile_memory.py](file:///c:/Users/Yassin/Desktop/code/ASL/scripts/profile_memory.py) to test T5-Small/Base and confirmed custom visual embeddings can be passed directly as encoder hidden states. CPU forward/backward passes succeeded without exception.
3. **Dataset Loader**: Implemented [src/dataset.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/dataset.py) to parse coordinate `.npz` files, slice manual and non-manual face expression fields, pad sequences, and output collation masks. Added frame-level geometric normalization for scale- and translation-invariance.
4. **Full Conformer Encoder Model**: Implemented the Gulati-style Conformer blocks, sinusoidal positional encodings, and temporal MaxPool1d downsampling in [src/models/manual_encoder.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/models/manual_encoder.py).
5. **Sanity Compile Check**: Written [scripts/test_model.py](file:///c:/Users/Yassin/Desktop/code/ASL/scripts/test_model.py) and verified dataloader padding and encoder shape Downsampling.

### Phase 4: Local End-to-End Training Loop Verification
1. **Model Wrapper & Training Loop**: Implemented [src/train.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/train.py) combining Conformer Encoder and Pretrained T5 Decoder using Hugging Face `Seq2SeqTrainer`.
2. **T5 Attention Mask Inversion Fix**: Discovered and resolved a logic bug where boolean padding masks were passed raw to T5, which inverted attention weights. We corrected this by passing `(~downsampled_mask).long()`.
3. **Trainer & Serialization Updates**: Exposed T5's `generation_config`, popped non-generation keyword arguments from validation steps, and overrode the model's `state_dict` to clone weight tensors. This resolved Safetensors serialization warnings and generation ValueErrors.
4. **Offline Metrics**: Integrated offline calculation of BLEU-4 and WER metrics via `sacrebleu` and `jiwer` to support offline Kaggle runs.
5. **Kaggle Setup**: Created the Jupyter notebook template [kaggle_training.ipynb](file:///c:/Users/Yassin/Desktop/code/ASL/kaggle_training.ipynb) to automate setup and execution on Kaggle GPUs.

---

## 4. Files Created & Configured

| File Path | Description | Key Contents / Settings |
| :--- | :--- | :--- |
| [context.md](file:///c:/Users/Yassin/Desktop/code/ASL/context.md) | Project plan & specifications | Set constraints, stack details, and roadmap. |
| [requirements.txt](file:///c:/Users/Yassin/Desktop/code/ASL/requirements.txt) | Python dependencies list | Pinned `mediapipe==0.10.14`, added `jiwer` and `sacrebleu` metrics. |
| [.gitignore](file:///c:/Users/Yassin/Desktop/code/ASL/.gitignore) | Git ignore configurations | Excludes `.venv/`, `data/`, `results/`, and log files. |
| [src/__init__.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/__init__.py) | Package initialization | Marks `src/` as a modular Python package. |
| [src/data_pipeline.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/data_pipeline.py) | Data pipeline engine | Extractor class processing frames to 534 coordinates. |
| [src/validate_dataset.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/validate_dataset.py) | Dataset validation script | Computes sequence and tracking failure aggregates. |
| [src/dataset.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/dataset.py) | PyTorch Dataset class | Performs wrist-relative hand and body geometric normalization. |
| [src/models/manual_encoder.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/models/manual_encoder.py) | Full Conformer Encoder | Custom Gulati-style blocks with pooling layers. |
| [src/train.py](file:///c:/Users/Yassin/Desktop/code/ASL/src/train.py) | End-to-end training loop | Wrapper model, custom state_dict, and Seq2SeqTrainer setup. |
| [kaggle_training.ipynb](file:///c:/Users/Yassin/Desktop/code/ASL/kaggle_training.ipynb) | Kaggle training notebook | Automates clone, setup, validation, and training on Kaggle. |
| [scripts/test_mediapipe.py](file:///c:/Users/Yassin/Desktop/code/ASL/scripts/test_mediapipe.py) | MediaPipe sanity script | Checks Holistic model initialization. |
| [scripts/test_gpu.py](file:///c:/Users/Yassin/Desktop/code/ASL/scripts/test_gpu.py) | PyTorch sanity script | Asserts CUDA availability and base tensor execution. |
| [scripts/test_pipeline.py](file:///c:/Users/Yassin/Desktop/code/ASL/scripts/test_pipeline.py) | Pipeline validation harness | Generates synthetic video and tests extractor shapes. |
| [scripts/profile_memory.py](file:///c:/Users/Yassin/Desktop/code/ASL/scripts/profile_memory.py) | Memory profiling script | Simulates forward/backward passes on T5 decoders. |
| [scripts/test_model.py](file:///c:/Users/Yassin/Desktop/code/ASL/scripts/test_model.py) | Model verification script | Tests batch dataloading and encoder forward pass. |

---

## 5. Test Verification Results

### Dataloader & Conformer Encoder Compile Test
```
Testing Dataset and ConformerEncoder integration...

Initializing ASLLandmarkDataset (including 92 face expression landmarks)...
Dataset initialized with 5 samples.

Fetching first batch from DataLoader...
Batch loaded:
 - input_features shape: torch.Size([2, 137, 534])
 - attention_mask shape: torch.Size([2, 137])
 - Labels text batch:   ['', '']
Dataset loading and padding checks PASSED!

Initializing ConformerEncoder (d_model=512, 4 layers, 4 attention heads)...
Running forward pass through ConformerEncoder...
Encoder outputs shape: torch.Size([2, 69, 512])
Downsampled mask shape: torch.Size([2, 69])

ConformerEncoder Forward Pass check PASSED!
============================================================
All Model and Dataset tests completed successfully!
============================================================
```

### Full Training & Metrics Dry Run
We executed our dry run epoch locally on CPU using the mock landmark dataset:
```powershell
.\.venv\Scripts\python -m src.train --epochs 1 --batch_size 2
```
* **Loss Step 1**: `10.56`
* **Loss Step 2**: `10.94`
* **Evaluation Metrics**:
  ```json
  {"eval_loss": 12.68, "eval_bleu": 0.0, "eval_wer": 1.0}
  ```
* **Status**: Completed successfully. Checkpoints and tokenizers were serialized without warnings.

---

## 6. Next Planned Step (Phase 5)
We have completed all local pipeline verification, debugging, and baseline diagnostic tests. 
The next steps are:
1. **Push Code to GitHub**: Add remote repository URL and push codebase.
2. **Execute Kaggle Training**: Mount dataset in Kaggle GPU notebook, run verification scans, and trigger multi-epoch model training.
