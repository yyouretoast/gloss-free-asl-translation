# Context: Continuous ASL Recognition System

## Goal
Reproduce and evaluate landmark-based gloss-free ASL translation, then quantify the impact of non-manual signals, signer generalization, and isolated-sign pretraining. Grounded in published technique and highly optimized for a solo developer with limited compute.

## Is
- Personal learning project
- Career portfolio artifact: GitHub repo + HF Spaces demo, audience = technical reviewers
- Research-replication of published techniques on real ASL data (such as the baseline in YouTube-ASL and modern Conformer models), focusing on a finishable, high-quality implementation.
- Differentiated on clean engineering: robust landmark extraction, proper sequence-to-sequence masking, and clear evaluation metrics.
- Development Model: Hybrid local/remote execution. Developed locally as modular Python scripts, executed on Kaggle GPUs for landmark extraction and model training.
- Ruthless MVP Success Criteria: 
  1. Validate coordinate extraction on real ASL video samples.
  2. Implement PyTorch dataset loader with geometric normalization, padding, and masking.
  3. Build and train a Conformer Encoder -> Pretrained T5-Small Decoder model.
  4. Perform a rigorous ablation study (comparing include_face=True vs. False) to measure the impact of non-manual signals.
  5. Export model to ONNX format for optimized inference.
  6. Deploy an interactive Gradio web application on Hugging Face Spaces.
  7. Evaluate model using BLEU-4 and WER on validation splits.
- Scoped-Out / Post-MVP Extensions: WLASL pretraining and complex cross-attention fusion modules (retaining lightweight manual vs. face ablation, ONNX serialization, and Gradio web demo in MVP).

## Isn't
- A deployed app with real users
- A claim to solve Deaf communication access — interpreters/VRS/captioning remain the real infrastructure
- Built or validated with Deaf community input
- Reliable enough for real communication — real benchmark SOTA ≈12 BLEU; a solo build lands lower
- An agentic AI demo
- Bidirectional — production (English→sign generation) was scoped out; see Rejected Approaches
- Capable of open-domain vocabulary or full ASL grammar (classifier predicates, depicting verbs)
- The most novel thing in this space — overlaps with much better-resourced industry efforts (e.g. SignGemma); treat as a competence + rigor demonstration, not a contribution

## Public Framing
✅ "Implemented and evaluated a landmark-based gloss-free ASL translation system using MediaPipe landmarks, Conformer encoders, and a pretrained T5 decoder."
❌ "Built an ASL translation system to help deaf people communicate" — overclaims impact and risks reading as the vanity-project pattern Deaf-community surveys flag.

---

## Architecture

### Recognition (Sign → English)
```
Webcam Buffer
     │
     ▼
MediaPipe Holistic Keypoints
     ├── Manual Stream (Hands + Pose) ──┐
     └── Non-Manual Stream (Face Mesh) ─┴─► Geometric Normalization → Concatenation → Linear Projection → Conformer → T5
```
* **Geometric Normalization**: Landmarks are normalized in the dataset loader to be translation and scale invariant:
  * **Pose**: Centered relative to the mid-shoulder point and scaled by shoulder width.
  * **Hands**: Centered relative to their respective wrist coordinates (landmark index 0) to capture translation-invariant hand shapes.
  * **Face**: Centered relative to the mesh centroid.
* **Feature Concatenation**: Manual + non-manual normalized landmarks are concatenated along the feature dimension (258 manual features + 276 face expression features = 534 total features per frame) and projected to `d_model` via a simple Linear layer.
* **Facial Expression Mesh**: Face coordinates are filtered to a 92-landmark subset (eyebrows, eyes, lips) in the extraction script to reduce dimensionality by 80% (configurable back to 468 via a single flag).
- Decoder: Pretrained `t5-small` decoder fine-tuned gloss-free. Using a pretrained language model leverages existing English syntax knowledge, ensuring fast training convergence on Kaggle GPUs.
- Encoder Architecture: Conformer Encoder (Gulati-style blocks) containing depthwise temporal convolutions and self-attention, providing local temporal bias for fast training.

---

## Datasets

| Dataset | Use | Note |
|---|---|---|
| YouTube-ASL | Primary training corpus | ~1000 hrs real ASL |
| How2Sign | Eval / light fine-tune | Standard benchmark |

🚫 Never use Phoenix-2014T — it's German Sign Language (DGS), not ASL.

---

## Rejected Approaches
- **Sign production (English → Sign, Direction B)** — cut entirely. Depends on gloss conversion, and no large parallel English↔ASL-gloss corpus exists (only a tiny one, NCSLGR, 889 videos, or a synthetic one, ASLG-PC12). This is a structural data problem, not an execution risk a feasibility spike would fix. Keeping it in scope meant carrying unresolvable uncertainty through the whole project — cut to fully de-risk the build. Do not reintroduce without a real fix for the gloss-data gap.
- VQ-VAE token bottleneck — high risk, no payoff over gloss-free seq2seq
- Hardcoded fingertip→face graph edges — arbitrary; replaced with cross-attention fusion
- WLASL as translation training data — isolated words can't train sentence translation
- Joint CTC+attention as primary decoder — needs gloss labels the training data doesn't have
- LangGraph agent bolted onto output — shallow, contrived, doubles risk; agent skills are a separate project

---

## Roadmap
```
1. Data pipeline & Extraction — MediaPipe landmarks extraction (pose, hands, 92 expression face landmarks). (Completed)
2. Real-World Data Validation (Phase 0.1) — Profile real coordinate dataset samples (lengths, drops, noise). (Completed)
3. Memory Profiling & Scaling (Phase 0.2) — Run a dry-run training epoch with T5-Small to set VRAM and batch bounds. (Completed)
4. Model Implementation — Build PyTorch dataloaders with geometric normalization, Conformer, and T5-Small wrapper. (Completed)
5. Training & Evaluation — Train model on Kaggle GPUs and evaluate using offline BLEU-4 and WER. (Local dry run verified)
6. Error Analysis — Perform qualitative error analysis on model predictions (identifying failures like speed or occlusion). (Pending)
```

---

## Evaluation
- Held-out-signer eval: Group splits strictly by Signer ID (ensuring validation/test signers are never seen during training). If metadata is missing or incomplete, fallback to standard stratified splits.
- Metrics: Report BLEU-4 and Word Error Rate (WER) to capture semantic and syntactic quality.
- Error Analysis: Catalog failures across visual categories (e.g. occlusion, speed, fingerspelling).

### Project Success Thresholds

| Stage | Success Threshold |
| :--- | :--- |
| **Landmark Extraction** | >95% of target clips processed successfully |
| **Data Validation** | Real-world coordinates profile documented and truncation limits set |
| **End-to-End Model** | Training runs without memory/instability crashes; model maps coordinates to coherent sentences |
| **Evaluation** | Model achieves better-than-trivial BLEU-4 and WER on validation splits |

---

## Limitations
- Constrained vocabulary only, not open-domain
- No classifier predicates / depicting verbs
- No gloss-annotated ASL corpus at scale — this is why production was cut, not just deprioritized
- Multi-month build, not a weekend project

---

## Key References
- Conformer (Gulati et al., Interspeech 2020) — core encoder model architecture basis
- YouTube-ASL (Uthus et al., NeurIPS 2023) — primary training corpus + architecture basis
- Sign2GPT (Wong, Camgöz, Bowden, arXiv 2024) — gloss-free SLT precedent
- Sign Language Transformers (Camgoz, Koller, Hadfield, Bowden, CVPR 2020) — optional future enhancement, needs gloss data not currently used
- How2Sign (Duarte et al., CVPR 2021) — eval benchmark
- WLASL — manual-stream pretraining only
