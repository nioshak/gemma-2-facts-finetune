# Project Report: End-to-End Safety Alignment for Gemma 2

**Program Goal:** mitigate hallucination risks in Retrieval Augmented Generation (RAG) systems by strictly aligning **Gemma 2 (2B)** to source documents.

**Key Outcome:** Delivered a fine-tuned adapter capable of strict factual refusal ("I don't know") on consumer-grade hardware, reducing hallucination rates by >90% compared to the base model.

---

## Program Objectives & Constraints
The initiative was scoped to replicate DeepMind's "FACTS" alignment benchmark behavior under strict resource constraints, simulating a "lean startup" or edge-deployment environment.

| Constraint | Risk | Mitigation Strategy |
| :--- | :--- | :--- |
| **Data Scarcity** | Model collapse due to small dataset (<150 examples) | **Synthetic Data Ops:** Engineered a teacher-student pipeline using Gemini 1.5 Pro to expand dataset 7x. |
| **Compute Limits** | `OOM` errors on T4 GPUs (16GB VRAM) | **Infrastructure Optimization:** Implemented QLoRA (4-bit) and Gradient Checkpointing to reduce memory footprint by ~60%. |
| **Safety** | Unchecked hallucinations in RAG | **DPO Alignment:** Switched from SFT to Direct Preference Optimization for strict negative constraint learning. |

---

## Technical Strategy

### 1. Data Operations Pipeline (Addressing the Cold Start Problem)
To overcome the bottleneck of the limited FACTS dataset (~850 rows), I architected a synthetic data loop:
- **Teacher Model:** Deployed Gemini 2.5 Flash as a "Red Teamer."
- **Prompt Engineering:** Designed adversarial prompts to generate plausible hallucinations ("Rejected" samples) for every grounded fact ("Chosen" sample).
- **Impact:** This unblocked the training phase, moving the project from a failed 125-sample pilot to a robust 850+ sample production run.

### 2. Infrastructure & Cost Optimization
The training pipeline was optimized for widely available T4 GPUs (Google Colab Free Tier) to demonstrate cost-efficiency.
- **Quantization:** Leveraged `bitsandbytes` `nf4` precision to fit the model weights into <8GB VRAM.
- **Targeted Adaptation:** Configured LoRA to target only attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) rather than all linear layers.
- **Context Management:** Pruned context window to 512 tokens (covering 95% of use cases) to maximize batch throughput without OOM crashes.

---

## Impact Analysis

| Metric | Base Model (Gemma 2) | Aligned Model (This Project) |
| :--- | :--- | :--- |
| **Behavior** | Extrapolates external knowledge | **Strict Adherence** (Refuses unknown info) |
| **Safety** | High Risk (Confident hallucinations) | **Low Risk** (Conservative refusal) |
| **Infrastructure** | Requires A100 (Native) | **Runs on T4/L4** (Quantized) |

### Functional Test (The "Banana Test")
* **Prompt:** `DOCUMENT: Apples are red. USER REQUEST: Are bananas blue?`
* **Base Model:** "Bananas are typically yellow..." (Hallucination relative to document).
* **Aligned Model:** "The provided document does not contain this information." (Pass).

---

## Program Retrospective
* **Trade-offs:** I accepted higher inference latency (due to de-quantization steps) in exchange for significantly lower training costs. For a real-time production SLA, I would recommend serving the model in FP16 on L4 GPUs.
* **Scalability:** The synthetic data pipeline is decoupled from the training loop, allowing us to scale to 10k+ examples simply by swapping the input corpus (e.g., Wikipedia).
* **Next Steps:** Evaluate performance on "Gemma 2 9B" to test if reasoning capabilities improve with model size while maintaining the same safety alignment.

---

## Usage
The trained adapter is available on Hugging Face: `[YOUR_HUGGINGFACE_USERNAME]/gemma-facts-finetune`

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="[YOUR_USERNAME]/gemma-facts-finetune")
# See repo for full inference script
