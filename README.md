Here is the raw `.md` code for your GitHub repository. I have incorporated the authors, the Hugging Face link, the files shown in your screenshot, and the technical findings from your report.

You can copy this directly into a file named `README.md` in your repository.

```markdown
# Modeling Semantic Shifts as Economic Shocks
### Grounding Large Language Models in DSGE Simulators for Policy Generation

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Adi3457/NLP-DSGE)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This project investigates whether Large Language Models (LLMs) can be grounded in the structural "physics" of macroeconomics. While LLMs are excellent at analyzing economic narratives, they lack an internal understanding of equilibrium constraints and intertemporal dynamics.

We bridge this gap by embedding a locally deployed **Qwen-2.5-1.5B** model inside a **Dynamic Stochastic General Equilibrium (DSGE)** simulator. We operationalize semantic shifts in online discourse as exogenous economic shocks and attempt to train the LLM via Reinforcement Learning (RL) to generate stabilizing policy actions.

## 🏗️ The Grounding Framework
Inspired by **Embodied AI** in robotics, we replace physical simulators with macroeconomic ones:
- **Environment:** A Python-native DSGE interface built on the `Snowdrop` framework.
- **Agent:** Qwen-2.5-1.5B fine-tuned via LoRA.
- **Input:** Semantic shocks (Temporal Semantic Volatility) extracted from financial text.
- **Action Space:** Semantic policy descriptors (e.g., "Aggressive Tightening", "Emergency Ease") mapped to numeric structural shocks.

## 🧪 Experimental Paradigms
We conducted a systematic study across three primary RL architectures:
1. **PPO (Proximal Policy Optimization):** Baseline actor-critic approach. Suffered from reward sparsity and critic instability.
2. **GRPO-1 (Group Relative Policy Optimization):** Eliminated the critic by using relative ranking of sampled trajectories. Improved numerical stability but failed to accumulate learning.
3. **GRPO-2 (Crisis-Amplified):** Our advanced formulation using "Sledgehammer" techniques:
   - **Crisis Amplification:** System initialized in hyperinflation or depression states.
   - **Persistence:** Policy actions injected with slow decay ($\rho=0.9$).
   - **Inaction Penalties:** Explicitly penalizing the agent for "holding" during volatile states.

## 📉 Key Results: The "Inversion Paradox"
Our research provides a definitive **negative result**. We discovered that the failure of RL in this domain is structural:
- **Reward Misalignment:** Stabilizing actions in DSGE models often increase quadratic loss in the short-run before providing long-term benefits.
- **Inversion:** Policy-gradient RL perceives this "stabilizing pain" as negative feedback, causing the agent to learn that "doing nothing" is optimal.
- **Conclusion:** Model-free RL is fundamentally incompatible with the intertemporal stiffness of macroeconomic systems.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `GRPO-1.ipynb` | Implementation of Group Relative Policy Optimization (Baseline). |
| `GRPO-2.ipynb` | Enhanced iteration with Crisis Amplification and Semantic Action Mapping. |
| `LATEX report.pdf` | Full technical paper detailing the research, methodology, and results. |
| **Model Configs** | **Structural YAML definitions for the DSGE Simulator:** |
| `model.yaml` | Quarterly Projection Model (QPM) - Standard Central Bank model. |
| `sw_model.yaml` | Smets-Wouters (2007) model for Business Cycle analysis. |
| `gsw_model.yaml` | Gali-Smets-Wouters model (COVID/Pandemic labor shocks). |
| `Ireland2004.yaml` | Ireland (2004) New Keynesian Technology shock model. |
| `MVF_US.yaml` | Multivariate Filter for US Output Gap estimation. |
| `RBC.yaml` | Real Business Cycle model for growth-focused simulations. |

---

## 🛠️ Installation & Usage

### 1. Requirements
- Python 3.9+
- [Snowdrop DSGE Framework](https://github.com/SimonHashtag/EconRL)
- HuggingFace `transformers`, `peft`, and `trl`

```bash
pip install torch transformers peft trl pyyaml pandas numpy
```

### 2. Accessing the Model
The fine-tuned policy agent is available on Hugging Face:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "Adi3457/NLP-DSGE"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_id)
```

---

## 👥 Authors
- **Aditya Dubey** - [f20220231@pilani.bits-pilani.ac.in](mailto:f20220231@pilani.bits-pilani.ac.in)
- **Namah Gupta** - [f20220126@pilani.bits-pilani.ac.in](mailto:f20220126@pilani.bits-pilani.ac.in)

```
```
