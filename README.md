# Modeling Semantic Shifts as Economic Shocks
### Grounding Large Language Models in DSGE Simulators for Policy Generation and Forecasting

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Adi3457/NLP-DSGE)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📌 Intellectual Motivation
Large Language Models (LLMs) are statistical models of language, not structural models of reality. They can produce linguistically plausible policy recommendations that fail catastrophically when evaluated inside formal macroeconomic systems because they lack an internal representation of equilibrium constraints or intertemporal trade-offs.

This research, inspired by **Embodied AI** in robotics, replaces physical simulators with macroeconomic ones. We treat semantic shifts in online discourse as exogenous economic shocks and investigate whether an LLM (**Qwen-2.5-1.5B**) can learn the "physics" of a **Dynamic Stochastic General Equilibrium (DSGE)** model through interaction and Reinforcement Learning.

---

## 🏗️ The Grounding Framework
Our architecture bridges the gap between high-dimensional language and structural economic shocks:
1.  **Semantic Shift Watchdog:** Monitors real-time discourse to capture structural shifts in meaning (Temporal Semantic Volatility).
2.  **DRL Policy Agent:** A Deep Reinforcement Learning agent that receives "semantic shock" vectors and determines policy actions (e.g., interest rate hikes, fiscal adjustments).
3.  **The Environment:** A Python-native DSGE simulator (built on `Snowdrop`) that provides feedback via a custom quadratic loss function.

---

## 🧪 Experimental Paradigms & Algorithms

We explored three primary Reinforcement Learning paradigms to achieve grounding:

### 1. PPO (Proximal Policy Optimization)
The baseline actor-critic approach. We found that PPO collapsed to inaction because the critic produced unstable value estimates due to the "stiffness" of the macroeconomic reward landscape.

### 2. GRPO-1 (Group Relative Policy Optimization)
Removed the critic bottleneck by using relative ranking of sampled trajectories. This improved numerical stability but revealed that the model struggled to associate specific outcomes with policy directions near equilibrium.

### 3. GRPO-2 (The "Sledgehammer" Approach)
Our advanced formulation designed to disambiguate the failure.
- **Crisis Amplification:** Initializing the system in hyperinflation or depression states to break the equilibrium trap.
- **Semantic Action-Tagging:** The LLM emits semantic intents (e.g., `TIGHTEN`, `EMERGENCY`) mapped to deterministic continuous shocks.
- **Persistence:** Shocks are injected with slow decay ($\rho=0.9$) to model intertemporal effects.

---

## 📉 Structural Finding: The "Inversion Paradox"
The core contribution of this work is a rigorous **negative result**. Our calibration grid search proved that the failure of policy-gradient RL in macroeconomics is structural:
- **Delayed Benefits:** Stabilizing actions in DSGE models often increase the sum of squared deviations (loss) in the short run.
- **Reward Misalignment:** Model-free RL agents perceive this "stabilizing pain" as negative feedback, instructing the agent that "fixing the economy" is a bad action.
- **Conclusion:** Macroeconomic physics fundamentally punishes short-horizon model-free RL.

---

## 📂 Repository Manifest

| File | Type | Description |
| :--- | :--- | :--- |
| `GRPO-1.ipynb` | Notebook | Baseline Group Relative Policy Optimization training loop. |
| `GRPO-2.ipynb` | Notebook | Advanced Crisis-Amplified GRPO training with semantic action mapping. |
| `LATEX report.pdf` | Document | The full technical paper detailing methodology and results. |
| **Model Configs** | **YAML** | **Structural DSGE definitions:** |
| `model.yaml` | DSGE | Standard Quarterly Projection Model (QPM). |
| `sw_model.yaml` | DSGE | Smets-Wouters (2007) non-linear business cycle model. |
| `gsw_model.yaml` | DSGE | Gali-Smets-Wouters model (Pandemic labor shocks). |
| `Ireland2004.yaml` | DSGE | Ireland (2004) New Keynesian technology shock model. |
| `MVF_US.yaml` | DSGE | Multivariate Filter for US Output Gap estimation. |
| `RBC.yaml` | DSGE | Real Business Cycle model for growth-focused simulations. |

---

## 🛠️ Setup & Access

### Installation
```bash
pip install torch transformers peft snowdrop-dsge pandas pyyaml
Loading the Fine-tuned Adapter
The model adapters are hosted on Hugging Face and can be applied to the base Qwen model:
code
Python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "Adi3457/NLP-DSGE"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adapter_path)
👥 Authors
Aditya Dubey - f20220231@pilani.bits-pilani.ac.in
Namah Gupta - f20220126@pilani.bits-pilani.ac.in
📜 Citation
code
Bibtex
@article{dubeygupta2025grounding,
  title={Modeling Semantic Shifts as Economic Shocks: Grounding LLMs in DSGE Simulators},
  author={Dubey, Aditya and Gupta, Namah},
  year={2025},
  journal={GitHub Repository},
  url={https://github.com/vasudeywos/NLP-DSGE}
}
