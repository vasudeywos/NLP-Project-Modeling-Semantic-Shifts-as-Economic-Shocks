Modeling Semantic Shifts as Economic Shocks 📉🤖
Grounding Large Language Models in DSGE Simulators for Policy Generation

![alt text](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)


![alt text](https://img.shields.io/badge/License-MIT-blue.svg)

This repository contains the code and structural models for a research project exploring how Large Language Models (LLMs) can be grounded in the structural laws of macroeconomics. By treating semantic shifts in online discourse as exogenous economic shocks, we attempt to train an LLM-based policy agent using Reinforcement Learning (RL) within a Dynamic Stochastic General Equilibrium (DSGE) simulator.

Project Overview

Current LLMs are statistical models of language, not structural models of reality. They can generate coherent economic prose but lack an internal representation of equilibrium constraints or intertemporal trade-offs.

Inspired by Embodied AI (where robots learn physics through simulators), this project uses a DSGE simulator as a "Macro-Gym" to ground an LLM (Qwen-2.5-1.5B) in economic structure.

Key Finding: The "Inversion Paradox"

Our research provides a rigorous negative result: Policy-gradient RL (PPO and GRPO) fails to achieve stable grounding in DSGE environments. We identify that DSGE models are "stiff" and equilibrium-restoring; stabilizing actions often cause short-term "pain" (volatility) that standard RL loss functions penalize, leading the agent to learn a policy of total inaction.

Architecture

The system consists of two tiered modules:

Semantic Shift Watchdog: Translates Temporal Semantic Volatility (TSV) from financial discourse into numerical "Semantic Shock" vectors.

DRL Policy Agent: A fine-tuned LLM that observes the economy and generates policy interventions (Monetary/Fiscal shocks) to minimize a quadratic loss function.

Methodology & Experiments

We tested three distinct Reinforcement Learning paradigms to bridge the language-to-simulator gap:

PPO (Proximal Policy Optimization): Collapsed to inaction due to critic instability and the "stiffness" of the macroeconomic reward landscape.

GRPO-1 (Group Relative Policy Optimization): Used relative ranking of trajectories to remove the critic. While variance was detected, learning failed to accumulate across epochs.

GRPO-2 (Crisis-Amplified): Our most aggressive formulation.

Semantic Action-Tagging: LLM emits semantic intents (e.g., TIGHTEN, EASE) instead of raw numbers.

Crisis Amplification: Initializing the system in hyperinflation/depression states to force a learning signal.

Inaction Penalties: Explicitly penalizing the agent for doing nothing during crises.

📂 Repository Structure
File	Description
GRPO-1.ipynb	Initial implementation of Group Relative Policy Optimization.
GRPO-2.ipynb	Enhanced iteration with crisis-amplified environments and semantic mapping.
model.yaml	Base Quarterly Projection Model (QPM) configuration.
sw_model.yaml	Non-linear Smets-Wouters business cycle model.
gsw_model.yaml	Gali-Smets-Wouters model (Pandemic/COVID focus).
Ireland2004.yaml	New Keynesian Tech-shock model.
MVF_US.yaml	Multivariate Filter for US Output Gap estimation.
RBC.yaml	Real Business Cycle model for growth-focused simulations.
LATEX report.pdf	Full technical paper detailing findings and the "Inversion Paradox."
🛠️ Setup & Installation

Prerequisites:

Python 3.9+

Snowdrop DSGE Framework

PyTorch & HuggingFace Transformers

code
Bash
download
content_copy
expand_less
# Clone the repository
git clone https://github.com/vasudeywos/NLP-DSGE-Grounding.git

# Install dependencies
pip install torch transformers peft snowdrop-dsge
Loading the Fine-tuned Model

The policy agent model is hosted on Hugging Face:

code
Python
download
content_copy
expand_less
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Adi3457/NLP-DSGE"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
Key Results from Calibration

Our manual grid search (bypassing the LLM) revealed why RL fails in this domain:

Scenario	Policy Action	Reward Result	Interpretation
Recession	Rate Cut	❌ FAIL	Short-term inflation spikes outweigh long-term output gains in the loss function.
Stagflation	Rate Hike	✅ PASS	Only works when the objective function is heavily weighted toward "Hawkish" inflation control.
Cost-Push	Any	❌ FAIL	DSGE short-run dynamics punish nearly all interventions under supply shocks.


Aditya Dubey - GitHub

Namah Gupta
