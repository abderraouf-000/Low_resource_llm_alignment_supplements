# Supplementary Material

This archive contains the implementation used for the experiments presented in the accompanying AAAI submission.

## Contents

- Source code for synthetic preference data generation.
- All generated preference data across all stages needed for training the reward model.
- Reward model training code.
- PPO policy optimization code.
- Evaluation dataset generation scripts (Except TruthfulQA, all of evaluation code are found in their respective repositories).
- Configuration files and execution scripts.
- Hyperparameter settings used in the experiments, are described in the technical supplementary material.
- Python dependencies (`requirements.txt`).

Model checkpoints across stages are **not included** due to the AAAI supplementary material size limit (50 MB). 
But, can be submitted after annonymity is lifted.

---

## Environment

Install the required packages with

```bash
pip install -r requirements.txt
```

If downloading models from Hugging Face, set the appropriate authentication token:

```bash
export HF_TOKEN=<your_token>
```

---

## Running the Pipeline

The entire training pipeline can be executed using

```bash
python Experiments_run.py
```

The pipeline performs the following stages:

1. Synthetic preference data generation (You can change to the pre-generated preference data by modifying the `Experiments_run.py` script).
2. Preference data post-processing.
3. Reward model training.
4. PPO policy optimization.
5. Evaluation.
6. Automatic checkpoint selection for the next curriculum iteration.

---

## Running Individual Components

### Synthetic preference generation

```bash
bash scripts/synthetic_preference_data_generation.sh
```

### Reward model training

```bash
bash scripts/reward_model_training_script_run.sh
```

### PPO training

```bash
bash scripts/ppo_training_run.sh
```

### Evaluation

```bash
bash scripts/run_evaluation_generations.sh
```

Script arguments can be modified directly in the corresponding shell scripts or supplied through the command line.

---

## Data

The repository expects instruction-following datasets together with the structural annotations described in the paper.

Generated synthetic preference data follows the standard preference-learning format consisting of instruction, chosen response, rejected response, and associated metadata.

---

## Evaluation

The evaluation scripts reproduce generations for the benchmarks reported in the paper, including:

- AlpacaEval
- IFEval
- TruthfulQA
- HumanEval

---

