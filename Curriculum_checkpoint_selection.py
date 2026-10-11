import os
import json
import numpy as np
from os.path import exists, join, isdir



SCORE_WEIGHTS = {
    "reward":      0.6,   
    "kl":          0.2,   
    "entropy_gap": 0.2,   
    "val_error":   0.0,   
}

MIN_KL = 0.01   
MAX_KL = 0.20  



def _moving_average(values: np.ndarray, window: int = 50) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel   = np.ones(window) / window
    padded   = np.pad(values, window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def _kl_penalty(kl: float) -> float:
    if kl < MIN_KL:
        return (MIN_KL - kl) / MIN_KL
    if kl > MAX_KL:
        return (kl - MAX_KL) / MAX_KL
    return 0.0


def _step_number(dirname: str) -> int:

    return int(dirname.replace("ppo-checkpoint-", ""))



def select_curriculum_checkpoint(input_directory: str) -> str:

    stats_path = os.path.join(input_directory, "training_stats.jsonl")
    if not exists(stats_path):
        print("No training stats");
        return "base_model"
        
        
    print(f"Loading training stats from: {stats_path}")

    required_keys = [
        "objective/rewards",
        "objective/kl_avg_seq",
        "ppo/policy/entropy",
        "objective/ref_entropies",
        "ppo/val/error",
    ]

    # Each line is one step's stats dict; accumulate into per-key lists.
    all_stats: dict[str, list[float]] = {k: [] for k in required_keys}

    with open(stats_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                step = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {lineno}: {e}") from e

            missing = [k for k in required_keys if k not in step]
            if missing:
                raise KeyError(f"Line {lineno} is missing keys: {missing}")

            for k in required_keys:
                all_stats[k].append(float(step[k]))


    for k in all_stats:
        all_stats[k] = all_stats[k][:]



    
    rewards     = _moving_average(np.array(all_stats["objective/rewards"],       dtype=float))[50:]
    kl          = _moving_average(np.array(all_stats["objective/kl_avg_seq"],    dtype=float))[50:]
    entropy     = _moving_average(np.array(all_stats["ppo/policy/entropy"],      dtype=float))[50:]
    ref_entropy = _moving_average(np.array(all_stats["objective/ref_entropies"], dtype=float))[50:]
    val_error   = _moving_average(np.array(all_stats["ppo/val/error"],           dtype=float))[50:]

    entropy_gap = ref_entropy - entropy   # want small → inverted below

    
    raw_signals = {
        "reward":      rewards,
        "kl":          kl,
        "entropy_gap": entropy_gap,
        "val_error":   val_error,
    }


    print("Rewards 680-750", rewards[680:750])

    signal_keys = ["reward", "kl", "entropy_gap", "val_error"]
    normalized  = {}
    for k in signal_keys:
        norm = _minmax_normalize(raw_signals[k])
        if k in ("kl", "entropy_gap", "val_error"):
            norm = 1.0 - norm
        normalized[k] = norm


    total_weight = sum(SCORE_WEIGHTS.values())
    n_steps      = len(rewards)

    step_scores = np.zeros(n_steps)
    for k in signal_keys:
        step_scores += SCORE_WEIGHTS[k] * normalized[k]
    step_scores /= total_weight


    kl_penalties = np.array([float(v) for v in kl])
    step_scores -= 0.2 * np.minimum(kl_penalties, 1.0)
    print("The step scores are : ", step_scores)
    best_step = int(np.argmax(step_scores))
    print(f"Best step by composite score: {best_step} (score={step_scores[best_step]:.4f})")

    print(
        f"  Best scoring step : {best_step}  "
        f"(score={step_scores[best_step]:.4f}  "
        f"reward={rewards[best_step]:+.3f}  "
        f"kl={kl[best_step]:.4f}  "
        f"entropy_gap={entropy_gap[best_step]:.3f}  "
        f"val_err={val_error[best_step]:.3f})"
    )

    checkpoint_dirs = [
        f for f in os.listdir(input_directory)
        if f.startswith("ppo-checkpoint-")
        and os.path.isdir(os.path.join(input_directory, f))
    ]

    if not checkpoint_dirs:
        print("No checkpoint directories found. Using base model.")
        return "base_model"

    closest_ckpt = min(checkpoint_dirs, key=lambda d: abs(_step_number(d) - (best_step + 50)))
    closest_step = _step_number(closest_ckpt)

    print(f"  Closest checkpoint: {closest_ckpt}  (step {closest_step})")

    return os.path.join(input_directory, closest_ckpt)