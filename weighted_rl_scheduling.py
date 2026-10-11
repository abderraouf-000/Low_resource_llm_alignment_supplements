import json
import numpy as np
import fire


def load_jsonl(file_path):
    """Load a JSONL file and return a list of dicts."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_structural_stats(data):
    """
    Compute per-structural-class statistics: mean, std, count of
    abs(positive_distance - negative_distance)
    """
    structural_classes = {}
    for item in data:
        structural_class = item.get("structural_class")
        if structural_class is None or item.get("contrastive_status") is None:
            continue

        positive = item.get("positive_distance")
        negative = item.get("negative_distance")
        if positive is None or negative is None:
            continue

        diff = abs(positive - negative)
        structural_classes.setdefault(structural_class, []).append(diff)

    stats = {}
    for cls, values in structural_classes.items():
        values_np = np.array(values)
        stats[cls] = {
            "mean": np.mean(values_np),
            "std": np.std(values_np),
            "count": len(values_np)
        }
    return stats

def compute_sampling_probs(stats, alpha=0.7, include_count=True):
    """
    Compute normalized sampling probabilities for each structural class
    based on normalized mean and std
    """
    classes = list(stats.keys())
    means = np.array([stats[c]["mean"] for c in classes])
    stds = np.array([stats[c]["std"] for c in classes])
    counts = np.array([stats[c]["count"] for c in classes])

    # Min-max normalization
    mean_norm = (means - means.min()) / (means.max() - means.min())
    std_norm = (stds - stds.min()) / (stds.max() - stds.min())

    # Combine with alpha weighting
    values = alpha * mean_norm + (1 - alpha) * std_norm

    # Optionally scale by count
    if include_count:
        values = values * (counts / counts.max())

    # Normalize to probabilities
    probs = values / values.sum()

    # Return dict
    prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
    return prob_dict

def main(jsonl_file, alpha=0.7):
    """
    Compute sampling probabilities for structural classes from a JSONL file.

    Args:
        jsonl_file (str): Path to the JSONL file
        alpha (float): Weight for mean in score combination (0-1)

    Returns:
        dict: Sampling probabilities per structural class
    """
    data = load_jsonl(jsonl_file)
    stats = compute_structural_stats(data)
    prob_dict = compute_sampling_probs(stats, alpha=alpha)

    print("Per-class statistics:")
    for cls, s in stats.items():
        print(f"{cls}: Mean={s['mean']:.4f}, Std={s['std']:.4f}, Count={s['count']}")

    print("\nSampling probabilities:")
    for cls, p in prob_dict.items():
        print(f"{cls}: {p:.4f}")

    return prob_dict

if __name__ == "__main__":
    fire.Fire(main)