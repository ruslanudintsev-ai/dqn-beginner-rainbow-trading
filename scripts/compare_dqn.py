"""
compare_dqn.py — Compare training runs from data/*.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../data"
PLOTS_DIR = "../plots"

def load_stats():
    models = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith("_stats.csv"):
            name = f.replace("_stats.csv", "")
            df = pd.read_csv(os.path.join(DATA_DIR, f))
            models.append({
                "Model": name,
                "Mean": df["mean_reward"].iloc[0],
                "Std": df["std_reward"].iloc[0],
                "Best": df["best_reward"].iloc[0],
            })
    return pd.DataFrame(models)

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_stats()
    if df.empty:
        print("No stats found in data/. Run the training scripts first.")
        return

    base_mean = df.loc[df["Model"] == "01_dqn", "Mean"]
    if not base_mean.empty:
        base = base_mean.iloc[0]
        df["ΔMean vs DQN"] = df["Mean"] - base
    else:
        df["ΔMean vs DQN"] = 0.0

    print("\n" + df.round(2).to_markdown(index=False))
    df.to_csv(os.path.join(DATA_DIR, "compare_summary.csv"), index=False)

    # Simple plot if rewards exist
    plt.figure(figsize=(10, 5))
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith("_avg20.csv"):
            name = f.replace("_avg20.csv", "")
            d = pd.read_csv(os.path.join(DATA_DIR, f))
            plt.plot(d["avg20"], label=name, alpha=0.8)
    plt.legend()
    plt.title("Average Reward (last 20 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Avg20 Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "compare_avg20.png"), dpi=150)
    print(f"\nPlot saved to {PLOTS_DIR}/compare_avg20.png")

if __name__ == "__main__":
    main()
