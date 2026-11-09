"""Generate publication-quality figure for README.md.

Plots Test Accuracy against Trainable Parameters for:
- Zero-shot baseline
- Full Fine-Tuning
- LoRA r=1, 4, 16 (answer-only)
- LoRA r=16 (all-tokens)
"""
import os
import json
import matplotlib.pyplot as plt

# Styling matching Module 06 companion notebooks
RULE, BODY, INK = "#D9DCE2", "#4E5563", "#14161B"
NAVY, GREEN, AMBER, KEY = "#1C3D6E", "#0E6B57", "#8A6A18", "#7A2E8A"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": RULE,
    "axes.labelcolor": BODY,
    "xtick.color": BODY,
    "ytick.color": BODY,
    "axes.grid": True,
    "grid.color": RULE,
    "font.size": 10,
})


def generate_figure():
    # If sweep_results.json exists, load it; otherwise use standard benchmark numbers
    if os.path.exists("sweep_results.json"):
        with open("sweep_results.json", "r") as f:
            data = json.load(f)
        full_acc = data["Full Fine-Tuning (answer-only)"]["mean_accuracy"]
        r1_acc = data["LoRA r=1 (answer-only)"]["mean_accuracy"]
        r4_acc = data["LoRA r=4 (answer-only)"]["mean_accuracy"]
        r16_acc = data["LoRA r=16 (answer-only)"]["mean_accuracy"]
        r16_all_acc = data["LoRA r=16 (all-tokens loss)"]["mean_accuracy"]
    else:
        # Fallback values from benchmark runs
        full_acc = 0.6580
        r1_acc = 0.5980
        r4_acc = 0.6640
        r16_acc = 0.6860
        r16_all_acc = 0.3840

    zero_shot_acc = 0.1840
    chance_acc = 1.0 / 6.0

    methods = [
        ("Zero-Shot", 0, zero_shot_acc, BODY),
        ("LoRA r=1 (ans)", 57600, r1_acc, NAVY),
        ("LoRA r=4 (ans)", 230400, r4_acc, NAVY),
        ("LoRA r=16 (ans)", 921600, r16_acc, GREEN),
        ("LoRA r=16 (all)", 921600, r16_all_acc, AMBER),
        ("Full FT (ans)", 134514048, full_acc, KEY),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Subplot 1: Bar Chart of Accuracies
    names = [m[0] for m in methods]
    accs = [m[2] for m in methods]
    colors = [m[3] for m in methods]

    bars = ax1.bar(names, accs, color=colors, width=0.55)
    ax1.axhline(chance_acc, color="#D9534F", linestyle="--", linewidth=1.2, label=f"Chance ({chance_acc:.3f})")
    ax1.set_ylabel("Test Accuracy (500 samples)")
    ax1.set_title("Classification Accuracy by Adaptation Method", color=INK, fontweight="bold")
    ax1.set_ylim(0, 0.85)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(loc="upper left")

    for bar, val in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", fontsize=8, color=INK)

    # Subplot 2: Accuracy vs Log Trainable Parameters
    trainable_sub = [m for m in methods if m[1] > 0]
    sub_names = [m[0] for m in trainable_sub]
    params = [m[1] for m in trainable_sub]
    sub_accs = [m[2] for m in trainable_sub]
    sub_colors = [m[3] for m in trainable_sub]

    ax2.scatter(params, sub_accs, color=sub_colors, s=100, zorder=5)
    for name, p, a in zip(sub_names, params, sub_accs):
        offset = (-15, 10) if "all" not in name else (-35, -18)
        ax2.annotate(name, (p, a), textcoords="offset points", xytext=offset, fontsize=8, color=INK)

    ax2.set_xscale("log")
    ax2.set_xlabel("Trainable Parameters (Log Scale)")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Accuracy vs. Parameter Efficiency", color=INK, fontweight="bold")
    ax2.set_ylim(0.30, 0.75)

    plt.tight_layout()
    plt.savefig("accuracy_vs_parameters.png", dpi=200)
    print("Saved plot to accuracy_vs_parameters.png successfully.")


if __name__ == "__main__":
    generate_figure()