import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# --- Chinese font setup ---
# Try common fonts that support Traditional Chinese on Windows
chinese_fonts = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
available = [f.name for f in fm.fontManager.ttflist]
chosen = next((f for f in chinese_fonts if f in available), None)

if chosen:
    matplotlib.rcParams["font.family"] = chosen
else:
    # Fallback: use sans-serif and hope the system has CJK support
    matplotlib.rcParams["font.family"] = "sans-serif"

matplotlib.rcParams["axes.unicode_minus"] = False  # prevent minus sign corruption

# --- Data ---
categories = ["0\n(非題黨)", "1\n(題黨)"]
chinese_counts = [4731, 11925]
english_counts = [14798, 4740]
chinese_total = sum(chinese_counts)
english_total = sum(english_counts)

x = np.arange(len(categories))
bar_width = 0.35
colors = ["#005EFF", "#FF0000"]  # blue, orange

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
fig.suptitle("語言與標籤分布  (label: 0=非題黨黨, 1=題黨黨)", fontsize=14, fontweight="bold", y=1.01)

for ax, counts, total, title in [
    (axes[0], chinese_counts, chinese_total, "Chinese (WCD)"),
    (axes[1], english_counts, english_total, "English (Webis)"),
]:
    bars = ax.bar(x, counts, width=bar_width * 2, color=colors)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}K" if v >= 1000 else str(int(v))))

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
output_path = "results/label_distribution.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {output_path}")
plt.show()
