import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W_PX, H_PX, DPI = 640, 140, 100

fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_PX)
ax.set_ylim(0, H_PX)
ax.axis("off")

stages = [
    ("Pass 1  Proposal",
     "40 ms event count map\nadaptive percentile (cap 99.0)\nconnected components"),
    ("Pass 2  Candidate scoring",
     "13 geometric / temporal features\nHistGB classifier, ROC-AUC 0.930\npersistence filter"),
    ("Pass 3  Window objectness",
     "21 features over 3 windows\nHistGB classifier, ROC-AUC 0.889\nconfidence x p_obj"),
    ("Pass 4  Emission",
     "NMS, confidence floor, top-K\ndual log-HGBR box regressor\nrank order preserved"),
]

left, right = 12.0, 628.0
gap = 14.0
n = len(stages)
bw = (right - left - gap * (n - 1)) / n
by, bh = 36.0, 78.0

ax.text(left, 130, "Input: 40 ms event window  (EVK4 1280x720 / DVX 640x480 / DAVIS 346x260)",
        fontsize=6.0, va="center", ha="left", color="#133c55")
ax.text(right, 130, "Output: <sequence>.txt + Evaluation_Metrics.xlsx",
        fontsize=6.0, va="center", ha="right", color="#133c55")

for i, (title, detail) in enumerate(stages):
    x = left + i * (bw + gap)
    ax.add_patch(FancyBboxPatch(
        (x, by), bw, bh,
        boxstyle="round,pad=0,rounding_size=4",
        linewidth=0.8, edgecolor="#205072", facecolor="#f2f4f7"))
    ax.text(x + bw / 2, by + bh - 13, title,
            fontsize=6.5, fontweight="bold", ha="center", va="center", color="#0b2545")
    ax.text(x + bw / 2, by + bh / 2 - 9, detail,
            fontsize=5.5, ha="center", va="center", color="#111", linespacing=1.5)
    if i < n - 1:
        ax.annotate("", xy=(x + bw + gap - 2, by + bh / 2),
                    xytext=(x + bw + 2, by + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", linewidth=0.9, color="#205072"))

ax.text(W_PX / 2, 15,
        "CPU-only, offline  |  three gradient-boosted tree models, 1.5 MB total  |  "
        "compute p99 < 40 ms on 18 of 21 sequences  |  RSS 114.3-130.7 MB",
        fontsize=6.0, ha="center", va="center", color="#133c55")

fig.savefig("experiments/frames/fig2_pipeline.png", dpi=DPI,
            facecolor="white", pad_inches=0)
print("WROTE experiments/frames/fig2_pipeline.png")
