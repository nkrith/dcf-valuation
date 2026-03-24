from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


@dataclass(frozen=True)
class HeatmapSpec:
    title: str
    x_labels: List[str]
    y_labels: List[str]
    market_price: Optional[float] = None  # if set, color cells green/red vs market


def plot_heatmap(
    matrix: List[List[float]],
    spec: HeatmapSpec,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    data = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, len(spec.x_labels) * 1.6), max(4, len(spec.y_labels) * 1.0)))

    # Use diverging colormap centred on market price if it falls within the data range
    norm = None
    cmap = "YlGnBu"
    if spec.market_price and spec.market_price > 0:
        vmin, vmax = float(data.min()), float(data.max())
        center = spec.market_price
        if vmin < center < vmax:
            # Market price is inside the data range — use diverging green/red
            norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            cmap = "RdYlGn"
        else:
            # Market price outside range — all cells are one side (all red or all green)
            cmap = "Reds_r" if center > vmax else "Greens"

    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    ax.set_title(spec.title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Terminal Growth (g)", fontsize=10)
    ax.set_ylabel("WACC", fontsize=10)
    ax.set_xticks(range(len(spec.x_labels)))
    ax.set_xticklabels(spec.x_labels, fontsize=9)
    ax.set_yticks(range(len(spec.y_labels)))
    ax.set_yticklabels(spec.y_labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, label="Intrinsic Price ($)", shrink=0.8)
    if spec.market_price:
        ax.set_title(f"{spec.title}\nMarket price: ${spec.market_price:,.2f}",
                     fontsize=12, fontweight="bold", pad=12)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"${val:,.0f}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="black")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    # Auto-open the saved image in the system viewer
    if save_path:
        import os, platform, subprocess
        try:
            if platform.system() == "Windows":
                os.startfile(os.path.abspath(save_path))
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", save_path])
            else:
                subprocess.Popen(["xdg-open", save_path])
        except Exception:
            pass  # fail silently if no viewer available
