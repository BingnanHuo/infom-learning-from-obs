"""Generate the combined architecture figure for the final report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "report" / "figures" / "cross_modal_infom_architecture.png"


def box(ax, xy, wh, title, body="", face="#f7f9fb", edge="#263238", title_size=11, body_size=8.5):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.12,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
    )
    if body:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.04,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color="#24313a",
            linespacing=1.18,
        )


def label(ax, x, y, text, size=9, weight="normal", color="#24313a"):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, fontweight=weight, color=color)


def arrow(ax, start, end, color="#374151", rad=0.0, lw=1.25, style="-|>", ms=12):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(patch)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 8.8), dpi=240)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    ax.axvspan(0.15, 7.85, color="#f8fafc", alpha=0.9, zorder=0)
    label(ax, 4.0, 8.55, "Cross-modal InFOM proof-of-concept pipeline", 13, "bold")

    # Data and bridge choices.
    box(
        ax,
        (2.9, 7.25),
        (2.2, 0.85),
        "paired bridge data",
        "state $s_t$, RGB $o_t$,\naction $a_t$, next transition\nreward-free",
        face="#dff3f7",
        title_size=10.5,
        body_size=8.0,
    )
    box(
        ax,
        (0.55, 6.45),
        (2.35, 0.85),
        "Method A",
        "RGB encoder -> normalized\nstate-like latent\nMSE state distillation",
        face="#fff2cc",
        title_size=10.5,
        body_size=8.0,
    )
    box(
        ax,
        (5.1, 6.45),
        (2.35, 0.85),
        "Method B",
        "RGB encoder + state MLP\nshared latent\nsymmetric InfoNCE",
        face="#fde2e2",
        title_size=10.5,
        body_size=8.0,
    )

    box(
        ax,
        (2.65, 5.45),
        (2.7, 0.75),
        "grounded latent $h_t$",
        "common representation passed into InFOM losses",
        face="#f9fafb",
        title_size=10.5,
        body_size=8.2,
    )

    # InFOM stages.
    box(
        ax,
        (0.55, 3.95),
        (2.15, 1.0),
        "reward-free pretraining",
        "flow occupancy\n+ BC actor\n+ bridge loss",
        face="#e0f2fe",
        title_size=10.2,
        body_size=8.4,
    )
    box(
        ax,
        (2.95, 3.95),
        (2.1, 1.0),
        "InFOM intention proxy",
        "$q(\\eta \\mid h_{t+1}, a_{t+1})$\nfuture-occupancy latent\nnot semantic intent",
        face="#e0e7ff",
        title_size=10.2,
        body_size=8.0,
    )
    box(
        ax,
        (5.3, 3.95),
        (2.15, 1.0),
        "reward-labeled FT",
        "TPV RGB -> $h_t$\nreward model, critic,\nflow, actor updates",
        face="#dcfce7",
        title_size=10.2,
        body_size=8.0,
    )

    # Deployment.
    box(
        ax,
        (0.55, 2.45),
        (2.2, 0.9),
        "deployment input",
        "learner-side low-dimensional state\n(no TPV RGB at evaluation)",
        face="#ece7fb",
        title_size=10.2,
        body_size=8.0,
    )
    box(
        ax,
        (2.95, 2.45),
        (2.1, 0.9),
        "state path",
        "fixed state normalizer\nor learned state encoder",
        face="#ede9fe",
        title_size=10.2,
        body_size=8.0,
    )
    box(
        ax,
        (5.3, 2.45),
        (2.15, 0.9),
        "policy action",
        "actor outputs 5-D control:\nend-effector xyz, yaw, gripper",
        face="#ede9fe",
        title_size=10.2,
        body_size=8.0,
    )

    # Boundary.
    box(
        ax,
        (0.8, 0.95),
        (6.4, 0.85),
        "claim boundary",
        "paired proof of concept: no unpaired third-person transfer,\nno semantic-intent proof, and no learned causal model",
        face="#f8fafc",
        edge="#64748b",
        title_size=10.5,
        body_size=8.2,
    )

    # Arrows.
    arrow(ax, (3.5, 7.25), (1.72, 7.3), color="#0f766e", rad=0.12)
    arrow(ax, (4.5, 7.25), (6.25, 7.3), color="#b91c1c", rad=-0.12)
    arrow(ax, (1.72, 6.45), (3.25, 6.2), color="#64748b", rad=-0.05)
    arrow(ax, (6.25, 6.45), (4.75, 6.2), color="#64748b", rad=0.05)
    arrow(ax, (4.0, 5.45), (1.62, 4.95), color="#2563eb", rad=0.08)
    arrow(ax, (4.0, 5.45), (4.0, 4.95), color="#2563eb")
    arrow(ax, (4.0, 5.45), (6.38, 4.95), color="#16a34a", rad=-0.08)
    arrow(ax, (2.7, 4.45), (2.95, 4.45), color="#2563eb")
    arrow(ax, (5.05, 4.45), (5.3, 4.45), color="#16a34a")
    arrow(ax, (2.75, 2.9), (2.95, 2.9), color="#7c3aed")
    arrow(ax, (5.05, 2.9), (5.3, 2.9), color="#7c3aed")
    arrow(ax, (6.38, 3.95), (6.38, 3.35), color="#64748b")
    arrow(ax, (6.38, 2.45), (5.95, 1.8), color="#64748b", rad=-0.1)

    label(ax, 0.92, 5.45, "1M pretraining", 8.5, "bold", "#2563eb")
    label(ax, 6.95, 5.45, "500K FT", 8.5, "bold", "#16a34a")
    label(ax, 4.0, 2.05, "Bridge extension over standard shared-interface InFOM", 9.5, "bold", "#374151")

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
