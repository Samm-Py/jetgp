"""
Plot the naive-vs-CostAware comparison from
data/hypad_naive_vs_costaware.json.

Produces a per-seed scatter on a log-y axis with median and IQR overlay for
each method. Writes the figure to
docs/source/_static/hypad_naive_vs_costaware.png.
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hypad_naive_vs_costaware.json")
    parser.add_argument(
        "--output",
        default="../docs/source/_static/hypad_naive_vs_costaware.png")
    args = parser.parse_args()

    payload = json.load(open(args.input))
    nv = sorted(payload["naive"], key=lambda r: r["seed"])
    ca = [r for r in payload["costaware"] if r.get("n_init") == 2]
    ca = sorted(ca, key=lambda r: r["seed"])
    assert len(nv) == len(ca) and all(n["seed"] == c["seed"]
                                        for n, c in zip(nv, ca)), \
        "naive and costaware seeds do not match"
    seeds = [r["seed"] for r in nv]

    nv_rmse = np.array([r["rmse"] for r in nv])
    ca_rmse = np.array([r["rmse"] for r in ca])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x_nv = np.full(len(nv_rmse), 0.0) + (np.arange(len(nv_rmse)) - 4.5) * 0.02
    x_ca = np.full(len(ca_rmse), 1.0) + (np.arange(len(ca_rmse)) - 4.5) * 0.02
    ax.scatter(x_nv, nv_rmse, c="#d62728", s=40, alpha=0.7, label="naive seeds")
    ax.scatter(x_ca, ca_rmse, c="#1f77b4", s=40, alpha=0.7,
               label="CostAware seeds")

    for x, vals, color in [(0.0, nv_rmse, "#d62728"),
                            (1.0, ca_rmse, "#1f77b4")]:
        med = float(np.median(vals))
        p25, p75 = (float(np.percentile(vals, 25)),
                    float(np.percentile(vals, 75)))
        ax.plot([x - 0.2, x + 0.2], [med, med], color=color, lw=3.0)
        ax.add_patch(plt.Rectangle((x - 0.15, p25), 0.3, p75 - p25,
                                     fill=False, ec=color, lw=1.5))

    ax.set_yscale("log")
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Naive\n(n=2 LHS + full ∇f)",
                        "CostAware\n(n_init=2, B=9)"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("test RMSE")
    ax.set_title("HYPAD-UQ Case 1, AD regime — total cost ≈ 9\n"
                 "(both methods start from same 2-point LHS DOE)")
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
