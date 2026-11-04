"""Figures for the revised manuscript (300 dpi PNG).

    python make_figures.py --results results --data data --out figures
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from data_utils import load_country

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 300})
C = {"train": "#1F3B63", "val": "#E08E0B", "test": "#B03A2E", "act": "#222222", "hdlm": "#1F77B4",
     "arima": "#2CA02C", "prophet": "#9467BD", "lstm": "#8C564B"}
mil = FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M")
kfmt = FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k")


def load_hist(results, tag):
    hs = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(results, f"history_{tag}_*.csv")))]
    L = min(len(h) for h in hs)
    return {c: np.stack([h[c].values[:L] for h in hs]) for c in hs[0].columns if c != "epoch"}, np.arange(1, L + 1)


def band(ax, x, arr, color, label, ls="-"):
    m, s = arr.mean(0), arr.std(0)
    ax.plot(x, m, color=color, lw=1.4, ls=ls, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)


def fig_curves(results, out):
    H, ep = load_hist(results, "HDLM")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    band(ax[0], ep, H["train_loss"], C["train"], "Training loss")
    band(ax[0], ep, H["val_loss"], C["val"], "Validation loss", "--")
    band(ax[0], ep, H["test_loss"], C["test"], "Testing loss")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("MSE (scaled units, log axis)")
    ax[0].set_title("(a) Loss", fontsize=9, loc="left"); ax[0].legend(fontsize=7.5)
    band(ax[1], ep, H["train_acc"], C["train"], "Training accuracy")
    band(ax[1], ep, H["test_acc"], C["test"], "Testing accuracy")
    ax[1].set_ylim(0, 100); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Forecast accuracy, 100(1 − WAPE) %")
    ax[1].set_title("(b) Accuracy", fontsize=9, loc="left"); ax[1].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig22_loss_accuracy_curves.png")); plt.close(fig)


def fig_forecast(results, out):
    p = pd.read_csv(os.path.join(results, "predictions_hdlm.csv"), parse_dates=["date"])
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for k, (f, t) in enumerate([("new_confirmed", "(a) Daily new cases"), ("new_deaths", "(b) Daily deaths"),
                                ("new_recovered", "(c) Daily recoveries")]):
        ax[k].plot(p["date"], p[f"actual_{f}"], color=C["act"], lw=1.1, label="Reported (JHU CSSE)")
        ax[k].plot(p["date"], p[f"pred_{f}"], color=C["hdlm"], lw=1.1, ls="--", label="HDLM one-week-ahead")
        ax[k].set_title(t, fontsize=9, loc="left")
        ax[k].xaxis.set_major_locator(mdates.MonthLocator()); ax[k].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax[k].yaxis.set_major_formatter(kfmt if f != "new_deaths" else FuncFormatter(lambda v, _: f"{v/1e3:.1f}k"))
    ax[0].legend(fontsize=7, loc="upper right"); ax[0].set_ylabel("Count per day")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig23_test_forecasts.png")); plt.close(fig)


def fig_confusion(out, tp=5756454, fn=21, fp=20, tn=202028012):
    M = np.array([[tp, fn], [fp, tn]], dtype=float)
    rowp = M / M.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    shade = np.array([[1, 0.15], [0.15, 1]])
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=1.15)
    lab = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{lab[i][j]}\n{int(M[i, j]):,}\n({rowp[i, j]:.5f}%)", ha="center", va="center",
                    fontsize=9, color="white" if i == j else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nCOVID-19-positive", "Predicted\nnormal"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Ledger\nCOVID-19-positive", "Ledger\nnormal"])
    ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig19_confusion_matrix.png")); plt.close(fig)


def fig_cumulative(data, out):
    df = load_country("India", data, start="2020-03-01", end="2021-09-30")
    for col, name, fc, fname in [("confirmed", "Cumulative confirmed cases", "#F2B134", "fig15_cumulative_confirmed_india.png"),
                                 ("deaths", "Cumulative deaths", "#D9534F", "fig16_cumulative_deaths_india.png")]:
        fig, ax = plt.subplots(figsize=(7.2, 2.5))
        ax.fill_between(df.index, df[col], color=fc, alpha=0.25, lw=0); ax.plot(df.index, df[col], color=fc, lw=1.6)
        ax.set_ylabel(name)
        ax.yaxis.set_major_formatter(mil if col == "confirmed" else kfmt)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_xlim(df.index[0], df.index[-1]); ax.set_ylim(0)
        fig.tight_layout(); fig.savefig(os.path.join(out, fname)); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fig_cumulative(a.data, a.out); fig_confusion(a.out)
    if os.path.exists(os.path.join(a.results, "predictions_hdlm.csv")):
        fig_curves(a.results, a.out); fig_forecast(a.results, a.out)
    print("figures written to", a.out)
