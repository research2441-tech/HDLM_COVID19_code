import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "mathtext.fontset": "dejavuserif"})


def box(ax, x, y, w, h, text, fc="#EEF3FA", ec="#1F3B63", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", color="#1F3B63", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------- Figure 1: single-patient data generation ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 14.4); ax.set_ylim(0, 6.6); ax.axis("off")
box(ax, 0.2, 2.6, 1.8, 1.2, "Patient\n(N = 1)", fc="#FFF4D6", ec="#8A6D1F", bold=True)
srcs = ["Testing centre\n(RT-PCR, ID)", "CT / X-ray\ncentre", "Laboratory\n(blood tests)", "Bedside monitor\n(SpO$_2$, HR, BP)"]
for i, s in enumerate(srcs):
    y = 5.1 - i * 1.45
    box(ax, 3.0, y, 2.4, 1.05, s, fs=8.5)
    arrow(ax, 2.0, 3.2, 3.0, y + 0.52)
    arrow(ax, 5.4, y + 0.52, 6.6, 3.2)
box(ax, 6.5, 2.45, 2.5, 1.5, "Treating doctor\n(clinical notes,\nseverity grade)", fc="#E8F5E9", ec="#2E6B34", fs=8.3)
arrow(ax, 9.0, 3.2, 9.7, 3.2)
box(ax, 9.7, 2.45, 2.2, 1.5, "Patient record\n$P_i=\\{t_i, mi_i,$\n$bm_i, ot_i\\}$", fc="#F3E8F7", ec="#5B2C6F", fs=8.3)
arrow(ax, 11.9, 3.2, 12.5, 3.2)
box(ax, 12.5, 2.45, 1.85, 1.5, "Cloud /\nweb registry\n(crowd-\nsourced)", fc="#E3F2FD", ec="#0D47A1", fs=8)
arrow(ax, 5.4, 5.62, 13.4, 3.95, ls="--", rad=-0.3, color="#555555")
ax.text(9.3, 6.35, "device data uploaded directly (duplicates arise here)", fontsize=8, color="#555555", ha="center")
fig.tight_layout(); fig.savefig("fig1_patient_data_generation.png", dpi=300); plt.close(fig)

# ---------------- Figure 9: forward / backward flow of the record CNN ----------------
fig, ax = plt.subplots(figsize=(7.4, 3.0)); ax.set_xlim(0, 15); ax.set_ylim(0, 6); ax.axis("off")
steps = [("Input record\ntensor $X$\n224×224×3", "#FFF4D6"),
         ("Conv 3×3\n+ ReLU", "#EEF3FA"),
         ("Max-pool\n2×2 +\nDropout", "#EEF3FA"),
         ("×3 further\nconv blocks", "#EEF3FA"),
         ("Flatten\n86,528", "#EEF3FA"),
         ("Dense 64\n+ ReLU", "#EEF3FA"),
         ("Dense 1\n+ sigmoid", "#E8F5E9"),
         ("Loss\n$\\mathcal{L}_{BCE}$", "#FDECEA")]
xs = [0.1 + i * 1.86 for i in range(len(steps))]
for (t, c), x in zip(steps, xs):
    box(ax, x, 3.3, 1.66, 1.5, t, fc=c, fs=7.4)
for i in range(len(xs) - 1):
    arrow(ax, xs[i] + 1.66, 4.05, xs[i + 1], 4.05)
eqs = ["$X$", "$Z^{(l)}=\\mathrm{ReLU}(W^{(l)}\\ast A^{(l-1)}+b^{(l)})$", "$A^{(l)}=\\max_{2\\times2}Z^{(l)}$",
       "", "$a=\\mathrm{vec}(A^{(4)})$", "$h=\\mathrm{ReLU}(W_dа+b_d)$".replace("а", "a"),
       "$\\hat{y}=\\sigma(w_o^{T}h+b_o)$", ""]
ax.text(xs[1] + 0.8, 1.85, eqs[1], ha="center", fontsize=8)
ax.text(xs[2] + 0.9, 1.2, eqs[2], ha="center", fontsize=8)
ax.text(xs[4] + 0.8, 1.85, eqs[4], ha="center", fontsize=8)
ax.text(xs[5] + 0.8, 1.2, eqs[5], ha="center", fontsize=8)
ax.text(xs[6] + 0.8, 1.85, eqs[6], ha="center", fontsize=8)
arrow(ax, xs[7] + 0.8, 3.25, xs[1] + 0.8, 3.25, ls="--", color="#B03A2E", rad=-0.08)
ax.text(7.6, 0.25, "back-propagation: $\\partial\\mathcal{L}/\\partial\\hat{y}\\rightarrow\\partial\\mathcal{L}/\\partial W_d\\rightarrow\\partial\\mathcal{L}/\\partial W^{(l)}$  (Adam update)",
        ha="center", fontsize=8.3, color="#B03A2E")
ax.text(7.5, 5.45, "forward pass (record-level screening: normal vs. COVID-19-positive)", ha="center", fontsize=8.5, color="#1F3B63")
fig.tight_layout(); fig.savefig("fig9_cnn_function.png", dpi=300); plt.close(fig)

# ---------------- Figure 11: stacked ("nested") LSTM ----------------
fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
cols = [1.2, 3.6, 6.0, 9.6]
labels = ["$t-W+1$", "$t-W+2$", "$t-W+3$", "$t$"]
for x, lab in zip(cols, labels):
    box(ax, x, 0.4, 1.6, 0.9, "$x_{" + lab.strip("$") + "}$", fc="#FFF4D6", ec="#8A6D1F")
    box(ax, x, 2.0, 1.6, 0.9, "conv features", fc="#EEF3FA", fs=8)
    box(ax, x, 3.6, 1.6, 1.0, "LSTM-1\n(64 units)", fc="#E8F5E9", ec="#2E6B34", fs=8.5)
    box(ax, x, 5.4, 1.6, 1.0, "LSTM-2\n(32 units)", fc="#E3F2FD", ec="#0D47A1", fs=8.5)
    arrow(ax, x + 0.8, 1.3, x + 0.8, 2.0); arrow(ax, x + 0.8, 2.9, x + 0.8, 3.6)
    arrow(ax, x + 0.8, 4.6, x + 0.8, 5.4)
for a, b in zip(cols[:-1], cols[1:]):
    if b - a < 3:
        arrow(ax, a + 1.6, 4.1, b, 4.1, color="#2E6B34"); arrow(ax, a + 1.6, 5.9, b, 5.9, color="#0D47A1")
ax.text(8.2, 4.05, "· · ·", fontsize=16, ha="center", va="center"); ax.text(8.2, 5.85, "· · ·", fontsize=16, ha="center", va="center")
arrow(ax, 7.6, 4.1, 7.75, 4.1, color="#2E6B34"); arrow(ax, 8.65, 4.1, 9.6, 4.1, color="#2E6B34")
arrow(ax, 7.6, 5.9, 7.75, 5.9, color="#0D47A1"); arrow(ax, 8.65, 5.9, 9.6, 5.9, color="#0D47A1")
box(ax, 12.0, 5.3, 2.7, 1.2, "Dense (3)\n$\\hat{y}_{t+1}$: new cases,\ndeaths, recoveries", fc="#FDECEA", ec="#B03A2E", fs=8)
arrow(ax, 11.2, 5.9, 12.0, 5.9, color="#B03A2E")
ax.text(0.15, 4.1, "$h^{(1)}_\\tau, c^{(1)}_\\tau$", fontsize=8.5, va="center")
ax.text(0.15, 5.9, "$h^{(2)}_\\tau, c^{(2)}_\\tau$", fontsize=8.5, va="center")
ax.text(12.0, 3.3, "dropout 0.2 between\nLSTM-1 and LSTM-2;\nonly $h^{(2)}_t$ reaches\nthe output layer", fontsize=7.8, color="#333333")
ax.text(6.9, 7.45, "W = 14-day input window from the CNN encoder (time runs left to right)", ha="center", fontsize=8.5)
fig.tight_layout(); fig.savefig("fig11_stacked_lstm.png", dpi=300); plt.close(fig)
print("ok")
