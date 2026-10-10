"""
Flowchart versions of the MeatGuard II experiment pipeline (minimal text):
an overview of the whole experiment and one detailed flow diagram per stage.

Shapes follow flowchart convention: rounded box = process, diamond = decision, parallelogram = data,
pill = start/end. Colours follow figure_plan.md (Okabe-Ito, one colour per stage).

Usage:
    python -m src.viz.flowcharts --out-dir D:/Research_Work/paperpwork/AI/MeatGuard_II-paper/figures/flowcharts
"""
import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Polygon

STAGE = {"data": "#0072B2", "train": "#E69F00", "deploy": "#009E73", "eval": "#56B4E9", "teacher": "#7F7F7F"}
INK = "#222222"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "mathtext.fontset": "dejavusans"})


class Chart:
    """Tiny flowchart canvas in inches; nodes are placed by centre (x, y)."""

    def __init__(self, width, height):
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width); self.ax.set_ylim(0, height); self.ax.axis("off")
        self.nodes = {}

    def _tint(self, colour, t=0.16):
        return tuple(np.array(to_rgb(colour)) * t + (1 - t))

    def node(self, key, x, y, text, colour, kind="process", w=1.15, h=0.42, fs=7, bold=False):
        ax = self.ax
        if kind == "decision":
            pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
            ax.add_patch(Polygon(pts, closed=True, fc="white", ec=colour, lw=1.0, zorder=2))
        elif kind == "data":
            k = 0.12
            pts = [(x - w / 2 + k, y + h / 2), (x + w / 2 + k, y + h / 2), (x + w / 2 - k, y - h / 2),
                   (x - w / 2 - k, y - h / 2)]
            ax.add_patch(Polygon(pts, closed=True, fc=self._tint(colour, 0.1), ec=colour, lw=1.0, zorder=2))
        else:
            rounding = h / 2 if kind == "terminal" else 0.07
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                        boxstyle=f"round,pad=0,rounding_size={rounding}",
                                        fc=colour if kind == "terminal" else self._tint(colour),
                                        ec=colour, lw=1.0, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.15,
                weight="bold" if bold else "normal", color="white" if kind == "terminal" else INK)
        self.nodes[key] = (x, y, w, h, kind)

    def port(self, key, side):
        x, y, w, h, _ = self.nodes[key]
        return {"n": (x, y + h / 2), "s": (x, y - h / 2), "e": (x + w / 2, y), "w": (x - w / 2, y)}[side]

    def edge(self, a, b, sa="s", sb="n", via=(), label=None, label_at=0, colour=INK, dashed=False, lw=0.9):
        """Polyline a -> b through optional waypoints; label placed near segment `label_at`."""
        start = a if isinstance(a, tuple) else self.port(a, sa)
        end = b if isinstance(b, tuple) else self.port(b, sb)
        pts = [start, *via, end]
        xs, ys = zip(*pts)
        self.ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=colour, lw=lw, zorder=1,
                     ls=(0, (3, 2)) if dashed else "-", solid_capstyle="butt")
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        self.ax.annotate("", xy=(x1, y1), xytext=(x0 + (x1 - x0) * 0.6, y0 + (y1 - y0) * 0.6),
                         arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, mutation_scale=7,
                                         shrinkA=0, shrinkB=0), zorder=4)
        if label:
            (lx0, ly0), (lx1, ly1) = pts[label_at], pts[label_at + 1]
            self.ax.text((lx0 + lx1) / 2 + 0.05, (ly0 + ly1) / 2 + 0.05, label, fontsize=6, color=colour,
                         ha="left", va="bottom", zorder=5)

    def band(self, x0, y0, x1, y1, title, colour):
        """Group title only (no background panel); the extent arguments position the title."""
        self.ax.text(x0 + 0.08, y1 - 0.08, title, ha="left", va="top", fontsize=7.5, weight="bold", color=colour)

    def save(self, path, dpi):
        save_figure(self.fig, path, dpi=dpi)
        plt.close(self.fig)
        print("wrote", path)


def save_figure(fig, path, retries=10, **kwargs):
    """Save via a temporary file and move it into place, retrying while another process (e.g. antivirus
    or a file-sync client scanning the previous PNG) briefly locks the target."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(tmp, facecolor="white", **kwargs)
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.5 * (attempt + 1))
    raise OSError(f"Could not replace {path}; it may be open in another program (temporary copy: {tmp})")


# ----------------------------------------------------------------------------- overview
def overview(path, dpi):
    c = Chart(7.48, 2.45)
    y = 1.62
    c.band(0.08, 0.98, 2.9, 2.38, "Data", STAGE["data"])
    c.band(3.0, 0.98, 4.6, 2.38, "Training", STAGE["train"])
    c.band(4.7, 0.98, 7.40, 2.38, "Compression & deployment", STAGE["deploy"])
    c.band(0.08, 0.07, 7.40, 0.86, "Evaluation", STAGE["eval"])
    steps = [("img", "MeatScan\nimages", "data", "data"), ("aud", "Duplicate\naudit", "data", "process"),
             ("ses", "Session\nsplit", "data", "process"), ("crp", "Meat\ncropping", "data", "process"),
             ("tch", "Teacher\ntraining", "teacher", "process"), ("kd", "Distillation", "train", "process"),
             ("prn", "Pruning", "deploy", "process"), ("ft", "Fine-tuning", "deploy", "process"),
             ("q8", "INT8", "deploy", "process"), ("app", "Android\napp", "deploy", "terminal")]
    xs = [0.5, 1.22, 1.88, 2.54, 3.42, 4.18, 5.0, 5.68, 6.36, 7.0]
    for (key, text, col, kind), x in zip(steps, xs):
        c.node(key, x, y, text, STAGE[col], kind=kind, w=0.56 if key != "img" else 0.6, h=0.46, fs=6.2)
    for a, b in zip([s[0] for s in steps[:-1]], [s[0] for s in steps[1:]]):
        c.edge(a, b, "e", "w")
    c.node("evm", 1.88, 0.42, "Classification\nmetrics", STAGE["eval"], w=1.0, h=0.4, fs=6.6)
    c.node("eff", 4.18, 0.42, "Efficiency &\nlatency", STAGE["eval"], w=1.0, h=0.4, fs=6.6)
    c.node("xai", 6.0, 0.42, "Explanation\nfaithfulness", STAGE["eval"], w=1.0, h=0.4, fs=6.6)
    c.edge("ses", "evm", "s", "n", label="test sessions", colour=STAGE["data"])
    c.edge("kd", "eff", "s", "n", colour=STAGE["train"])
    c.edge("app", "xai", "s", "e", via=[(7.0, 0.42)], colour=STAGE["deploy"])
    c.save(path, dpi)


# ----------------------------------------------------------------------------- stage 1
def stage1(path, dpi):
    c = Chart(7.48, 5.2)
    col = STAGE["data"]
    c.band(0.08, 0.15, 3.62, 5.12, "A  Capture-session split", col)
    c.band(3.86, 0.15, 7.40, 5.12, "B  Meat-region cropping", col)
    # A: session split
    xa = 1.85
    c.node("a_in", xa, 4.62, "Images (10,518)", col, "data", w=1.4, h=0.34)
    c.node("a_h", xa, 4.08, "dHash, 8 orientations", col, w=1.6, h=0.34)
    c.node("a_d", xa, 3.52, "Pairwise distance d", col, w=1.6, h=0.34)
    c.node("a_q1", xa, 2.86, "≤ 10 frames apart\nand d ≤ 12?", col, "decision", w=1.55, h=0.66, fs=6.4)
    c.node("a_q2", 0.75, 2.1, "d ≤ 6?", col, "decision", w=0.9, h=0.5, fs=6.4)
    c.node("a_l", xa, 2.1, "Link", col, w=0.75, h=0.32)
    c.node("a_s", xa, 1.55, "Sessions (1,666)", col, w=1.5, h=0.34)
    c.node("a_q3", xa, 0.98, "Session > ¼ of\nval/test target?", col, "decision", w=1.3, h=0.62, fs=6.0)
    c.node("a_tr", 3.12, 0.98, "Train", col, "terminal", w=0.58, h=0.3, fs=6.6)
    c.node("a_g", 0.6, 0.98, "Stratified\nassignment", col, w=0.8, h=0.42, fs=6.0)
    c.edge("a_in", "a_h"); c.edge("a_h", "a_d"); c.edge("a_d", "a_q1")
    c.edge("a_q1", "a_l", label="yes")
    c.edge("a_q1", "a_q2", "w", "n", via=[(0.75, 2.86)], label="no")
    c.edge("a_q2", "a_l", "e", "w", label="yes")
    c.edge("a_l", "a_s")
    c.edge("a_s", "a_q3")
    c.edge("a_q3", "a_tr", "e", "w", label="yes")
    c.edge("a_q3", "a_g", "w", "e", label="no")
    c.node("a_out", 0.6, 0.38, "train / val / test", col, "terminal", w=1.0, h=0.28, fs=6.0)
    c.edge("a_g", "a_out")
    c.ax.text(0.75, 1.8, "no: not linked", ha="center", va="top", fontsize=5.8, color="#666666")
    # B: cropping
    xb = 5.63
    c.node("b_in", xb, 4.62, "Image", col, "data", w=1.0, h=0.34)
    c.node("b_d", xb, 4.08, "Grounding DINO\n\"raw meat.\"", col, w=1.6, h=0.4, fs=6.6)
    c.node("b_q", xb, 3.42, "Box found?", col, "decision", w=1.25, h=0.56, fs=6.6)
    c.node("b_w", 4.45, 2.78, "Whole-frame\nprompt", col, w=0.95, h=0.42, fs=6.4)
    c.node("b_k", xb, 2.78, "Keep confident\nboxes", col, w=1.2, h=0.42, fs=6.4)
    c.node("b_s", xb, 2.15, "SAM 2.1 → mask", col, w=1.5, h=0.34)
    c.node("b_f", xb, 1.6, "Background → mean", col, w=1.6, h=0.34)
    c.node("b_c", xb, 1.05, "Square crop → 224²", col, "terminal", w=1.7, h=0.34, fs=6.6)
    c.edge("b_in", "b_d"); c.edge("b_d", "b_q")
    c.edge("b_q", "b_k", label="yes")
    c.edge("b_q", "b_w", "w", "n", via=[(4.45, 3.42)], label="no")
    c.edge("b_w", "b_s", "s", "w", via=[(4.45, 2.15)])
    c.edge("b_k", "b_s"); c.edge("b_s", "b_f"); c.edge("b_f", "b_c")
    c.ax.text(5.63, 0.45, "Output: cropped images + session split", ha="center", va="center", fontsize=6.6,
              weight="bold", color=col)
    c.save(path, dpi)


# ----------------------------------------------------------------------------- stage 2
def stage2(path, dpi):
    c = Chart(7.48, 4.7)
    t, s = STAGE["teacher"], STAGE["train"]
    c.band(0.08, 0.1, 2.62, 4.62, "A  Teacher selection", t)
    c.band(2.78, 0.1, 7.40, 4.62, "B  Distillation into MeatGuard-S", s)
    xa = 1.35
    c.node("t_in", xa, 4.02, "Training crops", t, "data", w=1.3, h=0.34)
    c.node("t_c", xa, 3.42, "4 ImageNet-pretrained\ncandidates", t, w=1.85, h=0.42, fs=6.6)
    c.node("t_f", xa, 2.78, "Fine-tune", t, w=1.2, h=0.34)
    c.node("t_q", xa, 2.08, "Best validation\nmacro-F1?", t, "decision", w=1.6, h=0.66, fs=6.4)
    c.node("t_o", xa, 1.3, "Teacher (frozen)", t, "terminal", w=1.5, h=0.34, fs=6.6)
    c.node("t_x", xa, 0.62, "Test: once", t, w=1.2, h=0.32, fs=6.4)
    c.edge("t_in", "t_c"); c.edge("t_c", "t_f"); c.edge("t_f", "t_q")
    c.edge("t_q", "t_o", label="selected"); c.edge("t_o", "t_x", dashed=True)
    # B: distillation loop. Teacher and student face each other; the two losses that need both sit between them.
    xt, xm, xs_ = 3.55, 4.75, 5.95
    c.node("b_in", xm, 4.05, "Batch", s, "data", w=0.9, h=0.34)
    c.node("b_t", xt, 2.85, "Teacher\n(frozen)", t, w=0.95, h=1.0)
    c.node("b_s", xs_, 2.85, "Student", s, w=0.95, h=1.0, bold=True)
    c.node("b_kl", xm, 3.15, "KL (T = 4)", s, w=0.82, h=0.32, fs=6.4)
    c.node("b_al", xm, 2.55, "Alignment", s, w=0.82, h=0.32, fs=6.4)
    c.node("b_lb", 6.95, 3.65, "Labels", s, "data", w=0.7, h=0.3, fs=6.2)
    c.node("b_ce", 6.95, 2.85, "CE", s, w=0.6, h=0.32, fs=6.4)
    c.node("b_l", xm, 1.8, "Total loss", s, w=1.1, h=0.34)
    c.node("b_bp", xm + 1.35, 1.2, "Update student\n+ adaptor", s, w=1.3, h=0.42, fs=6.2)
    c.node("b_q", xt, 1.2, "Val macro-F1\nimproved?", s, "decision", w=1.4, h=0.6, fs=6.0)
    c.node("b_ck", xt, 0.45, "Save checkpoint", s, "terminal", w=1.35, h=0.32, fs=6.2)
    c.edge("b_in", "b_t", "w", "n", via=[(xt, 4.05)])
    c.edge("b_in", "b_s", "e", "n", via=[(xs_, 4.05)])
    hw = 0.475                                         # half width of the tall boxes
    c.edge((xt + hw, 3.15), "b_kl", sb="w", colour=t)
    c.edge((xs_ - hw, 3.15), "b_kl", sb="e")
    c.edge((xt + hw, 2.55), "b_al", sb="w", colour=t)
    c.edge((xs_ - hw, 2.55), "b_al", sb="e")
    c.edge((xs_ + hw, 2.85), "b_ce", sb="w")
    c.edge("b_lb", "b_ce")
    c.ax.text(xt + hw + 0.04, 3.2, "logits", fontsize=5.6, color=t, va="bottom")
    c.ax.text(xt + hw + 0.04, 2.6, "features", fontsize=5.6, color=t, va="bottom")
    c.ax.text(xs_ - hw - 0.04, 3.2, "logits", fontsize=5.6, color=s, va="bottom", ha="right")
    c.ax.text(xs_ - hw - 0.04, 2.6, "Stage 3", fontsize=5.6, color=s, va="bottom", ha="right")
    c.edge("b_al", "b_l")
    c.edge("b_kl", (xm + 0.32, 1.97), "e", via=[(xm + 0.62, 3.15), (xm + 0.62, 2.2), (xm + 0.32, 2.2)])
    c.edge("b_ce", "b_l", "s", "e", via=[(6.95, 1.8)])
    c.edge("b_l", "b_bp", "s", "n", via=[(xm, 1.55), (xm + 1.35, 1.55)])
    c.edge("b_bp", "b_q", "w", "e")
    c.edge("b_q", "b_ck", label="yes")
    c.edge("b_q", "b_in", "w", "n", via=[(2.95, 1.2), (2.95, 4.38), (xm, 4.38)])
    c.ax.text(2.88, 2.8, "next epoch", rotation=90, ha="center", va="center", fontsize=6, color=INK)
    c.save(path, dpi)


# ----------------------------------------------------------------------------- stage 3
def stage3(path, dpi):
    c = Chart(7.48, 2.6)
    col = STAGE["deploy"]
    c.band(0.08, 0.08, 7.40, 2.52, "Compression and deployment", col)
    row1 = [("in", "Trained\nstudent", "data"), ("rk", "Rank channels\n(L1 norm)", "process"),
            ("rm", "Remove 30%\n(dependency graph)", "process"), ("ft", "Fine-tune", "process")]
    for (k, text, kind), x in zip(row1, [0.9, 2.4, 3.95, 5.55]):
        c.node(k, x, 1.85, text, col, kind, w=1.3, h=0.48, fs=6.6)
    c.node("q", 6.75, 1.85, "Recovered?", col, "decision", w=1.05, h=0.56, fs=6.4)
    row2 = [("ox", "ONNX"), ("tf", "TF SavedModel"), ("i8", "INT8 PTQ"), ("tl", "TFLite")]
    for (k, text), x in zip(row2, [6.75, 5.3, 3.95, 2.6]):
        c.node(k, x, 0.72, text, col, w=1.15, h=0.38, fs=6.6)
    c.node("app", 1.15, 0.72, "Android app\n+ latency", col, "terminal", w=1.25, h=0.42, fs=6.4)
    for a, b in zip(["in", "rk", "rm", "ft"], ["rk", "rm", "ft", "q"]):
        c.edge(a, b, "e", "w")
    c.edge("q", "ox", label="yes")
    c.edge("q", "ft", "n", "n", via=[(6.75, 2.33), (5.55, 2.33)], label="no: more epochs", label_at=1)
    for a, b in zip(["ox", "tf", "i8", "tl"], ["tf", "i8", "tl", "app"]):
        c.edge(a, b, "w", "e")
    c.ax.text(3.95, 0.4, "calibration: training split", ha="center", va="top", fontsize=6, color=col)
    c.save(path, dpi)


# ----------------------------------------------------------------------------- stage 4
def stage4(path, dpi):
    c = Chart(7.48, 3.3)
    col = STAGE["eval"]
    c.band(0.08, 0.08, 3.62, 3.22, "A  Classification and efficiency", col)
    c.band(3.86, 0.08, 7.40, 3.22, "B  Explanations", col)
    c.node("in", 1.85, 2.75, "Test sessions", col, "data", w=1.35, h=0.34)
    c.node("m", 1.85, 2.15, "MeatGuard-S variants,\nteacher, baselines", col, w=1.9, h=0.44, fs=6.6)
    c.node("p", 1.85, 1.55, "Predictions", col, w=1.2, h=0.34)
    c.node("met", 0.95, 0.85, "Metrics", col, w=0.95, h=0.34)
    c.node("ci", 0.95, 0.3, "Session\nbootstrap CIs", col, "terminal", w=1.15, h=0.4, fs=6.2)
    c.node("lat", 2.75, 0.85, "Size, MACs,\nlatency", col, w=1.05, h=0.42, fs=6.4)
    c.edge("in", "m"); c.edge("m", "p")
    c.edge("p", "met", "s", "n", via=[(1.85, 1.2), (0.95, 1.2)])
    c.edge("p", "lat", "s", "n", via=[(1.85, 1.2), (2.75, 1.2)])
    c.edge("met", "ci")
    c.node("x", 5.63, 2.75, "Attribution maps", col, w=1.5, h=0.34)
    c.ax.text(5.63, 2.43, "Grad-CAM · SHAP · occlusion", ha="center", va="center", fontsize=6, color="#555555")
    c.node("d", 4.55, 1.6, "Deletion /\ninsertion AUC", col, w=1.1, h=0.42, fs=6.4)
    c.node("ms", 5.63, 1.6, "Meat share", col, w=0.95, h=0.34, fs=6.4)
    c.node("r", 6.72, 1.6, "Randomise\nweights", col, w=1.0, h=0.42, fs=6.4)
    c.node("d2", 4.55, 0.65, "vs random\nordering", col, "terminal", w=1.05, h=0.4, fs=6.2)
    c.node("ms2", 5.63, 0.65, "vs mask\narea", col, "terminal", w=0.95, h=0.4, fs=6.2)
    c.node("r2", 6.72, 0.65, "Spearman ρ", col, "terminal", w=1.0, h=0.34, fs=6.2)
    for k, x in (("d", 4.55), ("ms", 5.63), ("r", 6.72)):
        c.edge("x", k, "s", "n", via=[(5.63, 2.12), (x, 2.12)] if k != "ms" else ())
    c.edge("d", "d2"); c.edge("ms", "ms2"); c.edge("r", "r2")
    c.save(path, dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    overview(out / "flow0_overview.png", args.dpi)
    stage1(out / "flow1_data_protocol.png", args.dpi)
    stage2(out / "flow2_distillation.png", args.dpi)
    stage3(out / "flow3_compression_deployment.png", args.dpi)
    stage4(out / "flow4_evaluation.png", args.dpi)


if __name__ == "__main__":
    main()
