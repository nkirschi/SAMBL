"""
Publication figures, separate from the dev dashboard in dashboard.py.

Each figure is a focused, vector (PDF), median+IQR, colourblind-safe (Okabe-Ito)
plot with log scales where appropriate and a caption-ready layout. Single-config
figures take one (results, config) tuple, while scaling figures take a
{d: (results, config)} sweep.

CLI:  uv run python src/figures.py --focal-d 20
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_io import load_study, load_point  # unified loaders

# Real LaTeX so the figure maths matches the thesis (Computer Modern + the same
# math packages the document loads).
matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
    "font.family": "serif",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight",
})

OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
# agent -> (colour, linestyle, marker, label)
STYLE = {
    "oracle":         (OK["black"],      ":",  None, "Oracle"),
    "dense_greedy":   (OK["blue"],       "-",  "o",  "Dense-Greedy"),
    "dense_excited":  (OK["sky"],        "--", "s",  "Dense-Excited"),
    "sparse_greedy":  (OK["vermillion"], "-",  "o",  "Sparse-Greedy"),
    "sparse_excited": (OK["orange"],     "--", "^",  "Sparse-Excited"),
}
LEARNING = ["dense_greedy", "dense_excited", "sparse_greedy", "sparse_excited"]


def _med_iqr(a, axis=0):
    return (np.nanmedian(a, axis), np.nanpercentile(a, 25, axis), np.nanpercentile(a, 75, axis))


def _final_regret(res, agent):
    return np.array([r.cumulative_regret(agent)[-1] for r in res])


def _traj(res, agent, key):
    return np.array([r.diagnostic_trajectory(agent, key) for r in res], dtype=float)


def _per_ep_regret(res, agent):
    a = np.array([[ep.cost for ep in r.episodes[agent]] for r in res])
    o = np.array([[ep.cost for ep in r.episodes["oracle"]] for r in res])
    return a - o


# ----------------------------------------------------------------- A. scaling
def fig_scaling_regret(sweep, outdir, fname="A_scaling_regret.pdf", suptitle=None):
    ds = sorted(sweep)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for agent in LEARNING:
        med, lo, hi = [], [], []
        for d in ds:
            fr = _final_regret(sweep[d][0], agent)
            m, l, h = np.median(fr), np.percentile(fr, 25), np.percentile(fr, 75)
            med.append(m); lo.append(l); hi.append(h)
        c, ls, mk, lab = STYLE[agent]
        ax1.plot(ds, med, color=c, ls=ls, marker=mk, ms=5, label=lab)
        ax1.fill_between(ds, lo, hi, color=c, alpha=0.15)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("dimension $d$"); ax1.set_ylabel("final cumulative regret $R_M$")
    ax1.set_title("(a) Regret scaling"); ax1.set_xticks(ds); ax1.set_xticklabels(ds)
    ax1.legend()

    ratio = [np.median(_final_regret(sweep[d][0], "dense_greedy"))
             / np.median(_final_regret(sweep[d][0], "sparse_greedy")) for d in ds]
    ax2.plot(ds, ratio, color=OK["vermillion"], marker="o", ms=5,
             label="empirical (median regret)")
    ax2.axhline(1.0, color="grey", ls=":", lw=1)
    ax2.set_xscale("log"); ax2.set_xlabel("dimension $d$")
    ax2.set_ylabel(r"dense / sparse regret"); ax2.set_xticks(ds); ax2.set_xticklabels(ds)
    ax2.set_title("(b) Sparse advantage vs $d$"); ax2.legend()
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, fname))
    plt.close(fig)


# ----------------------------------------------------- B. regret decomposition
def fig_regret_decomposition(res, cfg, outdir):
    M = cfg.max_episodes
    m = np.arange(1, M + 1)

    def smooth(y, w=5):  # centred rolling median to expose the rate above the noise
        return np.array([np.median(y[max(0, i - w // 2): i + w // 2 + 1])
                         for i in range(len(y))])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    sparse_sm = None
    for agent in ("dense_greedy", "sparse_greedy"):
        sm = smooth(np.median(_per_ep_regret(res, agent), axis=0))
        if agent == "sparse_greedy":
            sparse_sm = sm
        c, ls, mk, lab = STYLE[agent]
        pos = sm > 0
        ax.plot(m[pos], sm[pos], color=c, ls=ls, lw=1.6, label=lab)
    # reference slopes anchored to overlay the sparse curve's two regimes
    for exp_, anc, lab, st in [(0.5, 5, r"$\propto m^{-1/2}$ (transient)", dict(ls="--", lw=1)),
                               (1.0, M // 2, r"$\propto m^{-1}$ (tail)", dict(ls=":", lw=1.2))]:
        v = sparse_sm[anc - 1]
        ax.plot(m, v * (anc / m) ** exp_, color="grey", label=lab, **st)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(bottom=max(1e-2, np.nanmin(sparse_sm[sparse_sm > 0]) * 0.5))
    ax.set_xlabel("episode $m$"); ax.set_ylabel(r"per-episode regret $r_m$ (median)")
    ax.set_title(f"Regret decomposition ($d={cfg.system.d}$, $p={cfg.system.p}$, "
                 f"$s={cfg.system.sparsity}$)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "B_regret_decomposition.pdf"))
    plt.close(fig)


# ----------------------------------------------------------- C. speedup vs d
def fig_speedup_vs_d(sweep, outdir):
    from metrics import basin_entry_ratio
    ds = sorted(sweep)
    emp_med, emp_lo, emp_hi, theo = [], [], [], []
    for d in ds:
        res, cfg = sweep[d]
        # threshold = worst greedy final joint error (same rule as the dashboard)
        finals = [np.nanmedian(_traj(res, a, "error_joint")[:, -1])
                  for a in ("dense_greedy", "sparse_greedy")]
        thr = float(max(finals))
        ratios, med, _ = basin_entry_ratio(res, "dense_greedy", "sparse_greedy", threshold=thr)
        r = ratios[np.isfinite(ratios)]
        emp_med.append(med)
        emp_lo.append(np.percentile(r, 25) if len(r) else np.nan)
        emp_hi.append(np.percentile(r, 75) if len(r) else np.nan)
        theo.append(cfg.theoretical_speedup)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(ds, emp_med, color=OK["vermillion"], marker="o", ms=5, label="empirical median")
    ax.fill_between(ds, emp_lo, emp_hi, color=OK["vermillion"], alpha=0.15, label="IQR")
    ax.plot(ds, theo, color=OK["black"], ls="--", marker="x",
            label=r"theory $(d{+}p)/(s\log(d{+}p))$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dimension $d$"); ax.set_ylabel(r"basin-entry speedup $m_0^{d}/m_0^{s}$")
    ax.set_xticks(ds); ax.set_xticklabels(ds)
    ax.set_title("(c) Basin-entry speedup vs theory"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "C_speedup_vs_d.pdf"))
    plt.close(fig)


# ------------------------------------------------------------ D. A/B asymmetry
def fig_ab_asymmetry(res, cfg, outdir):
    M = cfg.max_episodes
    m = np.arange(1, M + 1)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for key, col, lab in [("error_A", OK["green"], r"drift $\|\hat A-A_\star\|_F/\|A_\star\|_F$"),
                          ("error_B", OK["vermillion"], r"control $\|\hat B-B_\star\|_F/\|B_\star\|_F$")]:
        med, lo, hi = _med_iqr(_traj(res, "sparse_greedy", key))
        ax.plot(m, med, color=col, lw=1.6, label=lab)
        ax.fill_between(m, lo, hi, color=col, alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("episode $m$"); ax.set_ylabel("relative parameter error")
    ax.set_title(f"(d) Drift vs control identifiability (Sparse-Greedy, $d={cfg.system.d}$)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "D_ab_asymmetry.pdf"))
    plt.close(fig)


# ------------------------------------------------------ E. excitation restores RE
def fig_excitation_re(res, cfg, outdir):
    M = cfg.max_episodes
    m = np.arange(1, M + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    # distinct colours (the global greedy/excited pair is too close for a 2-line plot)
    local = {"sparse_greedy": (OK["vermillion"], "-"), "sparse_excited": (OK["blue"], "--")}
    for agent in ("sparse_greedy", "sparse_excited"):
        c, ls = local[agent]
        lab = STYLE[agent][3]
        mb, lob, hib = _med_iqr(_traj(res, agent, "error_B"))
        ax1.plot(m, mb, color=c, ls=ls, lw=1.6, label=lab); ax1.fill_between(m, lob, hib, color=c, alpha=0.12)
        mg = np.nanmedian(_traj(res, agent, "gram_min_eig"), axis=0)
        ax2.plot(m, mg, color=c, ls=ls, lw=1.6, label=lab)
    ax1.set_yscale("log"); ax1.set_xlabel("episode $m$"); ax1.set_ylabel(r"control error $\|\hat B-B_\star\|_F/\|B_\star\|_F$")
    ax1.set_title("(e1) Excitation lowers control error"); ax1.legend()
    ax2.set_xlabel("episode $m$"); ax2.set_ylabel(r"restricted Gram min-eig (RE proxy)")
    ax2.set_title("(e2) Excitation restores curvature"); ax2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "E_excitation_re.pdf"))
    plt.close(fig)


# ----------------------------------------------------------- F. cond(B^T Q B)
def fig_btqb_cond(sweep, outdir):
    # The artifact that motivates equal-authority scaling: raw random-sparse control
    # matrices become ill-conditioned as p grows. Sampled fresh un-normalised (the
    # systems actually used in the experiments are normalised to kappa ~ 1).
    from system_generator import sample_synthetic_system
    ds = sorted(sweep)
    med, lo, hi = [], [], []
    for d in ds:
        cfg = sweep[d][1].system
        conds = []
        for seed in range(25):
            _, B, _, _ = sample_synthetic_system(d, cfg.p, cfg.s_A, cfg.s_B, seed,
                                              0.1, 1.9, 0.1, 1.9, normalise_B=False)
            conds.append(np.linalg.cond(B.T @ B))
        conds = np.array(conds)
        med.append(np.median(conds)); lo.append(np.percentile(conds, 25)); hi.append(np.percentile(conds, 75))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(ds, med, color=OK["purple"], marker="o", ms=5, label="median")
    ax.fill_between(ds, lo, hi, color=OK["purple"], alpha=0.15, label="IQR")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dimension $d$"); ax.set_ylabel(r"$\kappa(B_\star^\top Q\, B_\star)$")
    ax.set_xticks(ds); ax.set_xticklabels(ds)
    ax.set_title("(f) Raw control conditioning vs dimension"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "F_btqb_condition.pdf"))
    plt.close(fig)


# ------------------------------------------- G. regularisation-strength sweep
def fig_hyperparam(study_dir, outdir, chosen=0.02):
    """Sparse-Greedy final regret vs the LASSO constant c_lambda, one curve per d.
    Reads a c_lambda x d study (results/clambda); each point contributes its own
    config's c_lambda and d, so the grid is taken straight from the run."""
    study = load_study(study_dir)
    if not study:
        return
    by_d = {}
    for res, cfg in study.values():
        by_d.setdefault(cfg.system.d, []).append(
            (cfg.estimators.c_lambda, float(np.median(_final_regret(res, "sparse_greedy"))))
        )
    fig, ax = plt.subplots(figsize=(6, 4.2))
    palette = (OK["green"], OK["blue"], OK["vermillion"], OK["purple"], OK["sky"], OK["orange"])
    for d, col in zip(sorted(by_d), palette):
        pts = sorted(by_d[d])
        ax.plot([cl for cl, _ in pts], [r for _, r in pts],
                color=col, marker="o", ms=4, label=f"$d={d}$")
    ax.axvline(chosen, color="grey", ls=":", lw=1, label=fr"chosen $c_\lambda={chosen}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"regularisation constant $c_\lambda$")
    ax.set_ylabel("Sparse-Greedy regret $R_M$")
    ax.set_title(r"Regularisation strength $c_\lambda$"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "G_hyperparameter.pdf"))
    plt.close(fig)


# --------------------------------------------- single-system anchor (regret + A/B)
def fig_anchor(res, cfg, outdir, fname="I_anchor.pdf", suptitle=None):
    M = cfg.max_episodes
    m = np.arange(1, M + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) cumulative regret vs episode, median + IQR
    for agent in LEARNING:
        med, lo, hi = _med_iqr(np.cumsum(_per_ep_regret(res, agent), axis=1))
        c, ls, _, lab = STYLE[agent]
        ax1.plot(m, med, color=c, ls=ls, lw=1.6, label=lab)
        ax1.fill_between(m, lo, hi, color=c, alpha=0.12)
    ax1.set_xlabel("episode $m$"); ax1.set_ylabel(r"cumulative regret $R_m$")
    ax1.set_title("(a) Regret"); ax1.legend()

    # (b) drift vs control recovery: dense (solid) vs sparse (dashed), A green / B vermillion
    for agent, ls in (("dense_greedy", "-"), ("sparse_greedy", "--")):
        who = STYLE[agent][3].split("-")[0]
        for key, col, blk in (("error_A", OK["green"], "A"), ("error_B", OK["vermillion"], "B")):
            med = np.nanmedian(_traj(res, agent, key), axis=0)
            ax2.plot(m, med, color=col, ls=ls, lw=1.6, label=f"{who} err$_{blk}$")
    ax2.set_yscale("log")
    ax2.set_xlabel("episode $m$"); ax2.set_ylabel("relative parameter error")
    ax2.set_title("(b) Drift (A) vs control (B) recovery")
    ax2.legend(ncol=2, fontsize=8)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, fname))
    plt.close(fig)


# ------------------------------------------------ J. sparsity sweep (fixed d, vary s)
def fig_sparsity(sweep, outdir):
    ss = sorted(sweep)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) regret vs s for the four learning agents
    for agent in LEARNING:
        med, lo, hi = [], [], []
        for s in ss:
            fr = _final_regret(sweep[s][0], agent)
            med.append(np.median(fr)); lo.append(np.percentile(fr, 25)); hi.append(np.percentile(fr, 75))
        c, ls, mk, lab = STYLE[agent]
        ax1.plot(ss, med, color=c, ls=ls, marker=mk, ms=5, label=lab)
        ax1.fill_between(ss, lo, hi, color=c, alpha=0.15)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("row sparsity $s$"); ax1.set_ylabel("final cumulative regret $R_M$")
    ax1.set_title("(a) Regret vs sparsity"); ax1.set_xticks(ss); ax1.set_xticklabels(ss)
    ax1.legend()

    # (b) advantage vs s, with a 1/(s log(d+p)) shape reference anchored to the excess
    adv = [np.median(_final_regret(sweep[s][0], "dense_greedy"))
           / np.median(_final_regret(sweep[s][0], "sparse_greedy")) for s in ss]
    ax2.plot(ss, adv, color=OK["vermillion"], marker="o", ms=5, label="empirical (median regret)")
    cfg0 = sweep[ss[0]][1]
    dp = cfg0.system.d + cfg0.system.p
    raw = np.array([1.0 / (s * np.log(dp)) for s in ss])
    ref = 1.0 + (adv[0] - 1.0) * raw / raw[0]          # excess decays as 1/(s log(d+p))
    ax2.plot(ss, ref, color="grey", ls="--", lw=1, label=r"$1 + c/(s\log(d+p))$")
    ax2.axhline(1.0, color="grey", ls=":", lw=1)
    ax2.set_xscale("log"); ax2.set_xlabel("row sparsity $s$")
    ax2.set_ylabel(r"dense / sparse regret"); ax2.set_xticks(ss); ax2.set_xticklabels(ss)
    ax2.set_title("(b) Sparse advantage vs $s$"); ax2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "J_sparsity.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal-d", type=int, default=20)
    ap.add_argument("--synthetic-dir", default="results/synthetic")
    ap.add_argument("--spring-dir", default="results/springs")
    ap.add_argument("--ieee39-dir", default="results/ieee39")
    ap.add_argument("--sparsity-dir", default="results/sparsity")
    ap.add_argument("--clambda-dir", default="results/clambda")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def by_int(study_dir):
        """Load a study dir into {int(point): (results, cfg)} (e.g. d10->10, s2->2)."""
        if not os.path.isdir(study_dir):
            return {}
        return {int("".join(c for c in p if c.isdigit())): rc
                for p, rc in load_study(study_dir).items()}

    sweep = by_int(args.synthetic_dir)
    focal = sweep.get(args.focal_d)

    print(f"sweep d={sorted(sweep)}, focal d={args.focal_d}")
    fig_scaling_regret(sweep, args.out);            print("  [A] scaling regret")
    if focal:
        fig_regret_decomposition(*focal, args.out); print("  [B] regret decomposition")
    fig_speedup_vs_d(sweep, args.out);              print("  [C] speedup vs d")
    if focal:
        fig_ab_asymmetry(*focal, args.out);         print("  [D] A/B asymmetry")
        fig_excitation_re(*focal, args.out);        print("  [E] excitation / RE")
    fig_btqb_cond(sweep, args.out);                 print("  [F] cond(B^T Q B) vs d")
    if os.path.isdir(args.clambda_dir):
        fig_hyperparam(args.clambda_dir, args.out); print("  [G] c_lambda sweep")
    else:
        print("  [G] c_lambda sweep skipped (no results/clambda/)")

    spring = by_int(args.spring_dir)
    if spring:
        fig_scaling_regret(spring, args.out, fname="H_spring_scaling.pdf",
                           suptitle="Spring-mass chain (fixed $s=3$, $p=d/4$)")
        print(f"  [H] spring-chain scaling (d={sorted(spring)})")
    else:
        print("  [H] spring-chain scaling skipped (no results/spring/)")

    if os.path.isdir(args.ieee39_dir) and any(
            f.startswith("seed_") for f in os.listdir(args.ieee39_dir)):
        fig_anchor(*load_point(args.ieee39_dir), args.out,
                   fname="I_ieee39_anchor.pdf",
                   suptitle="IEEE 39-bus power grid ($d=78$, $p=9$)")
        print("  [I] IEEE 39-bus anchor")
    else:
        print("  [I] IEEE 39-bus anchor skipped (no results/ieee39/)")

    sparsity = by_int(args.sparsity_dir)
    if sparsity:
        fig_sparsity(sparsity, args.out)
        print(f"  [J] sparsity sweep (s={sorted(sparsity)})")
    else:
        print("  [J] sparsity sweep skipped (no results/sparsity/)")

    # Emit PNG previews next to the vector PDFs (chat clients render PNG inline;
    # the PDFs remain the thesis artefacts). No-op if poppler isn't installed.
    import subprocess, glob
    n_png = 0
    for pdf in sorted(glob.glob(os.path.join(args.out, "*.pdf"))):
        try:
            subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile",
                            pdf, pdf[:-4]], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            n_png += 1
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    print(f"  ({n_png} PNG previews)" if n_png else "  (no PNG: poppler/pdftoppm absent)")
    print(f"-> {args.out}/")


if __name__ == "__main__":
    main()
