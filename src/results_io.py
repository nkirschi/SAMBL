"""
Portable per-seed result I/O and a unified study loader. Decouples downstream
consumers (figures, dashboard, reports) from how experiments were produced.
Each point of a study lives at results/<study>/<point>/ as seed_<n>.npz plus a config.json.
"""

from __future__ import annotations

import os
import glob
import json
from dataclasses import asdict

import numpy as np

from common import (
    SystemConfig,
    CostConfig,
    EstimatorConfig,
    ExcitationConfig,
    ExperimentConfig,
)
from runner import SeedResult, EpisodeRecord

SEED_GLOB = "seed_*.npz"
CONFIG_NAME = "config.json"


# --------------------------------------------------------- SeedResult <-> npz
def seed_to_npz(res: SeedResult, path: str) -> None:
    """Serialise one seed to npz. Scalar diagnostics (cost, errors, F1, ...) become
    one (M,) array per key; matrix-valued diagnostics stored at checkpoint episodes
    (A_est, B_est) become a stacked (k, ...) array plus their episode indices."""
    arrays = {
        "seed": np.int64(res.seed),
        "A_star": np.asarray(res.A_star, dtype=np.float64),
        "B_star": np.asarray(res.B_star, dtype=np.float64),
        "btqb_min_eig": np.float64(res.btqb_min_eig),
        "btqb_max_eig": np.float64(res.btqb_max_eig),
        "agents": np.array(list(res.episodes.keys())),
        "supports": np.array(
            [sorted(int(j) for j in s) for s in (res.supports or [])], dtype=object
        ),
    }
    for ag, eps in res.episodes.items():
        arrays[f"cost::{ag}"] = np.array([e.cost for e in eps], dtype=np.float64)
        keys = set().union(*(e.diagnostics.keys() for e in eps)) if eps else set()
        for k in keys:
            present = [
                (m, e.diagnostics[k]) for m, e in enumerate(eps) if k in e.diagnostics
            ]
            if present and np.isscalar(present[0][1]):
                arrays[f"sdiag::{ag}::{k}"] = np.array(
                    [e.diagnostics.get(k, np.nan) for e in eps], dtype=np.float64
                )
            elif present:
                arrays[f"adiag::{ag}::{k}::eps"] = np.array([m for m, _ in present])
                arrays[f"adiag::{ag}::{k}::vals"] = np.stack(
                    [np.asarray(v, dtype=np.float64) for _, v in present]
                )
    np.savez_compressed(path, **arrays)


def seed_from_npz(path: str) -> SeedResult:
    z = np.load(path, allow_pickle=True)
    res = SeedResult(
        seed=int(z["seed"]),
        A_star=z["A_star"],
        B_star=z["B_star"],
        supports=[set(int(j) for j in s) for s in z["supports"]],
        btqb_min_eig=float(z["btqb_min_eig"]),
        btqb_max_eig=float(z["btqb_max_eig"]),
    )
    for ag in (str(a) for a in z["agents"]):
        eps = [EpisodeRecord(cost=float(c), diagnostics={}) for c in z[f"cost::{ag}"]]
        for f in z.files:
            if f.startswith(f"sdiag::{ag}::"):
                k = f.split("::", 2)[2]
                for m, v in enumerate(z[f]):
                    eps[m].diagnostics[k] = float(v)
            elif f.startswith(f"adiag::{ag}::") and f.endswith("::eps"):
                k = f[len(f"adiag::{ag}::") : -len("::eps")]
                vals = z[f"adiag::{ag}::{k}::vals"]
                for i, m in enumerate(z[f]):
                    eps[int(m)].diagnostics[k] = vals[i]
        res.episodes[ag] = eps
    return res


# --------------------------------------------------- ExperimentConfig <-> json
_NESTED = {
    "system": SystemConfig,
    "cost": CostConfig,
    "estimators": EstimatorConfig,
    "excitation": ExcitationConfig,
}


def config_to_json(cfg: ExperimentConfig, path: str) -> None:
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def config_from_json(path: str) -> ExperimentConfig:
    with open(path) as f:
        d = json.load(f)
    kw = {k: v for k, v in d.items() if k not in _NESTED}
    kw["agents"] = tuple(kw["agents"])
    for name, cls in _NESTED.items():
        kw[name] = cls(**d[name])
    return ExperimentConfig(**kw)


# ------------------------------------------------------------------- loaders
def persist_point(results, cfg, out_dir: str) -> None:
    """Write one point as per-seed npz files + config.json."""
    os.makedirs(out_dir, exist_ok=True)
    config_to_json(cfg, os.path.join(out_dir, CONFIG_NAME))
    for res in results:
        seed_to_npz(res, os.path.join(out_dir, f"seed_{res.seed}.npz"))


def _seed_num(path: str) -> int:
    return int(os.path.basename(path)[5:-4])  # seed_<n>.npz


def load_point(out_dir: str):
    """Load one point -> (results sorted by seed, config) from the per-seed npz layout."""
    seed_files = sorted(glob.glob(os.path.join(out_dir, SEED_GLOB)), key=_seed_num)
    if not seed_files:
        raise FileNotFoundError(f"No {SEED_GLOB} files in {out_dir}")
    results = sorted((seed_from_npz(p) for p in seed_files), key=lambda r: r.seed)
    return results, config_from_json(os.path.join(out_dir, CONFIG_NAME))


def load_study(study_dir: str) -> dict:
    """Load every point under a study dir -> {point_name: (results, config)}."""
    out = {}
    for entry in sorted(os.listdir(study_dir)):
        d = os.path.join(study_dir, entry)
        if glob.glob(os.path.join(d, SEED_GLOB)):
            out[entry] = load_point(d)
    return out
