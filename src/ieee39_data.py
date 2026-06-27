"""
Hard-coded reference data for the IEEE 39-bus (New England) test system.
Used by sample_ieee39 in system_generator.py

IEEE39_BRANCHES: the network topology and per-unit reactances are taken from
  MATPOWER's ``case39.m`` (Zimmerman & Murillo-Sanchez, MATPOWER, ``most/data/case39.m``;
  upstream data: G. W. Bills et al., "On-Line Stability Analysis Study," RP90-1, EPRI,
  Oct. 1970). Each entry is ``(F_BUS, T_BUS, BR_X)`` -- i.e. columns 1, 2 and 4 (BR_X,
  the series reactance) of the ``mpc.branch`` matrix; the column layout is defined in the
  MATPOWER manual / ``idx_brch.m`` (F_BUS=1, T_BUS=2, BR_R=3, BR_X=4, BR_B=5, ...). We
  keep only in-service branches with nonzero reactance (46 lines + transformers) and drop
  resistance/charging, since the swing-network Laplacian uses susceptance b = 1/x only.

IEEE39_GEN_H: generator inertia constants H [s] on a 100 MVA base. These are
  *dynamic* data and are NOT part of MATPOWER's static ``case39.m`` (which carries no
  machine dynamics). They are the standard 10-machine New England values from
  Athay, Podmore & Virmani, "A Practical Method for the Direct Analysis of Transient
  Stability," IEEE Trans. PAS-98(2), 1979, as reproduced in M. A. Pai, "Energy Function
  Analysis for Power System Stability," Kluwer, 1989 (and the Power System Toolbox,
  ``data39m.m``). Bus 39 is the large equivalent machine aggregating the rest of the
  interconnection (H = 500), a near-infinite bus that anchors the angle reference.

IEEE39_ACTUATED_BUSES: the nine actuated generators in the test system.
"""
from __future__ import annotations

from typing import Dict, Tuple

IEEE39_N_BUS: int = 39

# (from_bus, to_bus, series reactance x [pu])
# MATPOWER case39.m mpc.branch cols 1, 2, 4.
IEEE39_BRANCHES: Tuple[Tuple[int, int, float], ...] = (
    (1, 2, 0.0411), (1, 39, 0.025), (2, 3, 0.0151), (2, 25, 0.0086), (2, 30, 0.0181), (3, 4, 0.0213),
    (3, 18, 0.0133), (4, 5, 0.0128), (4, 14, 0.0129), (5, 6, 0.0026), (5, 8, 0.0112), (6, 7, 0.0092),
    (6, 11, 0.0082), (6, 31, 0.025), (7, 8, 0.0046), (8, 9, 0.0363), (9, 39, 0.025), (10, 11, 0.0043),
    (10, 13, 0.0043), (10, 32, 0.02), (12, 11, 0.0435), (12, 13, 0.0435), (13, 14, 0.0101), (14, 15, 0.0217),
    (15, 16, 0.0094), (16, 17, 0.0089), (16, 19, 0.0195), (16, 21, 0.0135), (16, 24, 0.0059), (17, 18, 0.0082),
    (17, 27, 0.0173), (19, 20, 0.0138), (19, 33, 0.0142), (20, 34, 0.018), (21, 22, 0.014), (22, 23, 0.0096),
    (22, 35, 0.0143), (23, 24, 0.035), (23, 36, 0.0272), (25, 26, 0.0323), (25, 37, 0.0232), (26, 27, 0.0147),
    (26, 28, 0.0474), (26, 29, 0.0625), (28, 29, 0.0151), (29, 38, 0.0156),
)

# Generator inertia constants H [s], 100 MVA base (Athay et al. 1979, Pai 1989)
# Bus 39 is the heavy equivalent machine anchoring the reference.
IEEE39_GEN_H: Dict[int, float] = {
    30: 42.0, 31: 30.3, 32: 35.8, 33: 28.6, 34: 26.0,
    35: 34.8, 36: 26.4, 37: 24.3, 38: 34.5, 39: 500.0,
}

# The nine actuated generators (the reference machine on bus 39 is left out).
IEEE39_ACTUATED_BUSES: Tuple[int, ...] = (30, 31, 32, 33, 34, 35, 36, 37, 38)
