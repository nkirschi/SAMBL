import numpy as np

from system_generator import sample_ieee39, _is_stabilisable
from ieee39_data import (
    IEEE39_BRANCHES,
    IEEE39_ACTUATED_BUSES,
    IEEE39_N_BUS,
)


class TestIEEE39:
    def test_dimensions(self):
        A, B, sup, _ = sample_ieee39(seed=0)
        assert A.shape == (78, 78) and B.shape == (78, 9)
        assert len(IEEE39_BRANCHES) == 46 and IEEE39_N_BUS == 39

    def test_structure_and_sparsity(self):
        n = IEEE39_N_BUS
        A, B, sup, _ = sample_ieee39(seed=0)
        for i in range(n):
            # angle row: theta_dot = omega, a single unit entry
            assert (A[2 * i] != 0).sum() == 1 and A[2 * i, 2 * i + 1] == 1.0
            # frequency row: own damping + (1 + degree) couplings -> between 3 and 7
            assert 3 <= (A[2 * i + 1] != 0).sum() <= 7
        assert (np.count_nonzero(B, axis=0) == 1).all()        # one entry per control
        assert all(r % 2 == 1 for r in np.flatnonzero(B.any(axis=1)))  # in freq rows

    def test_actuates_generators_not_slack(self):
        A, B, sup, _ = sample_ieee39(seed=0)
        actuated_rows = set(np.flatnonzero(B.any(axis=1)).tolist())
        assert actuated_rows == {2 * (b - 1) + 1 for b in IEEE39_ACTUATED_BUSES}
        assert 2 * (39 - 1) + 1 not in actuated_rows           # bus 39 left as reference

    def test_stabilisable_and_marginally_stable(self):
        A, B, _, _ = sample_ieee39(seed=0)
        assert _is_stabilisable(A, B)
        re = np.real(np.linalg.eigvals(A))
        assert (re < 1e-6).all()                # no unstable modes
        assert np.sum(np.abs(re) < 1e-9) == 1   # exactly one marginal (reference) mode

    def test_self_exploration_wellposed(self):
        A, B, _, _ = sample_ieee39(seed=0)
        BtB = B.T @ B
        assert np.linalg.matrix_rank(BtB) == 9
        assert np.linalg.cond(BtB) < 50         # no near-zero column (bus 39 excluded)

    def test_symmetric_coupling_and_supports(self):
        d = 78
        A, B, sup, _ = sample_ieee39(seed=0)
        # angle->angle coupling block is symmetric up to the 1/M_i row scaling sign-pattern
        adj = (A[1::2, 0::2] != 0)
        assert (adj == adj.T).all()             # Laplacian sparsity is symmetric
        for r in range(d):
            nz = set(np.flatnonzero(A[r]).tolist()) | {
                d + c for c in np.flatnonzero(B[r]).tolist()}
            assert nz == sup[r]

    def test_deterministic_without_jitter(self):
        A1, B1, _, _ = sample_ieee39(seed=1)
        A2, B2, _, _ = sample_ieee39(seed=2)   # different seed, no jitter
        assert np.array_equal(A1, A2) and np.array_equal(B1, B2)

    def test_jitter_makes_ensemble(self):
        A1, _, _, _ = sample_ieee39(seed=1, jitter=0.1)
        A2, _, _, _ = sample_ieee39(seed=2, jitter=0.1)
        assert not np.array_equal(A1, A2)              # varies per seed
        assert (A1 != 0).sum() == (A2 != 0).sum()      # same sparsity pattern
