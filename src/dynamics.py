from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class ContinuousLQREnv:
    """Stateless continuous-time linear SDE discretised with Euler-Maruyama."""
    A: NDArray[np.float64]
    B: NDArray[np.float64]
    sigma: float
    dt: float

    def step(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        noise: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Propagate one discretised step with supplied standard Gaussian noise."""
        drift = self.A @ x + self.B @ u
        diffusion = self.sigma * noise
        return x + self.dt * drift + np.sqrt(self.dt) * diffusion