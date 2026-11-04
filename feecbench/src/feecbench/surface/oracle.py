"""Energy-conserving midpoint acoustics and analytic semidiscrete modal control."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from .operators import Derived


def energy(ops, p, u):
    p, u = np.atleast_2d(p), np.atleast_2d(u)
    return 0.5 * (np.einsum("ti,it->t", p, ops.m0 @ p.T) + np.einsum("ti,it->t", u, ops.m1 @ u.T))


class Midpoint:
    def __init__(self, ops):
        self.ops = ops
        self.k = ops.stiffness
        self._cache = {}

    def step(self, p, u, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        op = self.ops
        # Bounded cache: regular steps plus the current off-grid remainder.
        key = float(dt)
        if key not in self._cache:
            if len(self._cache) >= 8:
                self._cache.clear()
            self._cache[key] = sla.factorized((op.m0 + dt**2 / 4 * self.k).tocsc())
        rhs = (op.m0 - dt**2 / 4 * self.k) @ p + dt * (op.d0.T @ (op.m1 @ u))
        pn = self._cache[key](rhs)
        return pn, u - dt / 2 * (op.d0 @ (p + pn))

    def solve(self, p0, u0, times, dt):
        """Query each output from a common regular integration grid, including off-grid queries.

        Remainder steps do not perturb the regular trajectory. This also permits
        residual checks at nearby query times without changing earlier steps.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError("times must be a nonempty finite nonnegative vector")
        if np.any(np.diff(times) <= 0) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("strictly increasing times and positive dt required")
        op = self.ops
        p, u = np.array(p0, dtype=float), np.array(u0, dtype=float)
        if p.shape != (len(op.points),) or u.shape != (len(op.edges),):
            raise ValueError("initial cochain shape mismatch")
        if not np.isfinite(p).all() or not np.isfinite(u).all():
            raise ValueError("initial cochains must be finite")
        ps, us, step = [], [], 0
        for t in times:
            n = int(np.floor(t / dt + 1e-12))
            while step < n:
                p, u = self.step(p, u, dt)
                step += 1
            rem = t - step * dt
            pq, uq = self.step(p, u, rem) if rem > 1e-12 * dt else (p, u)
            ps.append(pq.copy())
            us.append(uq.copy())
        return np.array(ps), np.array(us)


def pressure_modes(ops, count=32):
    n = len(ops.points)
    if not 1 <= count <= n:
        raise ValueError("invalid mode budget")
    if count >= n - 1:
        if n > 2000:
            raise ValueError("full dense spectrum limited to 2000 vertices")
        values, vectors = la.eigh(ops.stiffness.toarray(), ops.m0.toarray())
        values, vectors = values[:count], vectors[:, :count]
    else:
        values, vectors = sla.eigsh(
            ops.stiffness.tocsc(), k=count, M=ops.m0.tocsc(), sigma=-1e-6, v0=np.random.default_rng(0).normal(size=n)
        )
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    gram = vectors.T @ (ops.m0 @ vectors)
    if not np.isfinite(gram).all() or np.max(np.abs(gram - np.eye(count))) > 1e-7:
        raise ValueError("spectral mass orthogonality failed")
    residual = ops.stiffness @ vectors - (ops.m0 @ vectors) * values
    if np.linalg.norm(residual) / max(np.linalg.norm(ops.stiffness @ vectors), 1) > 1e-7:
        raise ValueError("spectral residual failed")
    values[0] = 0.0
    return np.maximum(values, 0), vectors


class ModalControl:
    """Truncated exact propagator for exact-plus-harmonic initial velocity.

    Omitted exact velocity and any out-of-corpus coexact content are discarded
    and measured by omitted_energy. Only the harmonic stationary part is kept.
    """

    def __init__(self, ops, values, vectors, harmonic):
        self.ops, self.values, self.vectors, self.harmonic = ops, values, vectors, harmonic

    def solve(self, p0, u0, times):
        op, lam, phi = self.ops, self.values, self.vectors
        a = phi.T @ (op.m0 @ p0)
        omega = np.sqrt(lam[1:])
        if np.any(omega <= 1e-8):
            raise ValueError("expected exactly one constant mode")
        edge_modes = (op.d0 @ phi[:, 1:]) / omega
        b = edge_modes.T @ (op.m1 @ u0)
        theta = np.asarray(times)[:, None] * omega
        at = a[1:] * np.cos(theta) + b * np.sin(theta)
        bt = b * np.cos(theta) - a[1:] * np.sin(theta)
        h = self.harmonic @ (self.harmonic.T @ (op.m1 @ u0))
        return at @ phi[:, 1:].T + a[0] * phi[:, 0], bt @ edge_modes.T + h

    def omitted_energy(self, p0, u0):
        p, u = self.solve(p0, u0, [0])
        parts = Derived(self.ops).hodge(u0 - u[0], self.harmonic)
        return {
            "pressure": float((p0 - p[0]) @ (self.ops.m0 @ (p0 - p[0]))),
            "exact_velocity": float(parts["exact"] @ (self.ops.m1 @ parts["exact"])),
            "coexact_velocity": float(parts["coexact"] @ (self.ops.m1 @ parts["coexact"])),
        }
