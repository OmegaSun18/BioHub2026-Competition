"""v0 tracker: per-track constant-velocity Kalman filter + one optimal assignment per frame pair that decides continuation, division, birth and death TOGETHER.

Per frame pair (t -> t+1), rows = [slot A of every active track, slot B of every active track, one 'birth' row per detection],
columns = [detections, 'death' dummy per A slot, 'no-division' dummy per B slot].
  A_i -> detection j : squared Mahalanobis distance of j to the track's predicted position (gated)
  A_i -> death       : c_death
  B_i -> detection j : Mahalanobis distance with the variance widened by sigma_div (a daughter sits beside the predicted parent position) + c_div
  B_i -> no-division : 0
  birth row j -> j   : c_birth
Slot B alone (B used, A on death) is turned into a plain continuation afterwards. All distances are in micrometres.
Repeated frames: r[t] = number of real (non-repeated) frames up to t. The time step between frame t and t+1 is dt = r[t+1] - r[t] (0 or 1).
With dt = 0 the prediction does not move and the (identical) observation is NOT used to update the filter, so a repeated image is not counted twice.
Appearance terms (intensity, DoG, erosion) are not in v0: they would add a term to the cost matrix here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6


@dataclass
class LinkConfig:
    sigma_m: tuple = (1.2, 1.0, 1.0)      # detection position noise (um) per axis z,y,x  (guess, untuned)
    sigma_a: float = 0.5                  # process noise: acceleration (um / frame^2)      (guess, untuned)
    sigma_v0: float = 2.0                 # prior speed of a new track (um / frame)         (guess, untuned)
    sigma_div: float = 4.0                # extra spread of a daughter around the predicted parent position (um)
    gate: float = 25.0                    # max squared Mahalanobis distance for a link
    gate_div: float = 25.0
    c_birth: float = 12.0
    c_death: float = 12.0
    c_div: float = 11.0
    allow_division: bool = True
    w_app: float = 1.0                    # weight of the appearance term (only used when features are given)
    app_sigma: float = 0.5                # a change of this much in log feature value counts as 1 sigma
    app_cap: float = 9.0                  # cap per feature on the squared normalised change (one odd detection cannot veto a link)


def _predict(pos, vel, Ppp, Ppv, Pvv, dt, cfg):
    q = cfg.sigma_a ** 2
    p = pos + vel * dt
    pp = Ppp + 2 * dt * Ppv + dt * dt * Pvv + q * dt ** 4 / 4
    pv = Ppv + dt * Pvv + q * dt ** 3 / 2
    vv = Pvv + q * dt ** 2
    return p, pp, pv, vv


def _new_state(obs, cfg):
    n = len(obs)
    R = np.array(cfg.sigma_m) ** 2
    return dict(pos=obs.copy(), vel=np.zeros((n, 3)), Ppp=np.tile(R, (n, 1)), Ppv=np.zeros((n, 3)), Pvv=np.full((n, 3), cfg.sigma_v0 ** 2))


def link(frames: list, real_idx: list, cfg: LinkConfig, feats: list | None = None):
    """frames[t] = (N_t, 3) positions in um (may be empty). real_idx[t] = number of real frames up to and including t.
    feats[t] = (N_t, F) log feature values (intensity / DoG / erosion ...) or None: adds  w_app * sum_f min((f_j - f_i)^2 / app_sigma^2, app_cap)  to every link cost
    (both continuation and daughter slots); the gate stays purely spatial.
    -> edges [(t, i, j)]: detection i of frame t -> detection j of frame t+1."""
    R = np.array(cfg.sigma_m) ** 2
    edges = []
    T = len(frames)
    node = np.arange(len(frames[0]))
    st = _new_state(np.asarray(frames[0], float).reshape(-1, 3), cfg)
    if feats is not None:
        st["f"] = np.asarray(feats[0], float).reshape(len(node), -1)
    for t in range(T - 1):
        D = np.asarray(frames[t + 1], float).reshape(-1, 3)
        m, n = len(D), len(node)
        dt = float(real_idx[t + 1] - real_idx[t])
        if n == 0 or m == 0:
            node, st = np.arange(m), _new_state(D, cfg)
            if feats is not None:
                st["f"] = np.asarray(feats[t + 1], float).reshape(m, -1)
            continue
        p, pp, pv, vv = _predict(st["pos"], st["vel"], st["Ppp"], st["Ppv"], st["Pvv"], dt, cfg)
        S = pp + R                                                              # (n, 3) innovation variance
        diff = D[None, :, :] - p[:, None, :]                                    # (n, m, 3)
        chi = (diff ** 2 / S[:, None, :]).sum(-1)
        cost = np.full((2 * n + m, m + 2 * n), BIG)
        app = 0.0
        if feats is not None:
            Df = np.asarray(feats[t + 1], float).reshape(m, -1)
            app = cfg.w_app * np.minimum((Df[None, :, :] - st["f"][:, None, :]) ** 2 / cfg.app_sigma ** 2, cfg.app_cap).sum(-1)
        a = np.where(chi <= cfg.gate, chi + app, BIG)
        cost[:n, :m] = a
        cost[np.arange(n), m + np.arange(n)] = cfg.c_death
        if cfg.allow_division:
            chiB = (diff ** 2 / (S + cfg.sigma_div ** 2)[:, None, :]).sum(-1)
            cost[n:2 * n, :m] = np.where(chiB <= cfg.gate_div, chiB + app + cfg.c_div, BIG)
            cost[n + np.arange(n), m + n + np.arange(n)] = 0.0
        cost[2 * n + np.arange(m), np.arange(m)] = cfg.c_birth
        cost[2 * n:, m:] = 0.0
        r, c = linear_sum_assignment(cost)
        col = np.full(2 * n + m, -1)
        col[r] = c
        A = np.where(col[:n] < m, col[:n], -1)
        B = np.where(col[n:2 * n] < m, col[n:2 * n], -1)
        for i in range(n):                                                      # B alone -> continuation
            if A[i] < 0 and B[i] >= 0:
                A[i], B[i] = B[i], -1
        child_of, fresh = {}, set()
        for i in range(n):
            if A[i] >= 0:
                edges.append((t, int(node[i]), int(A[i])))
                child_of[int(A[i])] = i
            if B[i] >= 0:
                edges.append((t, int(node[i]), int(B[i])))
                child_of[int(B[i])] = i
                fresh.add(int(A[i])); fresh.add(int(B[i]))                      # both daughters start new filters
        ns = _new_state(D, cfg)
        if feats is not None:
            ns["f"] = Df
        for j, i in child_of.items():
            if j in fresh:
                continue
            if dt == 0:                                                         # repeated image: keep the prediction, no update
                ns["pos"][j], ns["vel"][j] = p[i], st["vel"][i]
                ns["Ppp"][j], ns["Ppv"][j], ns["Pvv"][j] = pp[i], pv[i], vv[i]
                continue
            y = D[j] - p[i]
            Kp, Kv = pp[i] / S[i], pv[i] / S[i]
            ns["pos"][j] = p[i] + Kp * y
            ns["vel"][j] = st["vel"][i] + Kv * y
            ns["Ppp"][j] = pp[i] - Kp * pp[i]
            ns["Ppv"][j] = pv[i] - Kp * pv[i]
            ns["Pvv"][j] = vv[i] - Kv * pv[i]
        node, st = np.arange(m), ns
    return edges
