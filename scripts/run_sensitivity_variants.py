"""Formulation and sensitivity variants.

1. Two-dimensional sweep: fines-proxy multiplier vs excavation fines
   share (the compositional effect of screening the spoil stream).
2. Indicator-constraint formulation of the certification logic
   (objective and wall time vs the big-M formulation).
3. Single-grade-per-line restriction (sum of certification binaries
   per line <= 1).
4. Extended market-absorption multipliers (0.25 .. 2.0).

Appends a "sensitivity_variants" block to
results/multistream/benchmark_and_sensitivity.json (deterministic;
idempotent overwrite of that key only).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.multistream_ops import MultiStreamOps, build_ops_instance

OUT = Path(__file__).resolve().parents[1] / "results" / "multistream"


def solve(mm, time_limit=120):
    r = mm.solve(time_limit=time_limit)
    if r is None:
        raise RuntimeError("infeasible")
    return r


res = {}

# ---- 1. two-dimensional sweep (dual regime) ----
grid2d = []
for kp_mult in (1, 2, 4, 6):
    row = []
    for e_exc in (0.60, 0.45, 0.30, 0.15):
        inst = build_ops_instance()
        inst["c_exc"] = [e_exc] * inst["n_sites"]
        if kp_mult > 1:
            for d in inst["demands"]:
                d["kappa2"] = round(d["kappa2"] * kp_mult, 4)
        mm = MultiStreamOps(inst, enforce_spec=True, dual_attr=True)
        r = solve(mm)
        cell = {"kp_mult": kp_mult, "e_exc": e_exc,
                "certified_pct": round(r["certified_util"] * 100, 1),
                "cost_MCNY": round(r["cost_MCNY"], 2)}
        row.append(cell)
        print(f"  kp x{kp_mult} e_exc {e_exc:.2f}: "
              f"cert {cell['certified_pct']}%")
    grid2d.append(row)
res["kappa2_eexc_grid"] = grid2d

# ---- 2. indicator-constraint formulation ----
ind = {}
for label, kw in (("single", {}), ("dual", {"dual_attr": True})):
    inst = build_ops_instance()
    mm = MultiStreamOps(inst, enforce_spec=True, indicator=True, **kw)
    t0 = time.time()
    r = solve(mm)
    wall = time.time() - t0
    ind[label] = {"cost_MCNY": round(r["cost_MCNY"], 4),
                  "certified_pct": round(r["certified_util"] * 100, 2),
                  "wall_s": round(wall, 3)}
    print(f"  indicator {label}: {ind[label]}")
res["indicator_variant"] = ind

# ---- 3. single-grade-per-line restriction ----
inst = build_ops_instance()
mm = MultiStreamOps(inst, enforce_spec=True)
D = inst["n_dem"]
for f in range(inst["n_fac"]):
    for t in range(inst["n_tech"]):
        mm.m.add_constraint(
            mm.m.sum(mm.w[f, t, d] for d in range(D)) <= 1)
r = solve(mm)
res["single_grade_per_line"] = {
    "cost_MCNY": round(r["cost_MCNY"], 2),
    "certified_pct": round(r["certified_util"] * 100, 1),
    "fill_pct": round(r["fill_util"] * 100, 1)}
print(f"  single-grade-per-line: {res['single_grade_per_line']}")

# ---- 4. demand-absorption multipliers ----
dem = {}
for m in (0.25, 0.5, 1.5, 2.0):
    inst = build_ops_instance()
    for d in inst["demands"]:
        d["quantity"] = float(d["quantity"] * m)
    r = solve(MultiStreamOps(inst, enforce_spec=True))
    dem[f"x{m}"] = {"certified_pct": round(r["certified_util"] * 100, 1),
                    "cost_MCNY": round(r["cost_MCNY"], 2)}
    print(f"  demand x{m}: cert {dem[f'x{m}']['certified_pct']}%")
res["demand_grid"] = dem

path = OUT / "benchmark_and_sensitivity.json"
data = json.loads(path.read_text(encoding="utf-8"))
data["sensitivity_variants"] = res
path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("saved sensitivity_variants block ->", path)
