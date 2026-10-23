"""Benchmarks and sensitivity analyses for the multi-stream routing study.

1. Representation benchmark, decomposed:
   (a) district-homogeneous attributes with every stream's outlets kept
       (isolates the value of stream-level quality information);
   (b) single-flow merge into one homogeneous flow at the global
       generation-weighted mean attributes (the scalar-quality model of
       the literature; this also removes stream-specific outlets).
   Both cap regimes are solved for each variant.
2. Paired price x processing-cost grid: blending AND segregation solved
   in every cell, so the blending advantage is measured, not asserted.
3. Fines-composition sensitivity: all stream fines shares scaled by s,
   correct-direction bisection plus a fixed grid curve. Composition
   sensitivity only -- no washing yield, residue, or cost is modelled.

Writes results/multistream/benchmark_and_sensitivity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.multistream_ops import MultiStreamOps, build_ops_instance

OUT = Path(__file__).resolve().parents[1] / "results" / "multistream"


def solve(inst, segregate=False, **kw):
    m = MultiStreamOps(inst, enforce_spec=True, **kw)
    if segregate:
        for s in range(inst["n_sites"]):
            for f in range(inst["n_fac"]):
                for t in range(inst["n_tech"]):
                    m.m.add_constraint(m.a_exc[s, f, t] == 0)
    r = m.solve(time_limit=120)
    if r is None:
        raise RuntimeError("infeasible benchmark solve")
    return r


def summary(r):
    return {"cost_MCNY": round(r["cost_MCNY"], 2),
            "certified_pct": round(r["certified_util"] * 100, 1),
            "fill_pct": round(r["fill_util"] * 100, 1),
            "to_plant_Mt": round(r["to_plant_dem_Mt"] + r["to_plant_exc_Mt"]
                                 + r["to_plant_con_Mt"], 3),
            "plants": r["facilities_active"]}


results = {}

# ---- 1. representation benchmark -------------------------------------
multi_single = solve(build_ops_instance())
multi_dual = solve(build_ops_instance(), dual_attr=True)

agg = build_ops_instance()
Q = np.array(agg["Q_dem"]) + np.array(agg["Q_exc"]) + np.array(agg["Q_con"])
w = [np.array(agg[k]) / Q for k in ("Q_dem", "Q_exc", "Q_con")]
for pre in ("b", "c"):
    site = sum(np.array(agg[f"{pre}_{s}"]) * wi
               for s, wi in zip(("dem", "exc", "con"), w))
    for s in ("dem", "exc", "con"):
        agg[f"{pre}_{s}"] = list(site)
agg_single = solve(agg)
agg_dual = solve(agg, dual_attr=True)

ss = build_ops_instance()
# global generation-weighted mean: weight each district's blended
# attribute by the district's total generation, then divide by the total
site_b = (np.array(ss["b_dem"]) * w[0] + np.array(ss["b_exc"]) * w[1]
          + np.array(ss["b_con"]) * w[2])
site_c = (np.array(ss["c_dem"]) * w[0] + np.array(ss["c_exc"]) * w[1]
          + np.array(ss["c_con"]) * w[2])
b_avg = float(np.sum(site_b * Q) / np.sum(Q))
c_avg = float(np.sum(site_c * Q) / np.sum(Q))
ss["Q_dem"] = list(Q)
ss["Q_exc"] = [0.0] * len(Q)
ss["Q_con"] = [0.0] * len(Q)
ss["b_dem"] = [b_avg] * len(Q)
ss["c_dem"] = [c_avg] * len(Q)
single_single = solve(ss)
single_dual = solve(ss, dual_attr=True)

results["benchmark"] = {
    "multi_single": summary(multi_single),
    "multi_dual": summary(multi_dual),
    "homogeneous_attributes_outlets_kept": {
        "single": summary(agg_single), "dual": summary(agg_dual)},
    "single_flow_no_stream_outlets": {
        "merged_b_avg": round(b_avg, 4), "merged_c_avg": round(c_avg, 4),
        "single": summary(single_single), "dual": summary(single_dual)},
}
print("=== representation benchmark ===")
print("  multi:            ", results["benchmark"]["multi_single"],
      results["benchmark"]["multi_dual"])
print("  homogeneous attrs:", results["benchmark"]
      ["homogeneous_attributes_outlets_kept"])
print("  single flow:      ", results["benchmark"]
      ["single_flow_no_stream_outlets"])

# ---- 2. paired price x cost grid --------------------------------------
grid = []
for ps in (0.8, 1.0, 1.2):
    row = []
    for cs in (0.5, 1.0, 2.0):
        cell = {"price_mult": ps, "cost_mult": cs}
        for name, seg in (("blend", False), ("segregate", True)):
            inst = build_ops_instance()
            for d in inst["demands"]:
                d["price"] = round(d["price"] * ps, 2)
            for t in inst["techs"]:
                t["proc_cost"] = round(t["proc_cost"] * cs, 2)
            cell[name] = summary(solve(inst, segregate=seg))
        cell["cost_gap_MCNY"] = round(
            cell["segregate"]["cost_MCNY"] - cell["blend"]["cost_MCNY"], 2)
        row.append(cell)
        print(f"  price x{ps} cost x{cs}: "
              f"blend {cell['blend']['certified_pct']}% "
              f"{cell['blend']['cost_MCNY']}M | "
              f"segregate {cell['segregate']['certified_pct']}% "
              f"{cell['segregate']['cost_MCNY']}M | "
              f"gap {cell['cost_gap_MCNY']}M")
    grid.append(row)
results["price_cost_grid"] = grid

# ---- 3. fines-composition scaling --------------------------------------


def dual_certified(scale):
    inst = build_ops_instance()
    for k in ("c_dem", "c_exc", "c_con"):
        inst[k] = [c * scale for c in inst[k]]
    r = solve(inst, dual_attr=True)
    return r["certified_util"]


curve = {}
for s in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0):
    curve[f"{s}"] = round(dual_certified(s), 4)
    print(f"  fines scale {s}: certified {curve[f'{s}']:.4f}")

lo, hi = 0.05, 1.0   # certified > 0 at lo, 0 at hi; find the boundary
assert dual_certified(lo) > 0.001 and dual_certified(hi) <= 0.001
for _ in range(14):
    mid = (lo + hi) / 2
    if dual_certified(mid) > 0.001:
        lo = mid      # feasible side lies at or below mid
    else:
        hi = mid      # infeasible side lies at or above mid
results["fines_scaling"] = {
    "definition": "all stream fines shares multiplied by s",
    "curve": curve,
    "s_largest_feasible": round(lo, 3),
    "s_smallest_infeasible": round(hi, 3),
    "note": "composition sensitivity only; no washing yield/cost modelled",
}

# ---- 4. robustness ---------------------------------------------------------------------
rob = {}

# fines-removal differentiation for the sorting line (rho_e sweep)
for re_ in (0.52, 0.70, 0.90, 0.96):
    inst = build_ops_instance()
    inst["techs"][1]["rho_e"] = re_
    r = solve(inst, dual_attr=True)
    rob[f"rho_e_{re_:.2f}"] = summary(r)
    print(f"  rho_e {re_:.2f}: certified {r['certified_util']*100:.1f}% "
          f"cost {r['cost_MCNY']:.2f}M")

# (b) fines-cap proxy relaxation alone (kappa2 multiplier)
for m in (2, 4, 6, 8):
    inst = build_ops_instance()
    for d in inst["demands"]:
        d["kappa2"] = round(d["kappa2"] * m, 4)
    r = solve(inst, dual_attr=True)
    rob[f"kappa2_x{m}"] = summary(r)
    print(f"  kappa2 x{m}: certified {r['certified_util']*100:.1f}% "
          f"cost {r['cost_MCNY']:.2f}M")

# (c) district-attribute seed robustness (masonry-cap regime)
seeds = []
for sd in range(10):
    r = solve(build_ops_instance(seed=1000 + sd))
    seeds.append(round(r["certified_util"] * 100, 1))
rob["seed_certified_pct"] = seeds
print(f"  10 seeds certified range: {min(seeds)}-{max(seeds)}%")

# (c2) same seed sweep under the dual-attribute regime: the collapse itself
seeds_dual = []
for sd in range(10):
    r = solve(build_ops_instance(seed=1000 + sd), dual_attr=True)
    seeds_dual.append(round(r["certified_util"] * 100, 1))
rob["seed_certified_pct_dual"] = seeds_dual
print(f"  10 seeds dual certified range: {min(seeds_dual)}-{max(seeds_dual)}%")

# (d) market-absorption multiplier (masonry-cap regime)
for m in (0.5, 1.5):
    inst = build_ops_instance()
    for d in inst["demands"]:
        d["quantity"] = float(d["quantity"] * m)
    r = solve(inst)
    rob[f"demand_x{m}"] = summary(r)
    print(f"  demand x{m}: certified {r['certified_util']*100:.1f}% "
          f"cost {r['cost_MCNY']:.2f}M")

# (d2) market-absorption multiplier under the dual-attribute regime
for m in (0.5, 1.5):
    inst = build_ops_instance()
    for d in inst["demands"]:
        d["quantity"] = float(d["quantity"] * m)
    r = solve(inst, dual_attr=True)
    rob[f"demand_x{m}_dual"] = summary(r)
    print(f"  demand x{m} dual: certified {r['certified_util']*100:.1f}% "
          f"cost {r['cost_MCNY']:.2f}M")

# (e) mandatory plant intake under both caps (inefficient processing)
r = solve(build_ops_instance(), dual_attr=True, min_recycling=0.60)
rob["dual_minrec60"] = summary(r)
print(f"  dual + min 60% intake: certified {r['certified_util']*100:.1f}% "
      f"cost {r['cost_MCNY']:.2f}M")

results["robustness"] = rob
print(f"  bisection boundary: feasible up to s={lo:.3f}, "
      f"infeasible from s={hi:.3f}")

(OUT / "benchmark_and_sensitivity.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("saved", OUT / "benchmark_and_sensitivity.json")
