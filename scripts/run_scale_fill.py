"""Scalability check and direct-fill credit sweep.

1. Scale test: tile the nine districts three times (27 generation
   sites) and duplicate the four plants (8 facilities, capacities
   unchanged), then solve both cap regimes with the big-M model and
   the masonry-cap regime with the indicator formulation. Reports
   model sizes and wall times.
2. Fill-value sweep: 0 / 5 / 10 / 15 CNY per tonne direct-fill credit
   on the base instance (masonry-cap regime).

Writes results/multistream/scale_and_fill.json.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.multistream_ops import MultiStreamOps, build_ops_instance

RES = Path(__file__).resolve().parents[1] / "results" / "multistream"
RES.mkdir(parents=True, exist_ok=True)


def scaled_instance(k_sites=3, k_fac=2):
    base = build_ops_instance()
    inst = copy.deepcopy(base)
    S, F, D, T = base["n_sites"], base["n_fac"], base["n_dem"], base["n_tech"]
    S2, F2 = S * k_sites, F * k_fac

    def tile(v, reps):
        return np.tile(v, reps)

    inst["n_sites"], inst["n_fac"] = S2, F2
    for key in ("Q", "Q_total", "Q_excavation", "Q_construction",
                "Q_dem", "Q_exc", "Q_con",
                "b_dem", "b_exc", "b_con", "c_dem", "c_exc", "c_con"):
        inst[key] = tile(base[key], k_sites)
    inst["t_sf"] = np.tile(base["t_sf"], (k_sites, k_fac))
    inst["t_fd"] = np.tile(base["t_fd"], (k_fac, 1))
    inst["facilities"] = [copy.deepcopy(f) for _ in range(k_fac)
                          for f in base["facilities"]]
    inst["district_names"] = tile(np.array(base["district_names"]), k_sites).tolist()
    inst["district_is_urban"] = tile(np.array(base["district_is_urban"]),
                                     k_sites).tolist()
    return inst


def timed_solve(inst, **kw):
    model = MultiStreamOps(inst, **kw)
    n_var = model.m.number_of_variables
    n_bin = inst["n_fac"] * inst["n_tech"] * (1 + inst["n_dem"])
    n_con = model.m.number_of_constraints
    t0 = time.perf_counter()
    r = model.solve()
    wall = round(time.perf_counter() - t0, 3)
    return r, {"variables": n_var, "binaries": n_bin, "constraints": n_con,
               "wall_s": wall}


out = {"scale_test": {}, "fill_value_sweep": []}

big = scaled_instance()
out["scale_test"]["sites"] = big["n_sites"]
out["scale_test"]["plants"] = big["n_fac"]

r, meta = timed_solve(big, enforce_spec=True, dual_attr=False)
out["scale_test"]["single"] = {
    "cost_MCNY": round(r["cost_MCNY"], 2),
    "certified_pct": round(r["certified_util"] * 100, 1),
    **meta}
r2, meta2 = timed_solve(big, enforce_spec=True, dual_attr=True)
out["scale_test"]["dual"] = {
    "cost_MCNY": round(r2["cost_MCNY"], 2),
    "certified_pct": round(r2["certified_util"] * 100, 1),
    **meta2}
r3, meta3 = timed_solve(big, enforce_spec=True, dual_attr=False,
                        indicator=True)
out["scale_test"]["indicator_single"] = {
    "cost_MCNY": round(r3["cost_MCNY"], 2),
    "certified_pct": round(r3["certified_util"] * 100, 1),
    **meta3}

base = build_ops_instance()
for v in (0.0, 5.0, 10.0, 15.0):
    r, meta = timed_solve(base, enforce_spec=True, dual_attr=False,
                          fill_value=v)
    out["fill_value_sweep"].append({
        "vfill": v,
        "cost_MCNY": round(r["cost_MCNY"], 2),
        "certified_pct": round(r["certified_util"] * 100, 1),
        "fill_pct": round(r["fill_util"] * 100, 1),
        "wall_s": meta["wall_s"]})
    print(v, r["certified_util"], r["cost_MCNY"])

(RES / "scale_and_fill.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("saved", RES / "scale_and_fill.json")
print(json.dumps(out["scale_test"], indent=2)[:600])
