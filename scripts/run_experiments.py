"""Comprehensive experiment suite for the multi-stream routing study.

Five experiment blocks on the Hohhot real network:
  A. Blend vs segregate under different standard regimes
  B. Stream share sensitivity (excavation share 80-95%)
  C. Standard strictness sweep (grade threshold multiplier)
  D. Policy levers: disposal fee, mandatory recycling rate, source separation
  E. Dongsheng validation: use actual OD stream shares
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.multistream_ops import MultiStreamOps, build_ops_instance

OUT = Path(__file__).resolve().parents[1] / "results" / "multistream"
PY = None  # no subprocess needed, direct solve


def solve_scn(inst, label, time_limit=120, **kw):
    """Solve one scenario and return compact result."""
    # Handle segregation: force a_exc = 0
    segregate = kw.pop("segregate", False)
    model = MultiStreamOps(inst, **kw)
    if segregate:
        for s in range(inst["n_sites"]):
            for f in range(inst["n_fac"]):
                for t in range(inst["n_tech"]):
                    model.m.add_constraint(model.a_exc[s, f, t] == 0)
    t0 = time.time()
    r = model.solve(time_limit=time_limit)
    wall = time.time() - t0
    if r is None:
        return {"label": label, "status": "infeasible", "wall_s": wall}
    r["label"] = label
    r["status"] = "ok"
    r["wall_s"] = wall
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    # ========== A. Blend vs segregate under standard regimes ==========
    print("=== A. Blend vs segregate ===", flush=True)
    inst = build_ops_instance()
    results["A"] = []
    for label, kw in [
        ("blend_single", {}),
        ("blend_dual", {"dual_attr": True}),
        ("segregate_single", {"segregate": True}),
        ("segregate_dual", {"segregate": True, "dual_attr": True}),
    ]:
        r = solve_scn(inst, label, **kw)
        results["A"].append(r)
        if r["status"] == "ok":
            print(f"  {label:20s} cost {r['cost_MCNY']:6.0f}M cert "
                  f"{r['certified_util']*100:5.1f}% fill {r['fill_util']*100:5.1f}% "
                  f"total {r['total_util']*100:5.1f}% plants {r['facilities_active']}",
                  flush=True)
        else:
            print(f"  {label:20s} INFEASIBLE", flush=True)

    # ========== B. Stream share sensitivity ==========
    print("\n=== B. Stream share sensitivity ===", flush=True)
    results["B"] = []
    for exc_share in [0.80, 0.85, 0.90, 0.926, 0.95, 0.98]:
        # Rebuild with mass-consistent shares: the non-excavation remainder
        # (0.061 + 0.013 = 0.074 of generation) is redistributed in its
        # baseline con:dem ratio, so the three shares always sum to one.
        rem = 1.0 - exc_share
        shares = {"excavation": exc_share,
                  "construction": 0.061 * rem / 0.074,
                  "demolition": 0.013 * rem / 0.074}
        from src.models.hohhot_blend import build_instance
        base = build_instance(stream_shares=shares)
        from src.models.multistream_ops import build_ops_instance as boi
        inst_b = boi()  # use same structure but update Q values
        # Scale Q values
        total_Q = sum(base["Q_total"])
        inst_b["Q_exc"] = base["Q_excavation"]
        inst_b["Q_dem"] = base["Q"]
        inst_b["Q_con"] = base["Q_construction"]

        r1 = solve_scn(inst_b, f"exc{exc_share:.0%}_blend")
        r2 = solve_scn(inst_b, f"exc{exc_share:.0%}_dual", dual_attr=True)
        results["B"].extend([r1, r2])
        for r in [r1, r2]:
            if r["status"] == "ok":
                print(f"  exc={exc_share:.0%} {'dual' if 'dual' in r['label'] else 'blend'}: "
                      f"cert {r['certified_util']*100:.1f}% total {r['total_util']*100:.1f}%",
                      flush=True)
            else:
                print(f"  exc={exc_share:.0%} {'dual' if 'dual' in r['label'] else 'blend'}: "
                      f"INFEASIBLE", flush=True)

    # ========== C. Standard strictness sweep ==========
    print("\n=== C. Standard strictness ===", flush=True)
    results["C"] = []
    for mult in [0.5, 0.7, 1.0, 1.4, 2.0]:
        inst_c = build_ops_instance()
        for d in inst_c["demands"]:
            d["kappa"] = min(0.95, d["kappa"] * mult)
            d["kappa2"] = min(0.10, d["kappa2"] * mult)
        r1 = solve_scn(inst_c, f"std{mult}_blend")
        r2 = solve_scn(inst_c, f"std{mult}_dual", dual_attr=True)
        results["C"].extend([r1, r2])
        for r in [r1, r2]:
            if r["status"] == "ok":
                print(f"  std x{mult} {'dual' if 'dual' in r['label'] else 'blend'}: "
                      f"cert {r['certified_util']*100:.1f}%", flush=True)

    # ========== D. Policy levers ==========
    print("\n=== D. Policy levers ===", flush=True)
    results["D"] = []
    for fee in [10, 20, 30, 45, 60, 95]:
        inst_d = build_ops_instance()
        inst_d["landfill_fee"] = float(fee)
        r = solve_scn(inst_d, f"fee{fee}_blend")
        results["D"].append(r)
        if r["status"] == "ok":
            print(f"  fee={fee}: cert {r['certified_util']*100:.1f}% "
                  f"fill {r['fill_util']*100:.1f}% cost {r['cost_MCNY']:.0f}M",
                  flush=True)

    for min_rec in [0.0, 0.20, 0.40, 0.60]:
        inst_d = build_ops_instance()
        r = solve_scn(inst_d, f"rec{min_rec:.0%}_blend", min_recycling=min_rec)
        results["D"].append(r)
        if r["status"] == "ok":
            print(f"  min_rec={min_rec:.0%}: cert {r['certified_util']*100:.1f}% "
                  f"total {r['total_util']*100:.1f}%", flush=True)
        else:
            print(f"  min_rec={min_rec:.0%}: INFEASIBLE", flush=True)

    # ========== E. Dongsheng validation ==========
    print("\n=== E. Dongsheng stream shares ===", flush=True)
    results["E"] = []
    # Use actual Dongsheng shares: 92.6% exc, 6.1% con, 1.3% dem (confirmed)
    inst_e = build_ops_instance()
    r1 = solve_scn(inst_e, "dongsheng_blend")
    r2 = solve_scn(inst_e, "dongsheng_dual", dual_attr=True)
    results["E"].extend([r1, r2])

    with open(OUT / "comprehensive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nAll results saved to {OUT / 'comprehensive_results.json'}")


if __name__ == "__main__":
    main()
