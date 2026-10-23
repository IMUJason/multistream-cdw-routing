"""LP-bound decomposition with verified variable-type relaxation.

Patches the docplex constructors the model actually uses:
y <- Model.binary_var_matrix, w <- Model.binary_var_cube. After each
build the vartypes of y and w are asserted, so each experiment is
verified to relax exactly the variable family it claims to.

Writes results/multistream/bound_structure.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docplex.mp.model as dmp

from src.models.multistream_ops import MultiStreamOps, build_ops_instance

inst = build_ops_instance()
out = {}

NAMES = ("binary_var_matrix", "binary_var_cube", "binary_var_list")
ORIG = {name: getattr(dmp.Model, name) for name in NAMES}


def cont(name):
    target = name.replace("binary_", "continuous_")

    def maker(self, *a, **kw):
        return getattr(self, target)(*a, lb=0, ub=1, **kw)

    return maker


def solve_with(relax, label, tight=False):
    for name in NAMES:
        setattr(dmp.Model, name,
                cont(name) if name in relax else ORIG[name])
    try:
        mm = MultiStreamOps(inst, enforce_spec=True, tight_m=tight)
        y_t = type(mm.y[0, 0].vartype).__name__
        w_t = type(mm.w[0, 0, 0].vartype).__name__
        want_y = ("ContinuousVarType" if "binary_var_matrix" in relax
                  else "BinaryVarType")
        want_w = ("ContinuousVarType" if "binary_var_cube" in relax
                  else "BinaryVarType")
        assert y_t == want_y, (label, "y not relaxed as claimed", y_t)
        assert w_t == want_w, (label, "w not relaxed as claimed", w_t)
        r = mm.solve(time_limit=180)
        out[label + "_obj"] = r["objective"] if r else None
        print(label, "->", r["objective"] if r else "INFEASIBLE",
              f"(y: {y_t}, w: {w_t})")
    finally:
        for name in NAMES:
            setattr(dmp.Model, name, ORIG[name])


solve_with(set(), "milp")
solve_with({"binary_var_matrix", "binary_var_cube", "binary_var_list"},
           "lp_all")
solve_with({"binary_var_cube"}, "lp_w_only")
solve_with({"binary_var_matrix"}, "lp_y_only")

# tightened big-M variants (valid attribute-based constants)
solve_with(set(), "milp_tightM", tight=True)
solve_with({"binary_var_cube"}, "lp_w_tightM", tight=True)

milp = out["milp_obj"]
for k in ("lp_all", "lp_w_only", "lp_y_only", "lp_w_tightM"):
    v = out.get(k + "_obj")
    out[k + "_gap"] = (milp - v) / abs(milp) if v is not None else None

print("\n=== summary ===")
for k, v in out.items():
    print(f"  {k}: {v if not isinstance(v, float) else round(v, 6)}")

OUT = Path(__file__).resolve().parents[1] / "results" / "multistream"
(OUT / "bound_structure.json").write_text(json.dumps(out, indent=2))
print("saved", OUT / "bound_structure.json")
