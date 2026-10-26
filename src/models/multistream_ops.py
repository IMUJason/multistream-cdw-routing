"""Multi-stream blend model with EXISTING facilities (operations mode).

Key difference: facilities are already built (fixed cost is sunk).
The decision is: which streams to send to which facility, and whether
to blend or segregate. This mirrors the real policy question.

The excavation stream has two options:
  1. Direct fill (transport only, ~5 CNY/t value)
  2. Send to recycling plant (blend with demolition, sell certified product)

The demolition stream MUST go to a recycling plant to generate value
(landfilling it loses the certification revenue).

The blend/segregate decision: if excavation goes to the plant, it dilutes
brick share (helping certification) but increases fines (hurting dual-attr
certification). If excavation goes directly to fill, the plant processes
a richer demolition mix but handles less total volume.
"""
import numpy as np
from docplex.mp.model import Model as DocplexModel


class MultiStreamOps:
    """Operations-mode multi-stream model with existing facilities."""

    def __init__(self, inst, enforce_spec=True, dual_attr=False,
                 min_recycling=0.0, tight_m=False, indicator=False,
                 fill_value=5.0):
        self.inst = inst
        self.enforce_spec = enforce_spec
        self.dual_attr = dual_attr
        self.min_recycling = min_recycling
        self.tight_m = tight_m
        self.indicator = indicator
        self.fill_value = fill_value
        self._build()

    def _build(self):
        I = self.inst
        S, F, D, T = I["n_sites"], I["n_fac"], I["n_dem"], I["n_tech"]
        m = DocplexModel(name="multistream_ops")
        m.parameters.threads.set(1)
        self.m = m

        # Technology selection (no fixed cost - plants exist)
        y = m.binary_var_matrix(F, T, name="y")

        # Stream routing:
        # a_dem[s,f,t]: demolition from site s to facility f
        # a_exc[s,f,t]: excavation from site s to facility f (BLENDING)
        # a_con[s,f,t]: construction from site s to facility f
        a_dem = m.continuous_var_cube(S, F, T, lb=0.0, name="a_dem")
        a_exc = m.continuous_var_cube(S, F, T, lb=0.0, name="a_exc")
        a_con = m.continuous_var_cube(S, F, T, lb=0.0, name="a_con")

        # Direct fill for excavation (bypass plant)
        exc_fill = m.continuous_var_list(S, lb=0.0, name="exc_fill")

        # Landfill for each stream
        u_dem = m.continuous_var_list(S, lb=0.0, name="u_dem")
        u_exc = m.continuous_var_list(S, lb=0.0, name="u_exc")
        u_con = m.continuous_var_list(S, lb=0.0, name="u_con")

        # Certification and product
        w = m.binary_var_cube(F, T, D, name="w")
        p = m.continuous_var_cube(F, T, D, lb=0.0, name="p")
        r = m.continuous_var_matrix(F, T, lb=0.0, name="r")

        self.y = y
        self.a_dem, self.a_exc, self.a_con = a_dem, a_exc, a_con
        self.exc_fill = exc_fill
        self.u_dem, self.u_exc, self.u_con = u_dem, u_exc, u_con
        self.w, self.p, self.r = w, p, r

        phi = I["landfill_fee"]
        fill_value = self.fill_value  # CNY/t net benefit of direct fill
        obj = 0.0

        for f in range(F):
            m.add_constraint(m.sum(y[f, t] for t in range(T)) <= 1)
            cap_f = I["facilities"][f]["capacity"]
            for t in range(T):
                eta = I["techs"][t]["yield"]
                # distinct removal rates per attribute; when only "rho"
                # is given both attributes use it, so the published
                # equal-rate calibration is reproduced unchanged
                rho_b = I["techs"][t].get("rho_b", I["techs"][t]["rho"])
                rho_e = I["techs"][t].get("rho_e", I["techs"][t]["rho"])
                proc = I["techs"][t]["proc_cost"]

                if self.tight_m:
                    max_b = max(max(I["b_dem"]), max(I["b_exc"]),
                                max(I["b_con"]))
                    max_e = max(max(I["c_dem"]), max(I["c_exc"]),
                                max(I["c_con"]))
                    m_b = (1 - rho_b) * float(max_b) * cap_f
                    m_e = (1 - rho_e) * float(max_e) * cap_f
                else:
                    m_b = cap_f
                    m_e = cap_f

                total_in = (m.sum(a_dem[s, f, t] + a_exc[s, f, t] + a_con[s, f, t]
                                  for s in range(S)))
                m.add_constraint(total_in <= cap_f * y[f, t])

                out = eta * total_in
                m.add_constraint(out == m.sum(p[f, t, d] for d in range(D))
                                 + r[f, t])
                obj += phi * (1 - eta) * total_in  # residue disposal
                obj += phi * r[f, t]

                # Weighted brick share of the blend
                brick_in = m.sum(
                    float(I["b_dem"][s]) * a_dem[s, f, t]
                    + float(I["b_exc"][s]) * a_exc[s, f, t]
                    + float(I["b_con"][s]) * a_con[s, f, t]
                    for s in range(S))

                for d in range(D):
                    kap = I["demands"][d]["kappa"]
                    ship_ub = min(I["demands"][d]["quantity"], eta * cap_f)
                    m.add_constraint(p[f, t, d] <= ship_ub * w[f, t, d])
                    m.add_constraint(w[f, t, d] <= y[f, t])
                    if self.enforce_spec:
                        if self.indicator:
                            # indicator formulation: w=1 enforces the cap
                            m.add_indicator(
                                w[f, t, d],
                                (1 - rho_b) * brick_in
                                <= kap * eta * total_in, 1)
                            if self.dual_attr:
                                fines_in = m.sum(
                                    float(I["c_dem"][s]) * a_dem[s, f, t]
                                    + float(I["c_exc"][s]) * a_exc[s, f, t]
                                    + float(I["c_con"][s]) * a_con[s, f, t]
                                    for s in range(S))
                                m.add_indicator(
                                    w[f, t, d],
                                    (1 - rho_e) * fines_in
                                    <= I["demands"][d]["kappa2"] * eta
                                    * total_in, 1)
                        else:
                            m.add_constraint(
                                (1 - rho_b) * brick_in
                                <= kap * eta * total_in
                                + m_b * (1 - w[f, t, d]))
                            if self.dual_attr:
                                fines_in = m.sum(
                                    float(I["c_dem"][s]) * a_dem[s, f, t]
                                    + float(I["c_exc"][s]) * a_exc[s, f, t]
                                    + float(I["c_con"][s]) * a_con[s, f, t]
                                    for s in range(S))
                                m.add_constraint(
                                    (1 - rho_e) * fines_in
                                    <= I["demands"][d]["kappa2"] * eta
                                    * total_in
                                    + m_e * (1 - w[f, t, d]))

                    obj += (I["t_fd"][f, d] - I["demands"][d]["price"]) * p[f, t, d]

                for s in range(S):
                    obj += I["t_sf"][s, f] * (a_dem[s, f, t] + a_exc[s, f, t]
                                               + a_con[s, f, t])
                    obj += proc * (a_dem[s, f, t] + a_exc[s, f, t]
                                   + a_con[s, f, t])

        # Supply balances per stream
        for s in range(S):
            m.add_constraint(
                m.sum(a_dem[s, f, t] for f in range(F) for t in range(T))
                + u_dem[s] == I["Q_dem"][s])
            m.add_constraint(
                m.sum(a_exc[s, f, t] for f in range(F) for t in range(T))
                + exc_fill[s] + u_exc[s] == I["Q_exc"][s])
            m.add_constraint(
                m.sum(a_con[s, f, t] for f in range(F) for t in range(T))
                + u_con[s] == I["Q_con"][s])

            obj += phi * (u_dem[s] + u_exc[s] + u_con[s])
            obj -= fill_value * exc_fill[s]

        # Optional: minimum recycling rate
        if self.min_recycling > 0:
            total_waste = float(sum(I["Q_dem"]) + sum(I["Q_exc"]) + sum(I["Q_con"]))
            total_to_plant = m.sum(
                a_dem[s, f, t] + a_exc[s, f, t] + a_con[s, f, t]
                for s in range(S) for f in range(F) for t in range(T))
            m.add_constraint(total_to_plant >= self.min_recycling * total_waste)

        # Demand
        for d in range(D):
            m.add_constraint(
                m.sum(p[f, t, d] for f in range(F) for t in range(T))
                <= I["demands"][d]["quantity"])

        m.minimize(obj)

    def solve(self, time_limit=120):
        sol = self.m.solve(log_output=False, time_limit=time_limit)
        if sol is None:
            return None
        I = self.inst
        g = sol.get_value
        S, F, D, T = I["n_sites"], I["n_fac"], I["n_dem"], I["n_tech"]

        delivered = sum(g(self.p[f, t, d]) for f in range(F) for t in range(T)
                        for d in range(D))
        fill = sum(g(self.exc_fill[s]) for s in range(S))
        to_plant_dem = sum(g(self.a_dem[s, f, t]) for s in range(S)
                          for f in range(F) for t in range(T))
        to_plant_exc = sum(g(self.a_exc[s, f, t]) for s in range(S)
                          for f in range(F) for t in range(T))
        to_plant_con = sum(g(self.a_con[s, f, t]) for s in range(S)
                          for f in range(F) for t in range(T))
        landfilled = sum(g(self.u_dem[s]) + g(self.u_exc[s]) + g(self.u_con[s])
                        for s in range(S))
        total = sum(I["Q_dem"]) + sum(I["Q_exc"]) + sum(I["Q_con"])

        return {
            "objective": sol.objective_value,
            "cost_MCNY": sol.objective_value / 1e6,
            "delivered_Mt": delivered / 1e6,
            "direct_fill_Mt": fill / 1e6,
            "landfill_Mt": landfilled / 1e6,
            "to_plant_dem_Mt": to_plant_dem / 1e6,
            "to_plant_exc_Mt": to_plant_exc / 1e6,
            "to_plant_con_Mt": to_plant_con / 1e6,
            "certified_util": delivered / total,
            "fill_util": fill / total,
            "total_util": (delivered + fill) / total,
            "facilities_active": sum(1 for f in range(F) for t in range(T)
                                    if g(self.y[f, t]) > 0.5),
            "blend_ratio": to_plant_exc / max(to_plant_exc + to_plant_dem, 1),
        }


def build_ops_instance(seed=2024):
    """Build instance with per-stream data for operations model.

    seed controls the district attribute draws (brick and fines);
    the published calibration uses seed=2024.
    """
    from src.models.hohhot_blend import build_instance
    base = build_instance(seed=seed)
    S = base["n_sites"]
    rng = np.random.default_rng(2024)

    # Per-stream compositions
    b_dem = base["b_base"]  # demolition brick shares (0.22-0.62)
    b_exc = np.full(S, 0.05)  # excavation: minimal brick
    b_con = np.full(S, 0.15)  # construction: moderate

    c_dem = base["c_base"]  # demolition fines (0.07-0.15)
    c_exc = np.full(S, 0.60)  # excavation: mostly fines/soil
    c_con = np.full(S, 0.20)  # construction: moderate

    # Quantities
    Q_dem = base["Q"]  # demolition
    Q_exc = base["Q_excavation"]  # excavation
    Q_con = base["Q_construction"]  # construction

    base.update({
        "b_dem": b_dem, "b_exc": b_exc, "b_con": b_con,
        "c_dem": c_dem, "c_exc": c_exc, "c_con": c_con,
        "Q_dem": Q_dem, "Q_exc": Q_exc, "Q_con": Q_con,
    })
    return base


if __name__ == "__main__":
    inst = build_ops_instance()
    print("=== Multi-stream operations model (Hohhot) ===")
    print(f"Demolition: {sum(inst['Q_dem'])/1e4:.1f}万t | "
          f"Excavation: {sum(inst['Q_exc'])/1e4:.1f}万t | "
          f"Construction: {sum(inst['Q_con'])/1e4:.1f}万t")

    scenarios = [
        ("Free routing", {}),
        ("Min 60% recycling", {"min_recycling": 0.60}),
        ("Min 60% + dual-attr", {"min_recycling": 0.60, "dual_attr": True}),
        ("Segregated (exc→fill only)", {"min_recycling": 0.60,
                                         "segregate": True}),
    ]

    for label, kw in scenarios:
        if kw.pop("segregate", False):
            # Force excavation NOT to go to plant (segregation)
            model = MultiStreamOps(inst, enforce_spec=True,
                                   min_recycling=kw.get("min_recycling", 0),
                                   dual_attr=kw.get("dual_attr", False))
            # Add segregation constraint: a_exc = 0
            for s in range(inst["n_sites"]):
                for f in range(inst["n_fac"]):
                    for t in range(inst["n_tech"]):
                        model.m.add_constraint(
                            model.a_exc[s, f, t] == 0,
                            ctname=f"segregate_{s}_{f}_{t}")
        else:
            model = MultiStreamOps(inst, enforce_spec=True, **kw)

        r = model.solve(time_limit=120)
        if r:
            print(f"\n{label}:")
            print(f"  cost {r['cost_MCNY']:.0f}M | plants {r['facilities_active']}")
            print(f"  certified {r['certified_util']*100:.1f}% | "
                  f"fill {r['fill_util']*100:.1f}% | "
                  f"total util {r['total_util']*100:.1f}%")
            print(f"  to plant: dem {r['to_plant_dem_Mt']:.2f} | "
                  f"exc {r['to_plant_exc_Mt']:.2f} | "
                  f"con {r['to_plant_con_Mt']:.2f} Mt")
            print(f"  blend ratio (exc/plant total): {r['blend_ratio']*100:.0f}%")
        else:
            print(f"\n{label}: INFEASIBLE")
