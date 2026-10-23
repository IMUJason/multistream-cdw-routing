"""Build a blend-to-specification instance for the Hohhot network.

Maps the real facility/distance/generation data to the model structure:
  - 9 district generation nodes (3 streams each)
  - 4 recycling plants (certification + technology)
  - 9 sorting/transfer centers (source separation)
  - 7 landfill sites (disposal)
  - Demand points: internal construction demand (recycled aggregate)

This produces the same instance dict format as build_beijing_case(), so
all models in blend_net.py run on it unchanged.
"""
import csv
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "hohhot"
OUT = DATA / "hohhot_blend_instance.json"


def load_facilities():
    facs = list(csv.DictReader(open(DATA / "cdw_facilities_geocoded.csv",
                                     encoding="utf-8-sig")))
    recycling = [f for f in facs if f["type"] == "资源化厂"]
    sorting = [f for f in facs if "分拣" in f["type"]]
    landfill = [f for f in facs if f["type"] == "填埋场"]
    return recycling, sorting, landfill


def load_generation():
    """District-level waste generation estimates (v3, 2024 baseline)."""
    rows = list(csv.DictReader(
        open(DATA / "cdw_generation_by_district_estimates_v3.csv",
             encoding="utf-8-sig")))
    districts = []
    for r in rows:
        name = r["区县"].strip()
        gen_2024 = float(r["2024产生量估算_万吨(基准)"])
        districts.append({"name": name, "generation_Mt": gen_2024,
                          "lat": float(r.get("lat", 0) or 0),
                          "lng": float(r.get("lng", 0) or 0)})
    return districts


def load_distances():
    """29×29 distance matrix (km)."""
    with open(DATA / "distance_matrix_wide_km.csv", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        nodes = [h.strip() for h in header[1:]]
        matrix = {}
        for row in reader:
            src = row[0].strip()
            matrix[src] = {nodes[i]: float(row[i + 1]) for i in range(len(nodes))}
    return nodes, matrix


def build_instance(stream_shares=None, seed=2024):
    """Assemble the blend-to-specification instance.

    stream_shares: dict with keys 'excavation', 'construction', 'demolition'
    representing the fraction of total waste in each stream.
    Default from Dongsheng data: 92.6% excavation, 6.1% construction, 1.3% demolition.
    seed: RNG seed for the district attribute draws (published: 2024).
    """
    if stream_shares is None:
        stream_shares = {"excavation": 0.926, "construction": 0.061,
                         "demolition": 0.013}

    recycling, sorting, landfill = load_facilities()
    districts = load_generation()
    dist_nodes, dist_matrix = load_distances()

    # Generation sites: one per district, with 3 streams each
    # For the blend model, the demolition stream carries brick share;
    # excavation and construction streams have different compositions
    sites = []
    for d in districts:
        total_t = d["generation_Mt"] * 1e4  # 万吨 -> tonnes
        # Brick share by district type (urban core older = more masonry)
        # Use the same downtown/suburban logic as Beijing
        is_urban = d["name"] in ("新城区", "回民区", "玉泉区", "赛罕区")
        b_mean = 0.52 if is_urban else 0.30
        b_std = 0.06

        # Demolition stream: carries brick share (quality attribute)
        dem_t = total_t * stream_shares["demolition"]
        # Excavation stream: mostly soil/clay (second attribute dominant)
        exc_t = total_t * stream_shares["excavation"]
        # Construction stream: mixed, lower brick
        con_t = total_t * stream_shares["construction"]

        sites.append({
            "name": d["name"],
            "total_t": total_t,
            "excavation_t": exc_t,
            "construction_t": con_t,
            "demolition_t": dem_t,
            "brick_mean": b_mean,
            "brick_std": b_std,
            "is_urban": is_urban,
        })

    # Candidate facilities: the 4 real recycling plants
    # Each can have 2 technologies (simple crushing / advanced sorting)
    facilities = []
    for i, f in enumerate(recycling):
        # Use real design capacities from treatment facility data
        real_caps = {"路雅": 100e4, "土左旗城发": 30e4, "强鑫": 50e4,
                     "亿友": 150e4}
        cap = 50e4  # default
        for key, val in real_caps.items():
            if key in f["facility"]:
                cap = val
                break
        facilities.append({
            "id": i,
            "name": f["facility"],
            "coord": [float(f["lng"]), float(f["lat"])],
            "capacity": cap,
            "fixed_cost": 2.0e7 + i * 5e6,  # estimated from literature
        })

    # Landfills: 7 real sites
    landfills = []
    for i, f in enumerate(landfill):
        landfills.append({
            "id": i,
            "name": f["facility"],
            "coord": [float(f["lng"]), float(f["lat"])],
        })

    # Demand: recycled aggregate market
    # Use the total generation as a proxy for potential demand
    total_gen = sum(s["total_t"] for s in sites)
    # Grade ladder (same as Beijing model)
    targets = [(0.05, 67.0, 1.0), (0.15, 58.0, 1.6), (0.30, 50.0, 2.2),
               (0.10, 63.0, 1.3), (0.22, 55.0, 1.9), (0.40, 52.0, 2.5)]
    market_cover = 0.85
    total_demand = market_cover * total_gen * 0.81  # mean yield
    weights = np.array([w for (_k, _p, w) in targets], dtype=float)
    qty = weights / weights.sum() * total_demand
    demands = [{"id": d, "quantity": float(qty[d]), "kappa": targets[d][0],
                "price": targets[d][1],
                "kappa2": 0.014 + 0.02 * targets[d][0]}
               for d in range(len(targets))]

    # Distance matrices (approximate from the 29×29 matrix or compute)
    # For now use the distance matrix directly
    # Node naming: G1_新城区 ... for generation, facility names for facilities
    def get_dist(node_a, node_b, default=20.0):
        if node_a in dist_matrix and node_b in dist_matrix[node_a]:
            return dist_matrix[node_a][node_b]
        return default

    # Build t_sf (site to facility) from district names to facility names
    n_sites = len(sites)
    n_fac = len(facilities)
    t_sf = np.zeros((n_sites, n_fac))
    for i, s in enumerate(sites):
        src_node = f"G{i+1}_{s['name']}"
        for j, f in enumerate(facilities):
            # Try to find the facility in the distance matrix
            fac_name = f["name"][:20]
            dist_km = get_dist(src_node, fac_name, default=20.0)
            if dist_km == 20.0:
                # Try partial match
                for dn in dist_nodes:
                    if f["name"][:8] in dn:
                        dist_km = get_dist(src_node, dn, default=20.0)
                        break
            # Transport cost: 0.55 CNY per tonne-km + fixed
            t_sf[i, j] = 0.55 * dist_km + 3.0

    # t_fd (facility to demand) - use average distance
    t_fd = np.full((n_fac, len(demands)), 15.0)  # average 15 km to market

    # Technologies
    techs = [
        {"id": 0, "label": "simple crushing", "yield": 0.90, "rho": 0.00,
         "proc_cost": 8.0, "fixed_mult": 1.00},
        {"id": 1, "label": "advanced sorting", "yield": 0.55, "rho": 0.52,
         "proc_cost": 14.0, "fixed_mult": 1.25},
    ]

    # Brick shares for each site (with uncertainty)
    rng = np.random.default_rng(seed)
    b = np.array([np.clip(s["brick_mean"] + s["brick_std"] * rng.normal(),
                          0.10, 0.85) for s in sites])

    # Second attribute (soil/clay in demolition debris)
    rng_c = np.random.default_rng(seed + 4242)
    c = np.clip(0.11 + 0.04 * rng_c.normal(0, 1.0, size=n_sites), 0.03, 0.25)

    inst = {
        "n_sites": n_sites,
        "n_fac": n_fac,
        "n_dem": len(demands),
        "n_tech": 2,
        "Q": np.array([s["demolition_t"] for s in sites]),  # demolition feed
        "Q_total": np.array([s["total_t"] for s in sites]),
        "Q_excavation": np.array([s["excavation_t"] for s in sites]),
        "Q_construction": np.array([s["construction_t"] for s in sites]),
        "b_base": b,
        "c_base": c,
        "facilities": facilities,
        "demands": demands,
        "techs": techs,
        "t_sf": t_sf,
        "t_fd": t_fd,
        "landfill_fee": 30.0,  # Hohhot posted fee level
        "district_names": [s["name"] for s in sites],
        "district_is_urban": [s["is_urban"] for s in sites],
        "stream_shares": stream_shares,
    }
    return inst


if __name__ == "__main__":
    inst = build_instance()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in inst.items()}, f, ensure_ascii=False, indent=2,
                  default=str)
    print(f"Sites: {inst['n_sites']}, Facilities: {inst['n_fac']}, "
          f"Grades: {inst['n_dem']}")
    print(f"Total generation: {sum(inst['Q_total'])/1e4:.1f} 万吨")
    print(f"Demolition feed: {sum(inst['Q'])/1e4:.1f} 万吨")
    print(f"Excavation: {sum(inst['Q_excavation'])/1e4:.1f} 万吨")
    print(f"Brick shares: {[round(x,3) for x in inst['b_base']]}")
    print(f"Saved to {OUT}")
