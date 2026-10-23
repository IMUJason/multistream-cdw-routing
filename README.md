# Multi-stream CDW routing under grade certification — minimal reproducibility package

Code and stored results for reproducing the computational study
"Quality-constrained routing of multi-stream construction and
demolition waste under certification standards" (operations-mode
model: nine district generation nodes, four existing recycling plants,
two processing technologies, six product grades, certification caps on
masonry and fines content of the blended product).

## Contents

```
src/models/multistream_ops.py   operations MILP (dual removal rates,
                                optional indicator formulation,
                                tight-M variant, minimum-intake
                                constraint, seed-controlled instances)
src/models/hohhot_blend.py      instance builder from the public-data
                                CSVs in data/hohhot/
scripts/run_experiments.py      strategy x regime matrix, stream-share
                                sweep, cap-strictness sweep, fee and
                                mandatory-intake sweeps
scripts/run_benchmarks.py       representation benchmarks (attribute
                                aggregation vs single-flow merge),
                                paired price-cost grid, fines-scaling
                                bisection, robustness block
scripts/run_sensitivity_variants.py  two-dimensional proxy/e_exc
                                sweep, indicator formulation, single-
                                grade-per-line, demand multipliers
scripts/diag_bound_structure.py LP relaxation decomposition with
                                programmatically asserted variable
                                types, incl. tightened big-M variants
data/hohhot/                    public government-report data
results/multistream/            stored anchor outputs
```

## Environment

Python 3.10 with `docplex` and a full CPLEX 22.1.x engine (CPLEX
Studio, e.g. 22.1.1; the free community engine's 1000-variable limit
is not sufficient). A conda recipe:

```yaml
name: cdw-multistream
dependencies:
  - python=3.10
  - pip
  - pip:
      - docplex>=2.25
      - numpy
```

CPLEX Studio's Python API must be installed from the Studio
distribution and matched to the same Python 3.10.

## Reproduce

From the repository root (module form so `src` resolves):

```
python -m scripts.run_experiments          # strategy/regime matrix, sweeps
python -m scripts.run_benchmarks           # benchmarks, price-cost grid,
                                           # fines scaling, robustness
python -m scripts.run_sensitivity_variants # 2-D sweep, indicator,
                                           # single-grade, demand grid
python -m scripts.diag_bound_structure     # LP decomposition + tight M
```

All randomness is seed-controlled; runners overwrite the JSON files in
`results/multistream/`. Instances solve in under 0.05 s each at the
solver's default tolerances.

## Result mapping

| Study element | Runner | Stored result |
|---|---|---|
| Blend/segregate x cap-regime matrix (Table 3) | run_experiments (block A) | comprehensive_results.json ["A"] |
| Stream-share sweep (Fig. 3b) | run_experiments (block B) | comprehensive_results.json ["B"] |
| Disposal-fee sweep (Fig. 3a) | run_experiments (block D) | comprehensive_results.json ["D"] |
| Representation benchmarks (Table 3, rows 5-8) | run_benchmarks (block 1) | benchmark_and_sensitivity.json ["benchmark"] |
| Paired price-cost grid (Table 5) | run_benchmarks (block 2) | benchmark_and_sensitivity.json ["price_cost_grid"] |
| Fines-composition scaling + bisection | run_benchmarks (block 3) | benchmark_and_sensitivity.json ["fines_scaling"] |
| Robustness block (removal rates, proxy multipliers, seeds, demand, dual intake) | run_benchmarks (block 4) | benchmark_and_sensitivity.json ["robustness"] |
| Two-dimensional proxy/e_exc sweep, indicator formulation, single-grade restriction, demand multipliers | run_sensitivity_variants | benchmark_and_sensitivity.json ["sensitivity_variants"] |
| Relaxation decomposition and tight-M comparison | diag_bound_structure | bound_structure.json |

## Data provenance

`data/hohhot/` is compiled from public government reports (municipal
solid-waste information disclosures): facility registry with design
capacities and coordinates, district generation estimates, and a
29x29 road-distance matrix. The stream-split calibration record is not
redistributable and is therefore not included; the calibration
constants are fully specified in the paper.
