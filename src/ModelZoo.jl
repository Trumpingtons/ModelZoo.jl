"""
	ModelZoo

Models of the competitions and benchmarks of Pollis that no registered Julia package has, and the measures that score
them, written from the papers and the documented behaviour of each, so that they give the published values. The
measures:

- of tabular datasets, by Response Type (`TABULAR_MEASURES`, `tabular_scores`): `rmse`, `roc_auc` and `log_loss`, as
  TabArena scores them;
- of survival datasets (`SURVIVAL_MEASURES`, `survival_scores`): the eight measures of the Low-Dimensional Survival
  Benchmark, Harrell's and Uno's C, the integrated survival Brier score and log loss and their explained residual
  variation, D-calibration and alpha-calibration;
- `measure_loss(measure, value)`, a score as a number of which lower is better, to rank by (`MEASURE_BEST`).

The models:

- `APLR` and `BaggedAPLR`, automatic piecewise linear regression, gradient boosting of linear effects, hinge functions
  and their interactions, for numbers and classes, with `predict`: TabArena's APLR (default) is `BaggedAPLR` with its
  defaults;
- `CoxGAM`, Cox's proportional hazards model with smooth terms, each a penalized thin plate regression spline whose
  smoothing parameter minimizes the REML criterion, with `linear_predictor` and `survival` for new rows: the GAM of the
  Low-Dimensional Survival Benchmark;
- `ExtraTrees` and `BaggedExtraTrees`, forests of extremely randomized trees for classes and numbers, with `predict`:
  TabArena's ExtraTrees (default) is `BaggedExtraTrees` with its defaults;
- `FlexibleSurvival`, the flexible parametric survival model of Royston and Parmar, its log cumulative hazard a natural
  cubic spline of log time, with `linear_predictor` and `survival` for new rows: the Flex of the Low-Dimensional
  Survival Benchmark for each number of knots.

```julia
import ModelZoo
ModelZoo.tabular_scores("Binary", [true, false, true, false], [0.9, 0.2, 0.4, 0.4])
curve = ModelZoo.survival_curve([5, 8, 12, 12, 20], [1, 0, 1, 1, 0])
```
"""
module ModelZoo

import LinearAlgebra
import Printf
import Random

include("measures/tabular.jl")
include("measures/survival.jl")
include("measures/ranking.jl")
include("models/cox_gam.jl")
include("models/extra_trees.jl")
include("models/aplr.jl")
include("models/flexible_survival.jl")

end
