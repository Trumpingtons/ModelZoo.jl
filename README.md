# ModelZoo.jl

Models of the competitions and benchmarks of [Pollis](https://pollis.org) that no registered Julia package has, and the measures that score them. Each is written from its paper and the documented behaviour of its benchmark, and is checked against the scores the benchmark published, on the datasets and splits of [PollisCompetitions.jl](https://github.com/Trumpingtons/PollisCompetitions.jl). A model that a registered package already has is not repeated here: the Model Catalog of PollisCompetitions.jl says where to find it.

```julia
import ModelZoo
ModelZoo.tabular_scores("Binary", [true, false, true, false], [0.9, 0.2, 0.4, 0.4])
curve = ModelZoo.survival_curve([5, 8, 12, 12, 20], [1, 0, 1, 1, 0])
```

## Measures

**Tabular datasets**, by their Response Type, as TabArena (Erickson and 6 coauthors, 2025) scores them (`TABULAR_MEASURES`, `tabular_scores(response_type, truth, prediction)`):

- Continuous and Count: `rmse`, the root mean squared error;
- Binary: `roc_auc`, the area under the ROC curve of the probability of the second class (Hanley and McNeil, 1982), ties counting one half;
- Multiclass: `log_loss`, the mean negative log-likelihood, each probability kept within the machine precision of 0 and 1, as scikit-learn keeps it (Pedregosa and 18 coauthors, 2011).

**Survival datasets**: the eight measures of the Low-Dimensional Survival Benchmark (Burk, Zobolas, Bischl, Bender, Wright and Sonabend, 2026), with its settings (`SURVIVAL_MEASURES`, `survival_scores(train_time, train_event, test_time, test_event, times, survival)`, where `survival` holds each test row's probability of surviving past each of `times`):

- `harrell_c` and `uno_c`, the concordance indices of Harrell, Califf, Pryor, Lee and Rosati (1982) and of Uno, Cai, Pencina, D'Agostino and Wei (2011);
- `isbs` and `isll`, the integrated survival Brier score (Graf, Schmoor, Sauerbrei and Schumacher, 1999) and log loss, and `isbs_erv` and `isll_erv`, their explained residual variation against the Kaplan-Meier estimate of the same training rows;
- `dcalib`, D-calibration (Haider, Hoehn, Davis and Greiner, 2020), and `alpha_calib`, the calibration ratio of van Houwelingen (2000).

`survival_curve(time, event)` gives the Kaplan-Meier (Kaplan and Meier, 1958) and Nelson-Aalen (Aalen, 1978) estimates, and `measure_loss(measure, value)` turns a score into a number of which lower is better, to rank by (`MEASURE_BEST`).

## Tests

The tests read the data, the splits and the published scores from PollisCompetitions.jl:

- the survival measures of Kaplan-Meier and Nelson-Aalen, fitted on every split of the 34 survival datasets, give the benchmark's published scores;
- the C-indices of Cox's model, fitted by Survival.jl, give the published ones on 24 of those datasets; the splits of the other 10 are listed in the test with why they differ;
- the tabular measures give the values of scikit-learn 1.7.2 on fixed predictions on every tabular dataset whose Response Type has a measure.

## Licence

MIT.
