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
- Multiclass: `log_loss`, the mean negative log-likelihood, each probability kept within the machine precision of its type of 0 and 1, as scikit-learn keeps it (Pedregosa and 18 coauthors, 2011).

**Survival datasets**: the eight measures of the Low-Dimensional Survival Benchmark (Burk, Zobolas, Bischl, Bender, Wright and Sonabend, 2026), with its settings (`SURVIVAL_MEASURES`, `survival_scores(train_time, train_event, test_time, test_event, times, survival)`, where `survival` holds each test row's probability of surviving past each of `times`):

- `harrell_c` and `uno_c`, the concordance indices of Harrell, Califf, Pryor, Lee and Rosati (1982) and of Uno, Cai, Pencina, D'Agostino and Wei (2011);
- `isbs` and `isll`, the integrated survival Brier score (Graf, Schmoor, Sauerbrei and Schumacher, 1999) and log loss, and `isbs_erv` and `isll_erv`, their explained residual variation against the Kaplan-Meier estimate of the same training rows;
- `dcalib`, D-calibration (Haider, Hoehn, Davis and Greiner, 2020), and `alpha_calib`, the calibration ratio of van Houwelingen (2000).

`survival_curve(time, event)` gives the Kaplan-Meier (Kaplan and Meier, 1958) and Nelson-Aalen (Aalen, 1978) estimates, and `measure_loss(measure, value)` turns a score into a number of which lower is better, to rank by (`MEASURE_BEST`).

## Models

**Cox GAM** (`CoxGAM(time, event, X)`, or `CoxGAM(table, :time, :event)` for a table of columns): Cox's proportional hazards model with smooth terms, the GAM of the Low-Dimensional Survival Benchmark. Each numeric covariate with 10 or more distinct values enters as a thin plate regression spline of dimension 10 (Wood, 2003), the others linearly; the coefficients maximize the partial likelihood with Breslow's handling of ties, penalized by the splines' wiggliness, and the smoothing parameters minimize the REML criterion of Wood, Pya and Säfken (2016), by Newton's method (Wood, 2011). The criterion can have more than one local minimum, so the search starts from three points and keeps the lowest minimum; mgcv, which the benchmark ran, searches from one start, which Wood (2011) describes only in outline, so on some splits the two end at different minima. `linear_predictor(model, X)` gives the log relative risk of new rows and `survival(model, X, times)` their survival, from the Breslow estimate of the baseline hazard.

**Automatic piecewise linear regression** (`APLR(X, y)` and `BaggedAPLR(X, y)`, or `(table, :target)` for a table of columns): the model of von Ottenbreit and De Bin (2024) with the defaults of the Python package aplr 10.26, for numbers and classes. It is gradient boosting of terms that are linear effects or hinge functions of one covariate, max(x − s, 0) or min(x − s, 0), set to 0 where hinge functions of another covariate already in the model are 0 (interactions of two covariates). Each of up to 3000 steps moves the intercept, then adds the term, among the 7 that fitted best lately, whose coefficient fitted to the negative gradient by least squares (shrunk by the learning rate 0.5, and by half for interactions) reduces the squared error most; split points are searched among the boundaries of up to 300 bins of the covariate. The validation rows choose the number of steps, boosting stopping after 200 steps without improvement. Numbers use the squared error loss, and their predictions are kept within the range of the training predictions and responses; classes are modelled one against the rest by the logit link and the binomial loss, two classes by one model. A missing value is replaced by the median, with a column marking where it was missing, and the levels of a column that is not numeric become a column each. `BaggedAPLR` averages 8 models, each with one of 8 folds of the rows as its validation rows, as TabArena's APLR (default). `predict(model, X)` gives the probability of each class, or the predicted number. The models of the bag are fitted on Julia's threads.

**ExtraTrees** (`ExtraTrees(X, y)` and `BaggedExtraTrees(X, y)`, or `(table, :target)` for a table of columns): forests of extremely randomized trees (Geurts, Ernst and Wehenkel, 2006) for classes and numbers, with scikit-learn's settings as AutoGluon fits them: 300 trees, each on a bootstrap sample of the rows, grown until its leaves are pure or it has 15000 leaves, splitting each node at the best of random splits of √p covariates for classes and all p for numbers. A missing value is NaN: the threshold is drawn among the values that are not missing, and the rows whose value is missing go to a side drawn at random, as scikit-learn's random splits send them. `BaggedExtraTrees` averages 8 forests, each fitted without one of 8 folds of the rows, as TabArena's ExtraTrees (default). `predict(model, X)` gives the probability of each class, or the predicted number. The trees are grown on Julia's threads.

**Flexible parametric survival model** (`FlexibleSurvival(time, event, X; k)`): the model of Royston and Parmar (2002) on the proportional hazards scale, the Flex of the Low-Dimensional Survival Benchmark for each number of knots `k`. The log cumulative hazard is a natural cubic spline of log time plus a linear function of the covariates, with the knots flexsurv's `flexsurvspline` places by default: boundary knots at the smallest and largest log event times and `k` internal knots at equally spaced quantiles of the log event times. The coefficients maximize the likelihood of the right-censored times, which is concave in them, by Newton's method. `linear_predictor(model, X)` gives the spline's constant plus the covariates' effects and `survival(model, X, times)` the survival of new rows.

## Tests

The tests read the data, the splits and the published scores from PollisCompetitions.jl:

- the survival measures of Kaplan-Meier and Nelson-Aalen, fitted on every split of the 34 survival datasets, give the benchmark's published scores;
- the C-indices of Cox's model, fitted by Survival.jl, give the published ones on 24 of those datasets; the splits of the other 10 are listed in the test with why they differ;
- the eight measures of `CoxGAM` give the published means of the benchmark's GAM on 15 of those datasets. Of the other 19, 4 have test rows with a level the training rows lack, which the benchmark imputes at random; 6 have a covariate with more than 2000 distinct values, of which mgcv draws 2000 at random to build its spline; and 9 have splits on which mgcv stopped at a local minimum of the REML criterion higher than the one `CoxGAM` reaches from its three starts, listed in the test;
- `BaggedExtraTrees` scores, on every split of the smallest dataset of each Response Type (Binary, Continuous, Multiclass) whose covariates are all numbers with none missing, within 4 standard errors of TabArena's ExtraTrees (default) on the same splits (`test/extratrees_tabarena.toml`, written by `scripts/tabarena_scores.py`): the two draw different random numbers, so they can agree only in distribution;
- `BaggedAPLR` scores, on the same splits of the datasets chosen by the same rule, within 4 standard errors of TabArena's APLR (default) (`test/aplr_tabarena.toml`, written by `scripts/tabarena_scores.py`): AutoGluon splits the rows into its 8 folds with other random numbers, so the two can agree only in distribution;
- `FlexibleSurvival`, fitted with each number of knots from 1 to 10 on every split of the three smallest survival datasets whose covariates are all numbers, gives on each split the Harrell's and Uno's C of the benchmark's Flex tuned on Harrell's C, and within 1e-3 its integrated Brier score and log loss tuned on the integrated Brier score (`test/flexible_benchmark.toml`, written by `scripts/flexible_benchmark.py`). The benchmark tuned the number of knots on inner splits that were not published, so the published score must be that of one of the ten. flexsurv's search stops near the maximum of the likelihood, where `FlexibleSurvival` reaches it, so the integrated scores differ by up to about 2e-4, and on the splits listed in the test the C-indices differ;
- the tabular measures give the values of scikit-learn 1.7.2 on fixed predictions on every tabular dataset whose Response Type has a measure.

## Licence

MIT.
