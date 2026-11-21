# The eight measures of the Low-Dimensional Survival Benchmark (Burk, Zobolas, Bischl, Bender, Wright and Sonabend,
# 2026), from their definitions in the papers that introduced them, with the settings the benchmark documents for each
# and the conventions that its published values follow, so that the measures of a model give the scores it published.

"""The measures of a survival dataset, in the order the benchmark lists them."""
const SURVIVAL_MEASURES = ("harrell_c", "uno_c", "isll", "isll_erv", "isbs", "isbs_erv", "dcalib", "alpha_calib")

"""The share of the test rows no longer at risk at which the integrated scores stop (`p_max` of the benchmark)."""
const SURVIVAL_P_MAX = 0.8

"""The upper bound of D-calibration and of alpha-calibration."""
const SURVIVAL_TRUNCATE = 1000.0

"""The number of equal bins of [0, 1] in which D-calibration counts the rows."""
const SURVIVAL_BINS = 10

"""The floor of the scores: of a probability in the log loss, of a censoring weight, of a cumulative hazard."""
const SURVIVAL_EPS = 1e-3

"""
	as_written(t)

The time `t` as the published values hold it: written with 15 significant digits and read back. The times of a
predicted survival curve, and those over which a score is integrated, pass through this form.
"""
as_written(t) = parse(Float64, Printf.@sprintf("%.15g", t))

"""
	tie_times(time)

The times with those that differ by less than a tolerance made equal, as the survival curves of the benchmark treat
them: among the distinct finite times in increasing order, a time whose gap from the previous one is at most sqrt(eps)
in absolute terms, or relative to the mean of the distinct times, joins the previous one's group, and every time takes
the first time of its group. Infinite and missing times are left as they are.
"""
function tie_times(time::AbstractVector{<:Real})
	t = Float64.(time)
	distinct = sort!(unique(filter(isfinite, t)))
	length(distinct) < 2 && return t
	tolerance = sqrt(eps(Float64))
	scale = sum(abs, distinct) / length(distinct)
	firsts = [distinct[1]]
	for k in 2:length(distinct)
		gap = distinct[k] - distinct[k - 1]
		gap <= tolerance || gap / scale <= tolerance || push!(firsts, distinct[k])
	end
	length(firsts) == length(distinct) && return t
	[isfinite(x) ? firsts[searchsortedlast(firsts, x)] : x for x in t]
end

"""
	survival_curve(time, event)

The Kaplan-Meier estimate (Kaplan and Meier, 1958) of the survival of the rows and the Nelson-Aalen estimate (Aalen,
1978) of their cumulative hazard, at every distinct time of the rows, once tied by `tie_times`: `times` increasing,
`at_risk` the rows whose time is not before it, `survival` the product of 1 - d/r and `hazard` the sum of d/r over the
times up to it, d being the events at a time and r the rows at risk, and `n` the rows.
"""
function survival_curve(time::AbstractVector{<:Real}, event::AbstractVector{<:Integer})
	t = tie_times(time)
	times = sort!(unique(t))
	events = zeros(Int, length(times))
	leaving = zeros(Int, length(times))
	for (x, e) in zip(t, event)
		k = searchsortedfirst(times, x)
		leaving[k] += 1
		events[k] += e == 1
	end
	at_risk = reverse(cumsum(reverse(leaving)))
	survival = cumprod(1 .- events ./ at_risk)
	hazard = cumsum(events ./ at_risk)
	(; times, at_risk, survival, hazard, n = length(t))
end

"""
	step_index(times, t)

The position in `times` (increasing) of the value of a step function at `t`: that of the last time not after `t`, or 0
when `t` comes before the first.
"""
step_index(times, t) = searchsortedlast(times, t)

"""
	mortality(survival)

The risk of each row of a matrix of survival curves (a row each), by which the C-indices rank the rows when a model
gives only curves: the sum over the curve's times of the cumulative hazard -log S(t), an infinite one counted as 0.
"""
mortality(survival::AbstractMatrix) = [sum(s -> (h = -log(s); isinf(h) ? 0.0 : h), row) for row in eachrow(survival)]

"""
	concordance(time, event, risk, weight; train_time, train_event)

The concordance index of the rows' `risk`: over the pairs of rows whose order of failure is known, those in which a row
had its event before the other row's time, the weighted share in which the row that failed first has the higher risk,
a tie in risk counting one half. With `weight = :harrell`, every pair counts the same, as in Harrell, Califf, Pryor,
Lee and Rosati (1982); with `weight = :uno`, a pair counts by the inverse square of the probability of remaining
uncensored at the earlier time, from the Kaplan-Meier estimate of the censoring of the training rows, as in Uno, Cai,
Pencina, D'Agostino and Wei (2011), a probability of 0 taken as SURVIVAL_EPS. As in the published values: the index is
0.5 when every row has the same risk; a time before every training time has a weight of -1; and the index is given in
single precision.
"""
function concordance(time, event, risk, weight::Symbol; train_time = nothing, train_event = nothing)
	weight in (:harrell, :uno) || error("weight is :harrell or :uno, not :$(weight).")
	allequal(risk) && return 0.5
	censoring = weight === :uno ? survival_curve(train_time, 1 .- train_event) : nothing
	concordant, comparable = 0.0, 0.0
	for i in eachindex(time)
		event[i] == 1 || continue
		later = [j for j in eachindex(time) if time[j] > time[i]]
		isempty(later) && continue
		w = if weight === :harrell
			1.0
		else
			k = step_index(censoring.times, time[i])
			k == 0 ? -1.0 : (g = censoring.survival[k]; (g == 0 ? SURVIVAL_EPS : g)^-2)
		end
		comparable += w * length(later)
		concordant += w * sum(risk[i] > risk[j] ? 1.0 : risk[i] == risk[j] ? 0.5 : 0.0 for j in later)
	end
	comparable == 0 && error("No pair of rows has a known order of failure: the concordance cannot be computed.")
	Float64(Float32(concordant / comparable))
end

"""
	integrated_score(loss, test_time, test_event, times, survival; train_time, train_event)

The integrated survival Brier score (`loss = :brier`) of Graf, Schmoor, Sauerbrei and Schumacher (1999), or the
integrated survival log loss (`loss = :log`), of the predicted survival curves `survival` (a row per test row, a column
per time of `times`). At each distinct test time t up to the time at which more than SURVIVAL_P_MAX of the test rows
are no longer at risk, a row still alive at t scores the loss of its predicted survival S(t) against survival, weighted
by 1/G(t), and a row that had its event by t scores the loss of S(t) against failure, weighted by 1/G(T) at its own time
T, G being the Kaplan-Meier estimate of the censoring of the training rows; a row censored by t scores 0. The loss is
(1 - S)² or S² for the Brier score, and -log of the probability given to what happened, floored at SURVIVAL_EPS, for the
log loss. The mean over the rows at each time, of its finite values, is integrated over the times by the trapezoid rule
and divided by the time spanned. As in the published values, a row alive at a time before every training time scores
0, and a weight 1/G(T) at a time before every training time is 1, or 1/SURVIVAL_EPS when G(T) is 0.
"""
function integrated_score(loss::Symbol, test_time, test_event, times, survival::AbstractMatrix; train_time, train_event)
	loss in (:brier, :log) || error("loss is :brier or :log, not :$(loss).")
	tested = survival_curve(test_time, test_event)
	stop = findfirst(r -> 1 - r / tested.n > SURVIVAL_P_MAX, tested.at_risk)
	t_max = stop === nothing ? tested.times[end] : tested.times[stop]
	scored = filter(<=(t_max), sort!(unique(Float64.(test_time))))
	predicted = as_written.(times)
	censoring = survival_curve(train_time, 1 .- train_event)
	at(t) = step_index(censoring.times, t)
	function row_loss(i, j)
		k = step_index(predicted, scored[j])
		s = k == 0 ? 1.0 : survival[i, k]
		if test_time[i] > scored[j]
			at(scored[j]) == 0 && return 0.0
			value = loss === :brier ? (1 - s)^2 : -log(max(s, SURVIVAL_EPS))
			return value / censoring.survival[at(scored[j])]
		end
		test_event[i] == 1 || return 0.0
		value = loss === :brier ? s^2 : -log(max(1 - s, SURVIVAL_EPS))
		l = at(test_time[i])
		g = l == 0 ? 1.0 : censoring.survival[l] == 0 ? SURVIVAL_EPS : censoring.survival[l]
		value / g
	end
	means = map(eachindex(scored)) do j
		values = filter(isfinite, [row_loss(i, j) for i in eachindex(test_time)])
		isempty(values) ? NaN : sum(values) / length(values)
	end
	length(scored) == 1 && return means[1]
	x = as_written.(scored)
	area = sum((x[j + 1] - x[j]) * (means[j] + means[j + 1]) / 2 for j in 1:length(x) - 1)
	area / (maximum(x) - minimum(x))
end

"""
	survival_at_own_time(test_time, times, survival)

Each test row's predicted survival at its own time, read from its curve as a step function: 1 before the curve's first
time.
"""
function survival_at_own_time(test_time, times, survival::AbstractMatrix)
	predicted = as_written.(times)
	[(k = step_index(predicted, test_time[i]); k == 0 ? 1.0 : survival[i, k]) for i in eachindex(test_time)]
end

"""
	d_calibration(test_time, test_event, times, survival)

D-calibration (Haider, Hoehn, Davis and Greiner, 2020): the predicted survival of each test row at its own time, floored
at 1e-5, falls in one of SURVIVAL_BINS equal bins of [0, 1]. A row with an event counts 1 in its bin; a censored row,
whose survival at its event time is known only to be below its value S, is shared among its bin and those below it,
1/(B S) in each bin below and the rest in its own, B being the number of bins. The measure is the chi-square statistic
of the counts against equal counts, (B/n) times the sum of the squared differences of the counts from n/B, bounded by
SURVIVAL_TRUNCATE. As in the published values, a censored row in the first bin adds 1/(B S) to it once more.
"""
function d_calibration(test_time, test_event, times, survival::AbstractMatrix)
	B = SURVIVAL_BINS
	counts = zeros(B)
	for (s, e) in zip(max.(survival_at_own_time(test_time, times, survival), 1e-5), test_event)
		j = ceil(Int, B * s)
		if e == 1
			counts[j] += 1
		else
			share = 1 / (B * s)
			counts[1:max(j - 1, 1)] .+= share
			counts[j] += 1 - (j - 1) * share
		end
	end
	n = length(test_time)
	min(SURVIVAL_TRUNCATE, (B / n) * sum(abs2, counts .- n / B))
end

"""
	alpha_calibration(test_time, test_event, times, survival)

The calibration slope alpha of van Houwelingen (2000) as a ratio: the number of events among the test rows divided by
the number the predicted curves expect, the sum of each row's cumulative hazard -log S at its own time, a hazard of 0
or infinity counted as SURVIVAL_EPS. 1 is perfect calibration; bounded by SURVIVAL_TRUNCATE.
"""
function alpha_calibration(test_time, test_event, times, survival::AbstractMatrix)
	hazard = map(survival_at_own_time(test_time, times, survival)) do s
		h = -log(s)
		h == 0 || isinf(h) ? SURVIVAL_EPS : h
	end
	min(SURVIVAL_TRUNCATE, sum(test_event) / sum(hazard))
end

"""
	survival_scores(train_time, train_event, test_time, test_event, times, survival; crank = mortality(survival))

The eight measures of the Low-Dimensional Survival Benchmark on one split, as a Dict by the names of
`SURVIVAL_MEASURES`: Harrell's C and Uno's C of `crank`; the integrated survival log loss and Brier score, and their
explained residual variation (ERV), 1 minus the ratio of the score to that of the Kaplan-Meier estimate of the same
training rows; D-calibration; and alpha-calibration. `survival` holds the predicted survival of each test row (a row
each) at `times` (a column each, increasing). The event indicators are 1 for an event and 0 for a censored time.
"""
function survival_scores(train_time, train_event, test_time, test_event, times, survival::AbstractMatrix; crank = mortality(survival))
	size(survival) == (length(test_time), length(times)) || error("The survival matrix has $(size(survival, 1)) rows and $(size(survival, 2)) columns, not $(length(test_time)) and $(length(times)).")
	train = (; train_time, train_event)
	km = survival_curve(train_time, train_event)
	baseline = repeat(permutedims(km.survival), length(test_time))
	isll = integrated_score(:log, test_time, test_event, times, survival; train...)
	isbs = integrated_score(:brier, test_time, test_event, times, survival; train...)
	Dict(
		"harrell_c" => concordance(test_time, test_event, crank, :harrell),
		"uno_c" => concordance(test_time, test_event, crank, :uno; train...),
		"isll" => isll, "isll_erv" => 1 - isll / integrated_score(:log, test_time, test_event, km.times, baseline; train...),
		"isbs" => isbs, "isbs_erv" => 1 - isbs / integrated_score(:brier, test_time, test_event, km.times, baseline; train...),
		"dcalib" => d_calibration(test_time, test_event, times, survival),
		"alpha_calib" => alpha_calibration(test_time, test_event, times, survival),
	)
end
