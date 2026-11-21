# Automatic piecewise linear regression (APLR) of von Ottenbreit and De Bin (2024), as the Python package aplr 10.26
# fits it: componentwise gradient boosting whose terms are linear effects and hinge functions of one covariate, max(x - s,
# 0) or min(x - s, 0), possibly set to 0 where hinge functions of other covariates (its given terms) are 0, an
# interaction. In each boosting step the intercept moves by the learning rate times the mean negative gradient; each
# eligible term then finds its best split point among the bin boundaries of its covariate and the coefficient fitting
# the negative gradient by least squares with a small ridge penalty, shrunk by the learning rate (and by the interaction
# penalty for interactions); the term reducing the squared error most enters the model, or adds to its coefficient.
# Interactions are formed with the model terms whose search error was lowest when they entered the model. A term not
# among the max_eligible_terms best of a step leaves the search for ineligible_boosting_steps_added steps. The validation
# rows choose the number of steps: boosting stops after early_stopping_rounds steps without a lower validation error,
# and the model is that of the step with the lowest. Classes are modelled one against the rest by the logit link and
# the binomial loss (two classes by one model), the probabilities normalized to sum to 1. The behaviour the package's
# documentation leaves out follows its source; the code is written anew.

"""The rows a term searches on (those where none of its given terms is 0), sorted by its covariate."""
struct TermRows
	givens::Int
	rows::Vector{Int}
	values::Vector{Float64}
	zeroed::Vector{Int}
end

"""
The bins of a term's split point search, set up on its first search and kept by the term: their first and last
positions in the sorted rows (from 0), the split points at their starts (`left`) and ends (`right`) with the bin's
number, and the weighted mean of the covariate and the weight of each bin.
"""
struct TermBins
	starts::Vector{Int}
	stops::Vector{Int}
	left::Vector{Tuple{Int, Float64}}
	right::Vector{Tuple{Int, Float64}}
	xbar::Vector{Float64}
	weight::Vector{Float64}
end

"""
	APLRTerm

A term of an APLR model: the hinge function, or linear effect when `split` is NaN, of the covariate `base` (`right` for
max(x - split, 0), else min(x - split, 0)), set to 0 where any of `givens` is 0, times `coefficient`. `error` is the
squared error of its last split point search, `ineligible` the boosting steps it is left out of the search (typemax
for good), `can_be_given` whether its hinge function is 0 on a training row, so that it can be a given term.
"""
mutable struct APLRTerm
	base::Int
	split::Float64
	right::Bool
	givens::Vector{APLRTerm}
	coefficient::Float64
	error::Float64
	ineligible::Int
	can_be_given::Bool
	rows::Union{Nothing, TermRows}
	bins::Union{Nothing, TermBins}
end

APLRTerm(base::Integer, givens::Vector{APLRTerm} = APLRTerm[]) = APLRTerm(base, NaN, false, givens, 0.0, Inf, 0, false, nothing, nothing)

# A copy as the package copies a term: its rows, bins and ineligibility are not kept
copy_term(t::APLRTerm; givens = [copy_term(g) for g in t.givens]) =
	APLRTerm(t.base, t.split, t.right, givens, t.coefficient, t.error, 0, t.can_be_given, nothing, nothing)

approx_equal(a, b) = (isinf(a) && isinf(b)) ? signbit(a) == signbit(b) :
	abs(a - b) <= max((isinf(a) || isinf(b) ? min(abs(a), abs(b)) : max(abs(a), abs(b))) * eps(), eps())
approx_zero(a) = approx_equal(a, 0.0)

hinge(t::APLRTerm, x) = isnan(t.split) ? Float64(x) : t.right ? max(x - t.split, 0.0) : min(x - t.split, 0.0)

"""Whether none of the term's given terms is 0, on each row of `X`."""
function given_nonzero(t::APLRTerm, X::AbstractMatrix{<:Real})
	keep = trues(size(X, 1))
	for g in t.givens
		keep .&= .!approx_zero.(term_values(g, X))
	end
	keep
end

"""
	term_values(term, X)

The values of an APLR term (before its coefficient) on the rows of `X`.
"""
function term_values(t::APLRTerm, X::AbstractMatrix{<:Real})
	values = [hinge(t, x) for x in view(X, :, t.base)]
	isempty(t.givens) || (values[.!given_nonzero(t, X)] .= 0.0)
	values
end

interaction_level(t::APLRTerm) = length(unique([t.base; [g.base for g in t.givens]])) - 1

same_hinge(a::APLRTerm, b::APLRTerm) = a.base == b.base &&
	((approx_equal(a.split, b.split) && a.right == b.right) || (isnan(a.split) && isnan(b.split)))
same_givens(a::APLRTerm, b::APLRTerm) = length(a.givens) == length(b.givens) &&
	all(any(same_hinge(g, h) for h in b.givens) for g in a.givens)
same_term(a::APLRTerm, b::APLRTerm) = same_hinge(a, b) && same_givens(a, b)

"""The settings of an APLR boosting run, with the package's names."""
struct APLRSettings
	m::Int
	v::Float64
	bins::Int
	max_interaction_level::Int
	min_observations::Int
	ineligible_steps::Int
	max_eligible_terms::Int
	early_stopping_rounds::Int
	penalty_for_interactions::Float64
	penalty_for_non_linearity::Float64
	ridge_penalty::Float64
end

function term_rows(t::APLRTerm, X::AbstractMatrix{<:Real})
	keep = given_nonzero(t, X)
	rows = findall(keep)
	values = Float64.(X[rows, t.base])
	order = sortperm(values)
	TermRows(length(t.givens), rows[order], values[order], findall(.!keep))
end

# The weighted mean of `x` (in the order of the sorted rows) over positions s to e (from 0), the plain mean when the
# weight is not positive
function bin_mean(x::AbstractVector{Float64}, ws::AbstractVector{Float64}, s::Int, e::Int, W::Float64)
	positions = (s + 1):(e + 1)
	W > 0 ? sum(x[q] * ws[q] for q in positions) / W : sum(x[q] for q in positions) / length(positions)
end

# The bins of the sorted values: boundaries only where the value changes, none within min_observations of either end,
# at most `bins` of them, spread so that the bins hold about equal numbers of rows
function term_bins(R::TermRows, w::Vector{Float64}, bins::Int, min_observations::Int)
	values = R.values
	last = length(values) - 1
	starts = [0]
	if bins > 1
		candidates = [i for i in min_observations:(length(values) - min_observations) if i > 0 && !approx_equal(values[i + 1], values[i])]
		if !isempty(candidates)
			if length(candidates) < bins
				append!(starts, candidates)
			elseif bins == 2
				push!(starts, candidates[1])
			elseif bins == 3
				push!(starts, candidates[1], candidates[end])
			else
				push!(starts, candidates[1])
				desired = max((candidates[end] - candidates[1]) ÷ (bins - 2) + 1, 1)
				desired_second_last = desired * 4 ÷ 5
				previous = 1
				for k in 2:(length(candidates) - 2)
					if candidates[k] - candidates[previous] >= desired && candidates[end] - candidates[k] >= desired_second_last
						push!(starts, candidates[k])
						previous = k
					end
				end
				push!(starts, candidates[end])
			end
		end
	end
	stops = [starts[2:end] .- 1; last]
	left = [(b, values[s + 1]) for (b, s) in enumerate(starts) if 0 < s < last]
	right = [(b, values[e + 1]) for (b, e) in enumerate(stops) if 0 < e < last]
	ws = w[R.rows]
	weight = [sum(view(ws, (s + 1):(e + 1))) for (s, e) in zip(starts, stops)]
	xbar = [bin_mean(values, ws, s, e, W) for (s, e, W) in zip(starts, stops, weight)]
	TermBins(starts, stops, left, right, xbar, weight)
end

function make_ineligible!(t::APLRTerm)
	t.coefficient, t.error, t.ineligible = 0.0, Inf, typemax(Int)
	t
end

# The coefficient and squared error of fitting `g` by `c h` from the sums S(hg), S(hh) and S(gg), the coefficient shrunk
# by `factor` and the denominator increased by the ridge penalty
function coefficient_and_error(shg, shh, sgg, factor, ridge)
	c = factor * shg / (shh + ridge)
	isfinite(c) ? (c, sgg - 2c * shg + c^2 * shh) : (0.0, Inf)
end

interaction_factor(t::APLRTerm, settings::APLRSettings) = interaction_level(t) > 0 ? 1 - settings.penalty_for_interactions : 1.0

"""
	estimate!(term, X, g, w, ridge_weight, settings)

Finds the term's split point and coefficient for the negative gradient `g` and weights `w` on the training rows `X`,
and its squared error, as the package's split point search does: the split on the bins, then the coefficient and error
on the rows themselves. The rows follow the term's given terms; the bins are those of its first search.
"""
function estimate!(t::APLRTerm, X::AbstractMatrix{<:Real}, g::Vector{Float64}, w::Vector{Float64}, ridge_weight::Float64, settings::APLRSettings)
	(t.rows === nothing || t.rows.givens != length(t.givens)) && (t.rows = term_rows(t, X))
	R = t.rows::TermRows
	length(R.rows) < settings.min_observations && return make_ineligible!(t)
	t.bins === nothing && (t.bins = term_bins(R, w, settings.bins, max(settings.min_observations, 1)))
	B = t.bins::TermBins
	length(B.starts) <= 1 && interaction_level(t) == 0 && return make_ineligible!(t)
	ridge = settings.ridge_penalty > 0 ? settings.ridge_penalty * ridge_weight : 0.0
	gs, ws = g[R.rows], w[R.rows]
	W = B.weight
	gbar = [bin_mean(gs, ws, s, e, Wb) for (s, e, Wb) in zip(B.starts, B.stops, W)]
	sgg = sum(W .* gbar .^ 2)
	shrink = settings.v * interaction_factor(t, settings)
	# The linear effect, unless a given term has the same covariate
	t.split = NaN
	error_linear = Inf
	if all(h -> h.base != t.base, t.givens)
		_, error_linear = coefficient_and_error(sum(W .* B.xbar .* gbar), sum(W .* B.xbar .^ 2), sgg, shrink, ridge)
	end
	if settings.penalty_for_non_linearity < 1
		factor = shrink * (1 - settings.penalty_for_non_linearity)
		# Sums over the bins below a bin (min(x - s, 0)) and above it (max(x - s, 0)), the covariate centred for accuracy
		centre = sum(W .* B.xbar) / sum(W)
		z = B.xbar .- centre
		Szg, Sg, Szz, Sz, S1 = cumsum(W .* z .* gbar), cumsum(W .* gbar), cumsum(W .* z .^ 2), cumsum(W .* z), cumsum(W)
		below(a, b) = b <= 1 ? 0.0 : a[b - 1]
		above(a, b) = a[end] - a[b]
		split_left, error_left = NaN, error_linear
		for (b, s) in Iterators.reverse(B.left)
			d = s - centre
			_, e = coefficient_and_error(below(Szg, b) - d * below(Sg, b), below(Szz, b) - 2d * below(Sz, b) + d^2 * below(S1, b), sgg, factor, ridge)
			e < error_left && ((split_left, error_left) = (s, e))
		end
		split_right, error_right = NaN, error_linear
		for (b, s) in B.right
			d = s - centre
			_, e = coefficient_and_error(above(Szg, b) - d * above(Sg, b), above(Szz, b) - 2d * above(Sz, b) + d^2 * above(S1, b), sgg, factor, ridge)
			e < error_right && ((split_right, error_right) = (s, e))
		end
		t.right, t.split = error_left < error_right ? (false, split_left) : (true, split_right)
	end
	# Given terms of the same covariate that the split makes redundant
	filter!(h -> !(h.base == t.base && (h.right == t.right || !isfinite(h.split))), t.givens)
	# The coefficient and error on the rows themselves, those the given terms zero adding their squared gradient
	h = [hinge(t, x) for x in R.values]
	factor = settings.v * interaction_factor(t, settings) * (isnan(t.split) ? 1.0 : 1 - settings.penalty_for_non_linearity)
	c = factor * sum(h .* gs .* ws) / (sum(h .^ 2 .* ws) + ridge)
	if isfinite(c)
		t.coefficient = c
		t.error = sum(ws .* (gs .- c .* h) .^ 2) + sum((w[i] * g[i]^2 for i in R.zeroed); init = 0.0)
	else
		t.coefficient, t.error = 0.0, Inf
	end
	t.can_be_given = any(x -> approx_zero(hinge(t, x)), view(X, :, t.base))
	t
end

"""A boosted APLR model: an intercept and terms on the linear predictor, its link, and the bounds of its predictions."""
struct APLRModel
	intercept::Float64
	terms::Vector{APLRTerm}
	logit::Bool
	low::Float64
	high::Float64
end

logistic_clipped(eta) = (e = exp(clamp(eta, -16.0, 16.0)); e / (1 + e))

function linear_predictor(model::APLRModel, X::AbstractMatrix{<:Real})
	eta = fill(model.intercept, size(X, 1))
	for t in model.terms
		eta .+= t.coefficient .* term_values(t, X)
	end
	eta
end

# The given terms of an interaction with a model term: the model term's own given terms and its hinge function, keeping,
# when there are more than one, only those that are 0 on a training row where all the others are not
function necessary_givens(term::APLRTerm, X::AbstractMatrix{<:Real})
	candidates = [[copy_term(h) for h in term.givens]; copy_term(term; givens = APLRTerm[])]
	length(candidates) == 1 && return candidates
	nonzero = [.!approx_zero.(term_values(h, X)) for h in candidates]
	[candidates[c] for c in eachindex(candidates)
		if any(i -> !nonzero[c][i] && all(nonzero[o][i] for o in eachindex(candidates) if o != c), axes(X, 1))]
end

"""
	boost(X, y, train, validation, logit, settings)

One APLR boosting run on the rows `train`, its number of steps chosen on the rows `validation`.
"""
function boost(X::Matrix{Float64}, y::Vector{Float64}, train::Vector{Int}, validation::Vector{Int}, logit::Bool, settings::APLRSettings)
	Xt, Xv = X[train, :], X[validation, :]
	yt, yv = y[train], y[validation]
	w, wv = ones(length(train)), ones(length(validation))
	p = size(X, 2)
	transform(eta) = logit ? logistic_clipped.(eta) : eta
	function gradient(eta)
		logit || return yt .- eta
		mu = logistic_clipped.(eta)
		(yt ./ mu .- (yt .- 1) ./ (mu .- 1)) .* (2.5 .* sech.(eta ./ 2) .^ 2)
	end
	function validation_error(eta)
		mu = transform(eta)
		losses = logit ? -yv .* log.(mu) .- (1 .- yv) .* log.(1 .- mu) : (yv .- mu) .^ 2
		sum(wv .* losses) / sum(wv)
	end
	eligible = [APLRTerm(j) for j in 1:p]
	for (j, t) in enumerate(eligible)
		all(x -> approx_equal(x, Xt[1, j]), view(Xt, :, j)) && (t.ineligible = typemax(Int))
	end
	ridge_weights = [sum(w .* Xt[:, j] .^ 2) for j in 1:p]
	model = APLRTerm[]
	# The given terms of the interactions with each model term, and the rows and bins of each interaction (the same
	# in every step, as the package sets them up anew for each step's interactions)
	partner_givens = Dict{UInt, Vector{APLRTerm}}()
	prepared = Dict{Tuple{UInt, Int}, Tuple{TermRows, Union{Nothing, TermBins}}}()
	eta, eta_v = zeros(length(train)), zeros(length(validation))
	g = gradient(eta)
	errors = fill(Inf, settings.m)
	intercepts = zeros(settings.m)
	changes = fill((0, 0.0), settings.m)
	intercept, best_error, best_step = 0.0, Inf, 0
	for step in 1:settings.m
		# The intercept
		delta = settings.v * sum(g .* w) / sum(w)
		changed = !approx_zero(delta)
		intercept += delta
		intercepts[step] = intercept
		eta .+= delta
		eta_v .+= delta
		g = gradient(eta)
		errors[step] = validation_error(eta_v)
		null_error = sum(w .* g .^ 2)
		# The eligible terms
		best, lowest = 0, null_error
		for (k, t) in enumerate(eligible)
			t.ineligible == 0 || continue
			estimate!(t, Xt, g, w, ridge_weights[t.base], settings)
			if t.ineligible == 0 && t.error < lowest
				best, lowest = k, t.error
			end
		end
		# Interactions with the model terms of lowest search error
		if !isempty(model) && settings.max_interaction_level > 0 && settings.penalty_for_interactions < 1
			partners = [k for k in eachindex(model) if model[k].can_be_given]
			limit = settings.max_eligible_terms > 0 ? min(settings.max_eligible_terms, length(model)) : length(model)
			if limit < length(partners)
				partners = partners[sortperm([model[k].error for k in partners])][1:limit]
			end
			candidates = Tuple{APLRTerm, Tuple{UInt, Int}}[]
			for k in partners, j in 1:p
				eligible[j].ineligible == 0 || continue
				id = objectid(model[k])
				givens = get!(() -> necessary_givens(model[k], Xt), partner_givens, id)
				candidate = APLRTerm(j, [copy_term(h) for h in givens])
				# Not an interaction when it has no given term of another covariate
				all(h -> h.base == j, candidate.givens) && continue
				interaction_level(candidate) > settings.max_interaction_level && continue
				any(same_term(candidate, t) for t in model) && continue
				any(t.base == j && same_givens(candidate, t) for t in eligible) && continue
				haskey(prepared, (id, j)) && ((candidate.rows, candidate.bins) = prepared[(id, j)])
				push!(candidates, (candidate, (id, j)))
			end
			for (t, key) in candidates
				estimate!(t, Xt, g, w, ridge_weights[t.base], settings)
				haskey(prepared, key) || (prepared[key] = (t.rows, t.bins))
			end
			threshold = best == 0 ? null_error : eligible[best].error
			for (rank, k) in enumerate(sortperm([t.error for (t, _) in candidates]))
				candidates[k][1].error < threshold || break
				push!(eligible, copy_term(candidates[k][1]))
				rank == 1 && (best = length(eligible))
			end
		end
		# The best term enters the model or adds to its coefficient
		if best > 0
			t = eligible[best]
			changed |= !approx_zero(t.coefficient)
			backup = errors[step]
			eta .+= t.coefficient .* term_values(t, Xt)
			eta_v .+= t.coefficient .* term_values(t, Xv)
			g = gradient(eta)
			errors[step] = validation_error(eta_v)
			if !isfinite(errors[step])
				errors[step] = backup
				break
			end
			k = findfirst(u -> same_term(u, t), model)
			if k === nothing
				push!(model, copy_term(t))
				model[end].coefficient = 0.0
				k = length(model)
			end
			model[k].coefficient += t.coefficient
			changes[step] = (k, t.coefficient)
		end
		changed || break
		if errors[step] < best_error
			best_error, best_step = errors[step], step - 1
		elseif step - 1 > best_step + settings.early_stopping_rounds
			break
		end
		# Eligibility: the max_eligible_terms terms of lowest error stay in the search, the others leave it for a while
		if settings.ineligible_steps > 0 && settings.max_eligible_terms > 0
			count = 0
			for k in sortperm([t.error for t in eligible])
				if eligible[k].ineligible == 0
					count += 1
					count > settings.max_eligible_terms && (eligible[k].ineligible += settings.ineligible_steps)
				elseif eligible[k].ineligible < typemax(Int)
					eligible[k].ineligible -= 1
				end
			end
		end
	end
	# The model of the step with the lowest validation error
	chosen = argmin(errors)
	foreach(t -> t.coefficient = 0.0, model)
	for (k, delta) in changes[1:chosen]
		k > 0 && (model[k].coefficient += delta)
	end
	fitted = APLRModel(intercepts[chosen], filter(t -> !approx_zero(t.coefficient), model), logit, -Inf, Inf)
	predictions = transform(linear_predictor(fitted, Xt))
	APLRModel(fitted.intercept, fitted.terms, logit, max(minimum(predictions), minimum(yt)), min(maximum(predictions), maximum(yt)))
end

"""
	APLRCoding

How the columns of the data become APLR's covariates, as the package prepares them: a numeric column as it is, a
missing value (`missing` or NaN) replaced by the median of the training rows, with a column that is 1 where it was
missing when the training rows have missing values; any other column a column for each of its levels, sorted, 1 where
the row has that level (none for a level the training rows lack). Numeric columns come first.
"""
struct APLRCoding
	names::Vector{Symbol}
	medians::Dict{Symbol, Float64}
	missing_column::Dict{Symbol, Bool}
	levels::Dict{Symbol, Vector{String}}
end

level_label(v) = ismissing(v) ? "missing" : string(v)
numeric_value(v) = ismissing(v) ? NaN : Float64(v)

function median_value(values::Vector{Float64})
	isempty(values) && return 0.0
	sorted = sort(values)
	half = length(sorted) / 2
	for (i, v) in enumerate(sorted)
		approx_equal(Float64(i), half) && return i < length(sorted) ? (v + sorted[i + 1]) / 2 : v
		i > half && return v
	end
	sorted[end]
end

function aplr_coding(table, names::Vector{Symbol})
	medians, missing_column, levels = Dict{Symbol, Float64}(), Dict{Symbol, Bool}(), Dict{Symbol, Vector{String}}()
	for n in names
		values = table[n]
		if nonmissingtype(eltype(values)) <: Real
			x = numeric_value.(values)
			medians[n] = median_value(filter(!isnan, x))
			missing_column[n] = any(isnan, x)
		else
			levels[n] = sort!(unique(level_label.(values)))
		end
	end
	APLRCoding(names, medians, missing_column, levels)
end

function encode(coding::APLRCoding, table)
	columns = Vector{Float64}[]
	for n in coding.names
		haskey(coding.medians, n) || continue
		x = numeric_value.(table[n])
		push!(columns, [isnan(v) ? coding.medians[n] : v for v in x])
		coding.missing_column[n] && push!(columns, Float64.(isnan.(x)))
	end
	for n in coding.names
		haskey(coding.levels, n) || continue
		labels = level_label.(table[n])
		append!(columns, [Float64.(labels .== l) for l in coding.levels[n]])
	end
	reduce(hcat, columns)
end

matrix_table(X::AbstractMatrix) = NamedTuple{Tuple(Symbol("x", j) for j in axes(X, 2))}(Tuple(X[:, j] for j in axes(X, 2)))

"""
	APLR

A fitted APLR model: for numbers one model (`models[1]`), for classes one logit model for each class of `classes` (for
two classes one model, for the first class, the second class's being its negative), and the coding of the covariates.
"""
struct APLR
	models::Vector{APLRModel}
	classes::Union{Nothing, Vector}
	coding::APLRCoding
end

# The models of the folds combined into one, weighted by their shares of the training rows, as the package combines them
function combine(models::Vector{APLRModel}, weights::Vector{Float64})
	length(models) == 1 && return models[1]
	terms = APLRTerm[]
	for (model, weight) in zip(models, weights), t in model.terms
		u = copy_term(t)
		u.coefficient = t.coefficient * weight
		push!(terms, u)
	end
	APLRModel(sum(m.intercept * w for (m, w) in zip(models, weights)), terms, models[1].logit,
		minimum(m.low for m in models), maximum(m.high for m in models))
end

function fit_aplr(X::Matrix{Float64}, y::AbstractVector, coding::APLRCoding; classification::Bool,
	validation::Union{Nothing, AbstractVector{Bool}} = nothing, folds::Integer = 5, seed::Integer = 0, m::Integer = 3000,
	v::Real = 0.5, bins::Integer = 300, max_interaction_level::Integer = 1, min_observations_in_split::Real = 0.3,
	ineligible_boosting_steps_added::Integer = 15, max_eligible_terms::Integer = 7, early_stopping_rounds::Integer = 200,
	penalty_for_interactions::Real = 0.5, penalty_for_non_linearity::Real = 0.0, ridge_penalty::Real = 0.0001)
	n = size(X, 1)
	length(y) == n || error("X has $(n) rows and y $(length(y)) values.")
	min_observations = 0 <= min_observations_in_split < 1 ? ceil(Int, n^min_observations_in_split) : round(Int, min_observations_in_split)
	settings = APLRSettings(m, v, bins, max_interaction_level, min_observations, ineligible_boosting_steps_added, max_eligible_terms,
		early_stopping_rounds, clamp(penalty_for_interactions, 0, 1), clamp(penalty_for_non_linearity, 0, 1), max(ridge_penalty, 0.0))
	splits = if validation === nothing
		fold = rand(Random.Xoshiro(seed), 1:folds, n)
		[(findall(fold .!= k), findall(fold .== k)) for k in 1:folds]
	else
		[(findall(.!validation), findall(validation))]
	end
	weights = [length(s[1]) / sum(length(t[1]) for t in splits) for s in splits]
	fit(response, logit) = combine([boost(X, response, s[1], s[2], logit, settings) for s in splits], weights)
	if classification
		classes = sort!(unique(y))
		length(classes) >= 2 || error("Classification needs two or more classes.")
		models = length(classes) == 2 ? [fit(Float64.(y .== classes[1]), true)] : [fit(Float64.(y .== c), true) for c in classes]
		APLR(models, classes, coding)
	else
		APLR([fit(Float64.(y), false)], nothing, coding)
	end
end

"""
	APLR(X, y; classification, validation = nothing, folds = 5, seed = 0, m = 3000, v = 0.5, bins = 300,
		max_interaction_level = 1, min_observations_in_split = 0.3, ineligible_boosting_steps_added = 15,
		max_eligible_terms = 7, early_stopping_rounds = 200, penalty_for_interactions = 0.5, penalty_for_non_linearity = 0.0,
		ridge_penalty = 0.0001)

Fits automatic piecewise linear regression (von Ottenbreit and De Bin, 2024) with the defaults of the package aplr 10.26
to the covariates `X` (a row for each observation; a missing value NaN, replaced by the median) and the response `y`,
numbers or, with `classification`, classes. The rows of `validation` (a vector of Bool), when given, choose the number
of boosting steps of a model fitted to the other rows; else the rows are split at random into `folds` folds, a model is
fitted with each fold as its validation rows, and the models are combined, weighted by their numbers of training rows.
`min_observations_in_split` below 1 is a power of the number of rows. `predict(model, X)` gives the probability of each
class (a column for each, in sorted order), or the predicted number, kept within the range of the training predictions
and responses.

```julia
import ModelZoo
x = collect(range(0, 10; length = 200))
model = ModelZoo.APLR(reshape(x, :, 1), sin.(x); classification = false)
ModelZoo.predict(model, reshape([2.5, 7.5], :, 1))
```
"""
function APLR(X::AbstractMatrix{<:Real}, y::AbstractVector; classification::Bool = !(eltype(y) <: AbstractFloat), kwargs...)
	table = matrix_table(X)
	coding = aplr_coding(table, collect(keys(table)))
	fit_aplr(encode(coding, table), y, coding; classification, kwargs...)
end

function predict_encoded(model::APLR, X::Matrix{Float64})
	if model.classes === nothing
		return clamp.(linear_predictor(model.models[1], X), model.models[1].low, model.models[1].high)
	end
	if length(model.classes) == 2
		eta = linear_predictor(model.models[1], X)
		P = [logistic_clipped.(eta) logistic_clipped.(-eta)]
	else
		P = reduce(hcat, [logistic_clipped.(linear_predictor(m, X)) for m in model.models])
	end
	P ./ sum(P; dims = 2)
end

"""
	BaggedAPLR

APLR models fitted to the same rows, each with one fold of them as its validation rows, whose predictions are averaged.
"""
struct BaggedAPLR
	models::Vector{APLR}
	classes::Union{Nothing, Vector}
	coding::APLRCoding
end

function fit_bagged_aplr(X::Matrix{Float64}, y::AbstractVector, coding::APLRCoding; folds::Integer = 8, seed::Integer = 0,
	classification::Bool, kwargs...)
	fold = bag_folds(y, folds, seed; classification)
	models = Vector{APLR}(undef, folds)
	Threads.@threads for k in 1:folds
		models[k] = fit_aplr(X, y, coding; classification, validation = fold .== k, kwargs...)
	end
	BaggedAPLR(models, models[1].classes, coding)
end

"""
	BaggedAPLR(X, y; folds = 8, seed = 0, classification, kwargs...)

Splits the rows into `folds` folds at random, stratified by class for classes (as AutoGluon's bags split them), fits an
`APLR` model (with `kwargs`) to the rows with each fold as its validation rows, and averages their predictions:
TabArena's APLR (default), which AutoGluon fits as a bag of 8 models. The models are fitted on Julia's threads.
"""
function BaggedAPLR(X::AbstractMatrix{<:Real}, y::AbstractVector; classification::Bool = !(eltype(y) <: AbstractFloat), kwargs...)
	table = matrix_table(X)
	coding = aplr_coding(table, collect(keys(table)))
	fit_bagged_aplr(encode(coding, table), y, coding; classification, kwargs...)
end

predict_encoded(model::BaggedAPLR, X::Matrix{Float64}) = sum(predict_encoded(m, X) for m in model.models) ./ length(model.models)

predict(model::Union{APLR, BaggedAPLR}, X::AbstractMatrix{<:Real}) = predict_encoded(model, encode(model.coding, matrix_table(X)))

"""
	APLR(train, target; kwargs...)
	BaggedAPLR(train, target; kwargs...)

Fits to a table of columns (a NamedTuple of vectors) whose column `target` is the response and whose other columns are
the covariates, coded as `APLRCoding` says. `predict(model, test)` takes a table with the same columns.

```julia
function predict(train, test, target)
	model = ModelZoo.BaggedAPLR(train, target)
	ModelZoo.predict(model, test)
end
```
"""
function APLR(train::NamedTuple, target; classification::Bool = !(eltype(train[Symbol(target)]) <: AbstractFloat), kwargs...)
	X, y, coding = aplr_table(train, target)
	fit_aplr(X, y, coding; classification, kwargs...)
end

function BaggedAPLR(train::NamedTuple, target; classification::Bool = !(eltype(train[Symbol(target)]) <: AbstractFloat), kwargs...)
	X, y, coding = aplr_table(train, target)
	fit_bagged_aplr(X, y, coding; classification, kwargs...)
end

function aplr_table(train::NamedTuple, target)
	names = [n for n in keys(train) if String(n) != String(target)]
	y = collect(skipmissing(train[Symbol(target)]))
	length(y) == length(train[Symbol(target)]) || error("The response $(target) has missing values.")
	coding = aplr_coding(train, names)
	encode(coding, train), y, coding
end

predict(model::Union{APLR, BaggedAPLR}, table::NamedTuple) = predict_encoded(model, encode(model.coding, table))
