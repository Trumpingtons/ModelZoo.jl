# Cox's proportional hazards model with smooth terms, a generalized additive model (GAM) of the log hazard: each
# covariate enters linearly or as a penalized thin plate regression spline (Wood, 2003), the coefficients maximize the
# partial likelihood with Peto's (Breslow's) handling of ties, penalized by the wiggliness of the splines, and the
# smoothing parameters minimize the Laplace approximate restricted marginal likelihood (REML) of Wood, Pya and Säfken
# (2016), by Newton's method (Wood, 2011). The survival of new rows is the Breslow estimate of the baseline cumulative
# hazard, scaled by their relative risk.

"""
	ThinPlateSpline

A thin plate regression spline of one covariate (Wood, 2003), with penalty order 2: the basis of the full thin plate
spline on the distinct values of the covariate, `η(r) = r^3 / 12`, truncated to the `k` eigenvectors of its matrix
with the largest eigenvalues in absolute value and constrained to be orthogonal to the polynomials of degree 1, then
those two polynomials, the penalty null space. `knots` are the distinct values, less `shift`, their mean; `UZ` maps
the truncated parameters to those of the full spline; `penalty` is the wiggliness penalty of the `k - 2` truncated
parameters.
"""
struct ThinPlateSpline
	knots::Vector{Float64}
	shift::Float64
	UZ::Matrix{Float64}
	penalty::Matrix{Float64}
end

"""
	ThinPlateSpline(x; k = 10, max_knots = 2000)

The thin plate regression spline of the covariate `x` of basis dimension `k`. With more than `max_knots` distinct
values, the basis is built on `max_knots` of them spread evenly over their order.
"""
function ThinPlateSpline(x::AbstractVector{<:Real}; k::Integer = 10, max_knots::Integer = 2000)
	distinct = sort!(unique(Float64.(x)))
	length(distinct) >= k || error("A thin plate regression spline of dimension $(k) needs $(k) distinct values, not $(length(distinct)).")
	if length(distinct) > max_knots
		distinct = distinct[round.(Int, range(1, length(distinct); length = max_knots))]
	end
	shift = sum(distinct) / length(distinct)
	knots = distinct .- shift
	F = LinearAlgebra.eigen(LinearAlgebra.Symmetric([abs(a - b)^3 / 12 for a in knots, b in knots]))
	largest = sortperm(abs.(F.values); rev = true)[1:k]
	U, D = F.vectors[:, largest], F.values[largest]
	# The parameters δ = U δk of the truncated spline with T' δ = 0, T the polynomials of degree 1 at the knots: δk in
	# the null space of T' U
	constraint = transpose([ones(length(knots)) knots]) * U
	Z = (LinearAlgebra.qr(transpose(constraint)).Q * Matrix{Float64}(LinearAlgebra.I, k, k))[:, 3:k]
	penalty = transpose(Z) * LinearAlgebra.Diagonal(D) * Z
	ThinPlateSpline(knots, shift, U * Z, (penalty + transpose(penalty)) / 2)
end

"""
	basis(spline, x)

The `k` columns of the spline at the values `x`: the truncated spline's, then 1 and `x`, less the spline's shift.
"""
function basis(spline::ThinPlateSpline, x::AbstractVector{<:Real})
	z = Float64.(x) .- spline.shift
	B = Matrix{Float64}(undef, length(z), size(spline.UZ, 2) + 2)
	# In blocks of rows, so that the distances to the knots of many rows are never held at once
	for rows in Iterators.partition(eachindex(z), 1000)
		B[rows, 1:(end - 2)] = (abs.(z[rows] .- transpose(spline.knots)) .^ 3 ./ 12) * spline.UZ
	end
	B[:, end - 1] .= 1
	B[:, end] = z
	B
end

"""
	SmoothTerm

A covariate as a smooth term: its spline, the reparameterization `Z` that makes its values sum to zero over the
training rows (identifiable beside the other terms), the scale of each column, and its penalty in those columns.
"""
struct SmoothTerm
	column::Int
	spline::ThinPlateSpline
	Z::Matrix{Float64}
	scale::Vector{Float64}
	penalty::Matrix{Float64}
	rank::Int
end

function SmoothTerm(column::Integer, x::AbstractVector{<:Real}; k::Integer = 10, max_knots::Integer = 2000)
	spline = ThinPlateSpline(x; k, max_knots)
	B = basis(spline, x)
	sums = vec(sum(B; dims = 1))
	Z = (LinearAlgebra.qr(reshape(sums, :, 1)).Q * Matrix{Float64}(LinearAlgebra.I, k, k))[:, 2:k]
	columns = B * Z
	scale = vec(sqrt.(sum(abs2, columns; dims = 1) ./ size(columns, 1)))
	full = zeros(k, k)
	full[1:(k - 2), 1:(k - 2)] = spline.penalty
	penalty = transpose(Z) * full * Z ./ (scale * transpose(scale))
	SmoothTerm(column, spline, Z, scale, (penalty + transpose(penalty)) / 2, k - 2)
end

columns(term::SmoothTerm, x::AbstractVector{<:Real}) = (basis(term.spline, x) * term.Z) ./ transpose(term.scale)

"""
	CoxGAM

A fitted Cox model with smooth terms: the covariates entering linearly (`linear`, columns of the training matrix) and
as smooth terms (`smooths`), the coefficients, the smoothing parameter of each smooth term, and the training times,
events and linear predictors, from which the Breslow estimate of the baseline hazard comes.
"""
struct CoxGAM
	linear::Vector{Int}
	smooths::Vector{SmoothTerm}
	coefficients::Vector{Float64}
	smoothing::Vector{Float64}
	train_time::Vector{Float64}
	train_event::Vector{Int}
	train_lp::Vector{Float64}
	coding::Any
end

function design(linear::AbstractVector{<:Integer}, smooths::AbstractVector{SmoothTerm}, X::AbstractMatrix{<:Real})
	reduce(hcat, [[Float64.(X[:, linear])]; [columns(term, view(X, :, term.column)) for term in smooths]])
end

"""
The partial likelihood of Cox's model with Breslow's handling of ties at the linear predictor `eta`, and its gradient
and information matrix in the coefficients of the columns of `X`. `order` sorts the rows by decreasing time and
`starts` gives the first row, in that order, of each run of equal times. Also returned, for the derivatives of the
information: the relative risks `w` (scaled), `c` (the sum of d/W over the event times up to each row's), and, for each
event time, its number of events `d`, its total relative risk at risk `W`, and the mean covariates at risk `m`.
"""
function partial_likelihood(X, time, event, eta, order, starts)
	n, p = size(X)
	top = maximum(eta)
	w = exp.(eta .- top)
	W, Sx = 0.0, zeros(p)
	d, Ws, means, groups = Int[], Float64[], Vector{Float64}[], UnitRange{Int}[]
	loglik = 0.0
	for (g, first) in enumerate(starts)
		last = g < length(starts) ? starts[g + 1] - 1 : n
		events = 0
		for r in first:last
			i = order[r]
			W += w[i]
			Sx .+= w[i] .* view(X, i, :)
			if event[i] == 1
				events += 1
				loglik += eta[i]
			end
		end
		if events > 0
			loglik -= events * (log(W) + top)
			push!(d, events); push!(Ws, W); push!(means, Sx ./ W); push!(groups, first:last)
		end
	end
	# c: for each row, the sum of d/W over the event times at or before its time (the times at which it is at risk)
	c = zeros(n)
	running = 0.0
	# The event times in increasing order are the groups in reverse; a row's time is that of its run of equal times
	cumulative = zeros(length(starts))
	k = length(d)
	for (h, first) in Iterators.reverse(enumerate(starts))
		if k >= 1 && first == groups[k].start
			running += d[k] / Ws[k]
			k -= 1
		end
		cumulative[h] = running
	end
	for (h, first) in enumerate(starts)
		last = h < length(starts) ? starts[h + 1] - 1 : n
		for r in first:last
			c[order[r]] = cumulative[h]
		end
	end
	M = isempty(means) ? zeros(0, p) : reduce(vcat, transpose.(means))
	gradient = transpose(X) * (Float64.(event .== 1) .- w .* c)
	information = transpose(X) * (X .* (w .* c)) .- transpose(M) * (M .* d)
	(; loglik, gradient, information = (information + transpose(information)) / 2, w, c, d, W = Ws, M, groups)
end

"""
The coefficients that maximize the partial log-likelihood less half the penalty `S`, by Newton's method from `beta`,
halving a step that does not increase it; with the partial likelihood at them.
"""
function penalized_fit(X, time, event, S, beta, order, starts; tolerance = 1e-10, iterations = 200)
	objective(f, b) = f.loglik - transpose(b) * S * b / 2
	fit = partial_likelihood(X, time, event, X * beta, order, starts)
	for _ in 1:iterations
		score = fit.gradient .- S * beta
		A = LinearAlgebra.Symmetric(fit.information .+ S)
		step = LinearAlgebra.cholesky(A; check = false)
		direction = LinearAlgebra.issuccess(step) ? step \ score : LinearAlgebra.pinv(Matrix(A)) * score
		current = objective(fit, beta)
		fraction = 1.0
		local trial, candidate
		for _ in 1:40
			candidate = beta .+ fraction .* direction
			trial = partial_likelihood(X, time, event, X * candidate, order, starts)
			objective(trial, candidate) >= current - 1e-12 * abs(current) && break
			fraction /= 2
		end
		beta, fit = candidate, trial
		maximum(abs, fraction .* direction; init = 0.0) <= tolerance * (1 + maximum(abs, beta; init = 0.0)) && break
	end
	beta, fit
end

"""
The REML criterion of the log smoothing parameters `rho` (Wood, Pya and Säfken, 2016), up to a constant, and its
gradient: minus the partial log-likelihood at the penalized fit, plus half the penalty, plus half the log determinant of
the penalized information, less half the log pseudo-determinant of the penalty.
"""
function reml(rho, X, time, event, penalties, blocks, ranks, beta, order, starts)
	p = size(X, 2)
	lambda = exp.(rho)
	S = zeros(p, p)
	for (j, block) in enumerate(blocks)
		S[block, block] .+= lambda[j] .* penalties[j]
	end
	beta, fit = penalized_fit(X, time, event, S, beta, order, starts)
	A = LinearAlgebra.Symmetric(fit.information .+ S)
	F = LinearAlgebra.cholesky(A; check = false)
	LinearAlgebra.issuccess(F) || return Inf, fill(NaN, length(rho)), beta
	value = -fit.loglik + transpose(beta) * S * beta / 2 + LinearAlgebra.logdet(F) / 2 - sum(ranks .* rho) / 2
	# The derivatives: the coefficients move with rho_j by v_j = -A⁻¹ λ_j S_j β, and the information with them
	Ainv = inv(F)
	q = vec(sum((X * Ainv) .* X; dims = 2))
	U = fit.M * Ainv
	gradient = zeros(length(rho))
	for (j, block) in enumerate(blocks)
		Sjb = zeros(p)
		Sjb[block] = lambda[j] .* (penalties[j] * beta[block])
		v = -(Ainv * Sjb)
		a = X * v
		trace = derivative_trace(fit, X, a, q, U, order, starts)
		gradient[j] = (transpose(beta) * Sjb + lambda[j] * sum(Ainv[block, block] .* penalties[j]) + trace - ranks[j]) / 2
	end
	value, gradient, beta
end

"""
tr(A⁻¹ dH), dH the derivative of the information as the linear predictor moves by `a`: the relative risks move by
w a, so W by Wa = Σ w a over the rows at risk, c by -Σ d Wa / W², and the means at risk m by (Σ w a x - m Wa) / W.
`q` holds x' A⁻¹ x of each row and `U` the rows A⁻¹ m of the event times.
"""
function derivative_trace(fit, X, a, q, U, order, starts)
	n, p = size(X)
	wa = fit.w .* a
	# Over the event times in decreasing time order, as partial_likelihood found them: Wa and Σ w a x at risk
	Wa = zeros(length(fit.d))
	Sxa = zeros(p)
	cross = zeros(length(fit.d))
	total, k = 0.0, 1
	for (g, first) in enumerate(starts)
		last = g < length(starts) ? starts[g + 1] - 1 : n
		for r in first:last
			i = order[r]
			total += wa[i]
			Sxa .+= wa[i] .* view(X, i, :)
		end
		if k <= length(fit.d) && fit.groups[k].start == first
			Wa[k] = total
			cross[k] = LinearAlgebra.dot(Sxa, view(U, k, :))
			k += 1
		end
	end
	# dc of each row: minus the sum of d Wa / W² over the event times at or before its time
	dcum = zeros(length(starts))
	running, k = 0.0, length(fit.d)
	for (h, first) in Iterators.reverse(enumerate(starts))
		if k >= 1 && fit.groups[k].start == first
			running += fit.d[k] * Wa[k] / fit.W[k]^2
			k -= 1
		end
		dcum[h] = running
	end
	dc = zeros(n)
	for (h, first) in enumerate(starts)
		last = h < length(starts) ? starts[h + 1] - 1 : n
		for r in first:last
			dc[order[r]] = -dcum[h]
		end
	end
	first_term = sum((wa .* fit.c .+ fit.w .* dc) .* q)
	# The means' term: -Σ d (dm' A⁻¹ m + m' A⁻¹ dm) = -2 Σ d dm' u, with dm' u = (Σ w a x)' u / W - Wa m' u / W
	mu = vec(sum(fit.M .* U; dims = 2))
	second_term = -2 * sum(fit.d .* (cross .- Wa .* mu) ./ fit.W)
	first_term + second_term
end

"""
	CoxGAM(time, event, X; smooth = auto, k = 10, max_knots = 2000)

Fits Cox's model with smooth terms to the rows of the covariate matrix `X`, with their times `time` and event
indicators `event` (1 for an event, 0 for a censored time). The columns in `smooth` enter as thin plate regression
splines of basis dimension `k`, each with its own smoothing parameter, and the others linearly; by default, the columns
with 10 or more distinct values are smooth. The coefficients maximize the partial likelihood with Breslow's handling
of ties, penalized by the splines' wiggliness, and the smoothing parameters minimize the REML criterion. The criterion
can have more than one local minimum: the search starts from three points and keeps the lowest minimum, where mgcv
searches from one start, so the two can end at different minima.

```julia
import ModelZoo
x = collect(1.0:40.0)
time, event = 50 .- x .+ 10 .* sin.(x ./ 4), [i % 4 == 0 ? 0 : 1 for i in 1:40]
model = ModelZoo.CoxGAM(time, event, reshape(x, :, 1))
ModelZoo.survival(model, [5.0 20.0]', [10.0, 30.0])
```
"""
function CoxGAM(time::AbstractVector{<:Real}, event::AbstractVector{<:Integer}, X::AbstractMatrix{<:Real};
	smooth::AbstractVector{<:Integer} = [j for j in axes(X, 2) if length(unique(view(X, :, j))) >= 10], k::Integer = 10,
	max_knots::Integer = 2000, coding = nothing)
	time, event = Float64.(time), Int.(event)
	smooths = [SmoothTerm(j, view(X, :, j); k, max_knots) for j in smooth]
	linear = [j for j in axes(X, 2) if !(j in smooth)]
	D = design(linear, smooths, X)
	order = sortperm(time; rev = true)
	sorted = time[order]
	starts = [r for r in eachindex(sorted) if r == 1 || sorted[r] != sorted[r - 1]]
	blocks, offset = UnitRange{Int}[], length(linear)
	for term in smooths
		push!(blocks, (offset + 1):(offset + size(term.penalty, 1)))
		offset += size(term.penalty, 1)
	end
	penalties, ranks = [term.penalty for term in smooths], [term.rank for term in smooths]
	beta = zeros(size(D, 2))
	if isempty(smooths)
		beta, _ = penalized_fit(D, time, event, zeros(length(beta), length(beta)), beta, order, starts)
		lambda = Float64[]
	else
		# The scale of the smoothing parameters: the information at zero over the penalty, block by block. The REML
		# criterion can have more than one local minimum, so the search starts there and at 5 below and above it on the
		# log scale, and the lowest minimum is kept. Each log smoothing parameter stays within 40 of that scale, beyond
		# which a smooth term is a straight line or interpolates.
		H0 = partial_likelihood(D, time, event, zeros(size(D, 1)), order, starts).information
		rho = [log(max(LinearAlgebra.tr(H0[b, b]), eps()) / LinearAlgebra.tr(P)) for (b, P) in zip(blocks, penalties)]
		searches = [minimize_reml(rho .+ shift, D, time, event, penalties, blocks, ranks, beta, order, starts, rho .- 40, rho .+ 40) for shift in (0, -5, 5)]
		best = searches[argmin([search[3] for search in searches])]
		beta, lambda = best[2], exp.(best[1])
	end
	CoxGAM(linear, smooths, beta, lambda, time, event, D * beta, coding)
end

"""
The log smoothing parameters that minimize the REML criterion, by Newton's method from `rho` (Wood, 2011): the Hessian
from forward differences of the analytic gradient, its eigenvalues replaced by their absolute values (kept away from 0)
so that each step goes downhill, steps of at most 5 in each parameter, halved until the criterion falls. Each parameter
is kept between `low` and `high`. Returns the parameters, the coefficients and the criterion.
"""
function minimize_reml(rho, D, time, event, penalties, blocks, ranks, beta, order, starts, low, high; iterations = 200, tolerance = 1e-8)
	rho = clamp.(rho, low, high)
	value, gradient, beta = reml(rho, D, time, event, penalties, blocks, ranks, beta, order, starts)
	# Where the penalized information is not positive definite, there is no search from this start
	isfinite(value) || return rho, beta, Inf
	m = length(rho)
	for _ in 1:iterations
		# Parameters at a bound with the gradient pushing them out stay there
		free = [!((rho[j] <= low[j] && gradient[j] > 0) || (rho[j] >= high[j] && gradient[j] < 0)) for j in eachindex(rho)]
		maximum(abs, gradient .* free) <= tolerance * (1 + abs(value)) && break
		# A column of the Hessian by the forward difference, else the backward one, else none (the criterion is not
		# defined on either side)
		H = zeros(m, m)
		for j in 1:m
			for h in (1e-4, -1e-4)
				shifted = copy(rho)
				shifted[j] += h
				column = (reml(shifted, D, time, event, penalties, blocks, ranks, beta, order, starts)[2] .- gradient) ./ h
				if all(isfinite, column)
					H[:, j] = column
					break
				end
			end
		end
		F = LinearAlgebra.eigen(LinearAlgebra.Symmetric((H + transpose(H)) / 2))
		values = max.(abs.(F.values), 1e-7 * max(1, maximum(abs, F.values)))
		local candidate, trial
		accepted = false
		# The Newton step, then, should it fail to go downhill, the steepest descent step
		for direction in (-(F.vectors * ((transpose(F.vectors) * (gradient .* free)) ./ values)), -gradient)
			step = direction .* free
			step .*= min(1, 5 / max(maximum(abs, step), eps()))
			fraction = 1.0
			for _ in 1:30
				candidate = clamp.(rho .+ fraction .* step, low, high)
				trial = reml(candidate, D, time, event, penalties, blocks, ranks, beta, order, starts)
				if isfinite(trial[1]) && trial[1] <= value + 1e-4 * fraction * LinearAlgebra.dot(gradient, step)
					accepted = true
					break
				end
				fraction /= 2
			end
			accepted && break
		end
		accepted || break
		moved = maximum(abs, candidate .- rho)
		rho, value, gradient, beta = candidate, trial[1], trial[2], trial[3]
		moved <= 1e-10 && break
	end
	rho, beta, value
end

"""
	linear_predictor(model, X)

The linear predictor (log relative risk) of the rows of `X`, with the columns of the training matrix.
"""
linear_predictor(model::CoxGAM, X::AbstractMatrix{<:Real}) = design(model.linear, model.smooths, X) * model.coefficients

"""
	survival(model, X, times)

The survival of each row of `X` past each of `times`, a row for each row of `X`: the Breslow estimate of the
cumulative baseline hazard of the training rows at each time, whose times are first tied (`tie_times`), times each
row's relative risk.
"""
function survival(model::CoxGAM, X::AbstractMatrix{<:Real}, times::AbstractVector{<:Real})
	breslow_survival(model.train_time, model.train_event, model.train_lp, linear_predictor(model, X), times)
end

function breslow_survival(train_time, train_event, train_lp, lp, times)
	t = tie_times(train_time)
	top = maximum(train_lp)
	risk = exp.(train_lp .- top)
	distinct = sort!(unique(t))
	leaving, events = zeros(length(distinct)), zeros(length(distinct))
	for i in eachindex(t)
		k = searchsortedfirst(distinct, t[i])
		leaving[k] += risk[i]
		events[k] += train_event[i] == 1
	end
	at_risk = reverse(cumsum(reverse(leaving)))
	hazard = cumsum(events ./ at_risk)
	baseline = [(k = searchsortedlast(distinct, x); k == 0 ? 0.0 : hazard[k]) for x in times]
	exp.(-(exp.(lp .- top)) * transpose(baseline))
end

# Tables: the numeric columns as they are, the others as factors, a column for each level after the first in sorted
# order, a level the training rows lack counting as the first.

struct TableCoding
	names::Vector{Symbol}
	levels::Dict{Symbol, Vector{Any}}
end

function encode(coding::TableCoding, table)
	columns = Vector{Float64}[]
	for name in coding.names
		values = table[name]
		if haskey(coding.levels, name)
			for level in coding.levels[name][2:end]
				push!(columns, [isequal(v, level) ? 1.0 : 0.0 for v in values])
			end
		else
			push!(columns, Float64.(values))
		end
	end
	reduce(hcat, columns)
end

"""
	CoxGAM(train, time, event; k = 10, max_knots = 2000)

Fits Cox's model with smooth terms to a table of columns (a NamedTuple of vectors) holding the covariates, the times
(the column named `time`) and the event indicators (`event`). Numeric columns with 10 or more distinct values are
smooth, other numeric columns enter linearly, and the other columns are factors, coded by a column for each level after
the first; constant columns are left out.

```julia
function predict(train, test, time, event, times)
	model = ModelZoo.CoxGAM(train, time, event)
	ModelZoo.survival(model, test, times)
end
```
"""
function CoxGAM(train::NamedTuple, time, event; k::Integer = 10, max_knots::Integer = 2000)
	names = [n for n in keys(train) if !(String(n) in (String(time), String(event))) && length(unique(train[n])) > 1]
	levels = Dict{Symbol, Vector{Any}}(n => sort!(unique(train[n])) for n in names if !(eltype(train[n]) <: Real))
	coding = TableCoding(names, levels)
	X = encode(coding, train)
	smooth = Int[]
	j = 0
	for n in names
		if haskey(levels, n)
			j += length(levels[n]) - 1
		else
			j += 1
			length(unique(train[n])) >= 10 && push!(smooth, j)
		end
	end
	CoxGAM(Float64.(train[Symbol(time)]), Int.(train[Symbol(event)]), X; smooth, k, max_knots, coding)
end

linear_predictor(model::CoxGAM, table::NamedTuple) = linear_predictor(model, encode(model.coding, table))
survival(model::CoxGAM, table::NamedTuple, times::AbstractVector{<:Real}) = survival(model, encode(model.coding, table), times)
