# The flexible parametric survival model of Royston and Parmar (2002) on the proportional hazards scale: the log
# cumulative hazard is a natural cubic spline of log time plus a linear function of the covariates, its coefficients
# maximizing the full likelihood of right-censored times. The knots are those flexsurv's flexsurvspline places by
# default: boundary knots at the smallest and largest log event times, and `k` internal knots at equally spaced
# quantiles of the log event times.

"""
	FlexibleSurvival

A fitted flexible parametric survival model: `knots`, the boundary and internal knots on the axis of log time, in
increasing order; `gamma`, the coefficients of the spline (the constant, log time, then a basis function for each
internal knot); `beta`, the effects of the covariates; `loglik`, the maximized log likelihood.
"""
struct FlexibleSurvival
	knots::Vector{Float64}
	gamma::Vector{Float64}
	beta::Vector{Float64}
	loglik::Float64
end

"""
	quantile7(sorted, p)

The `p` quantile of the increasing values `sorted`, by linear interpolation between order statistics (Hyndman and Fan's
definition 7, R's default).
"""
function quantile7(sorted::AbstractVector{Float64}, p::Real)
	h = (length(sorted) - 1) * p + 1
	low = floor(Int, h)
	low >= length(sorted) ? sorted[end] : sorted[low] + (h - low) * (sorted[low + 1] - sorted[low])
end

"""
	spline_basis(knots, x)

The columns of the natural cubic spline of Royston and Parmar (2002) at the log times `x`, and their derivatives with
respect to log time: the constant, `x`, and for each internal knot `kj` the function
`(x - kj)^3+ - l (x - kmin)^3+ - (1 - l) (x - kmax)^3+`, with `l = (kmax - kj) / (kmax - kmin)`, linear beyond the
boundary knots `kmin` and `kmax`.
"""
function spline_basis(knots::AbstractVector{Float64}, x::AbstractVector{<:Real})
	low, high = knots[1], knots[end]
	inner = knots[2:(end - 1)]
	B, dB = zeros(length(x), length(inner) + 2), zeros(length(x), length(inner) + 2)
	B[:, 1] .= 1
	B[:, 2] .= x
	dB[:, 2] .= 1
	cube(v) = v > 0 ? v^3 : 0.0
	square(v) = v > 0 ? v^2 : 0.0
	for (j, knot) in enumerate(inner)
		l = (high - knot) / (high - low)
		B[:, j + 2] .= cube.(x .- knot) .- l .* cube.(x .- low) .- (1 - l) .* cube.(x .- high)
		dB[:, j + 2] .= 3 .* (square.(x .- knot) .- l .* square.(x .- low) .- (1 - l) .* square.(x .- high))
	end
	B, dB
end

"""
	FlexibleSurvival(time, event, X; k = 1, tolerance = 1e-10, iterations = 200)

Fits the flexible parametric survival model of Royston and Parmar (2002) with `k` internal knots to the times `time`
(positive), the event indicators `event` (1 for an event, 0 for a censored time) and the covariates `X` (a row for each
time). The log cumulative hazard of a row with covariates `z` at time `t` is `s(log t) + z' beta`, `s` the natural
cubic spline of `spline_basis` with the knots flexsurv places by default (boundary knots at the extreme log event
times, internal knots at the quantiles `1 / (k + 1), ..., k / (k + 1)` of the log event times), so that `k = 0` is the
Weibull model. The log likelihood is concave in the coefficients, so its maximum is unique; Newton's method finds it,
each step halved until the likelihood rises and the hazard stays positive at every event time, and stops when the
increase it expects falls below `tolerance`.

```julia
import ModelZoo
x = collect(1.0:40.0)
time, event = 50 .- x .+ 10 .* sin.(x ./ 4), [i % 4 == 0 ? 0 : 1 for i in 1:40]
model = ModelZoo.FlexibleSurvival(time, event, reshape(x, :, 1); k = 2)
ModelZoo.survival(model, reshape([5.0, 20.0], :, 1), [10.0, 30.0])
```
"""
function FlexibleSurvival(time::AbstractVector{<:Real}, event::AbstractVector{<:Integer}, X::AbstractMatrix{<:Real};
	k::Integer = 1, tolerance::Real = 1e-10, iterations::Integer = 200)
	n = length(time)
	size(X, 1) == n == length(event) || error("The times, the event indicators and the rows of the covariates differ in number.")
	all(>(0), time) || error("The times must be positive.")
	x = log.(Float64.(time))
	died = event .== 1
	deaths = sort(x[died])
	length(unique(deaths)) >= 2 || error("The model needs two or more distinct event times.")
	knots = [deaths[1]; [quantile7(deaths, j / (k + 1)) for j in 1:k]; deaths[end]]
	B, dB = spline_basis(knots, x)
	W = [B Float64.(X)]
	D = [dB zeros(n, size(X, 2))][died, :]
	m = size(W, 2)
	# The log likelihood, its gradient and its Hessian at θ; -Inf where the hazard is not positive at an event time
	function loglik(θ)
		slope = D * θ
		any(<=(0), slope) && return -Inf
		η = W * θ
		sum(log.(slope)) + sum(η[died]) - sum(x[died]) - sum(exp.(η))
	end
	# The start: the exponential model with the rate of events per unit of time, whose slope is 1 everywhere
	θ = zeros(m)
	θ[1], θ[2] = log(count(died) / sum(Float64.(time))), 1.0
	value = loglik(θ)
	for _ in 1:iterations
		slope, risk = D * θ, exp.(W * θ)
		gradient = transpose(D) * (1 ./ slope) + transpose(W[died, :]) * ones(count(died)) - transpose(W) * risk
		information = transpose(D) * (D ./ slope .^ 2) + transpose(W) * (W .* risk)
		F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(information); check = false)
		step = LinearAlgebra.issuccess(F) ? F \ gradient : LinearAlgebra.pinv(information) * gradient
		expected = LinearAlgebra.dot(gradient, step)
		expected / 2 < tolerance && break
		t = 1.0
		while t > 1e-12
			candidate = loglik(θ .+ t .* step)
			if candidate >= value
				θ, value = θ .+ t .* step, candidate
				break
			end
			t /= 2
		end
		t > 1e-12 || break
	end
	FlexibleSurvival(knots, θ[1:(k + 2)], θ[(k + 3):end], value)
end

"""
	linear_predictor(model::FlexibleSurvival, X)

The linear predictor of the rows of `X`: the spline's constant plus the covariates' effects, the log cumulative hazard
less the spline of log time.
"""
linear_predictor(model::FlexibleSurvival, X::AbstractMatrix{<:Real}) = model.gamma[1] .+ X * model.beta

"""
	survival(model::FlexibleSurvival, X, times)

The survival of each row of `X` past each of `times` (positive), a row for each row of `X`:
`exp(-exp(s(log t) + z' beta))`.
"""
function survival(model::FlexibleSurvival, X::AbstractMatrix{<:Real}, times::AbstractVector{<:Real})
	B, _ = spline_basis(model.knots, log.(Float64.(times)))
	exp.(-exp.(transpose(B * model.gamma) .+ X * model.beta))
end
