using Test
import LinearAlgebra, Survival, TOML
import ModelZoo as MZ
import PollisCompetitions as PC

# The covariates of one split of a survival dataset as the Low-Dimensional Survival Benchmark prepares them (mlr3pipelines
# 0.7.1 in its bl(): fixfactors, imputesample of factors, collapsefactors at 5% and 5 levels, removeconstants), the
# factors of info.toml coded as R's model.matrix codes them (a column per level after the first). The numeric columns
# with `smooth_from` or more distinct training values are smooth terms, after the others; of the others, those coxph
# keeps as linearly independent, with the intercept, in order. Also returned: whether every test row's levels are among
# the training rows' (the benchmark imputes a level they lack at random).
label(x::AbstractString) = String(x)
label(x::Real) = isinteger(x) ? string(Int(x)) : string(x)
# collapsefactors: with more than 5 levels, the levels of 5% or more of the training rows and the 5 most frequent are
# kept, and the others merge into the last level kept
function collapse(levels, counts)
	length(levels) <= 5 && return Dict(l => l for l in levels)
	order = sortperm(counts; rev = true, alg = MergeSort)
	keep = unique([[levels[k] for k in order if counts[k] / sum(counts) >= 0.05]; levels[order[1:5]]])
	Dict(l => (l in keep ? l : keep[end]) for l in levels)
end
function covariates(data, fields, train, test; smooth_from = typemax(Int))
	factors = get(fields, "factors", Dict{String, Any}())
	features = [String(n) for n in keys(data) if !(String(n) in (fields["target"], fields["event"]))]
	comparable = true
	linear, linear_test, smooth, smooth_test = Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]
	for n in features
		values = getproperty(data, Symbol(n))
		if haskey(factors, n)
			x = label.(values)
			present = [l for l in factors[n] if any(==(l), x[train])]
			all(v -> v in present, x[test]) || (comparable = false)
			merging = collapse(present, [count(==(l), x[train]) for l in present])
			for l in unique(merging[l] for l in present)[2:end]
				push!(linear, [merging[v] == l ? 1.0 : 0.0 for v in x[train]])
				push!(linear_test, [get(merging, v, "") == l ? 1.0 : 0.0 for v in x[test]])
			end
		elseif length(unique(values[train])) >= smooth_from
			push!(smooth, Float64.(values[train]))
			push!(smooth_test, Float64.(values[test]))
		else
			push!(linear, Float64.(values[train]))
			push!(linear_test, Float64.(values[test]))
		end
	end
	# removeconstants; then the columns coxph keeps, in order, by a pivoted QR with the intercept
	kept = [k for k in eachindex(linear) if length(unique(linear[k])) > 1]
	linear, linear_test = linear[kept], linear_test[kept]
	if !isempty(linear)
		F = LinearAlgebra.qr([ones(length(train)) reduce(hcat, linear)], LinearAlgebra.ColumnNorm())
		independent = sort(filter(>(1), F.p[1:LinearAlgebra.rank(F.R; rtol = 1e-12)])) .- 1
		linear, linear_test = linear[independent], linear_test[independent]
	end
	X = reduce(hcat, [linear; smooth]; init = zeros(length(train), 0))
	Xt = reduce(hcat, [linear_test; smooth_test]; init = zeros(length(test), 0))
	(; X, Xt, smooth = collect(length(linear) .+ eachindex(smooth)), comparable)
end

# A ModelZoo.jl model against a TabArena model (default) whose bag of 8 models it fits: on the smallest dataset of each
# Response Type among those TabArena scored it on whose covariates are all numbers with none missing (so that the
# models alone are compared, not the preparation of the data), every split is fitted by `fit` and scored and compared
# with TabArena's score on the same split (`file`, written by scripts/tabarena_scores.py). AutoGluon's bags split the
# rows with other random numbers, so the two agree only in distribution: the mean of the differences over the splits
# must be within 4 standard errors of 0, the standard error being that of the differences. TabArena's probabilities are
# single precision, as its log losses show, so scikit-learn's log_loss clips them at the machine epsilon of Float32,
# and so does this test.
function compare_with_tabarena(fit, model_id, file)
	published = TOML.parsefile(joinpath(@__DIR__, file))
	runs = Set(run["dataset_id"] for run in PC.metadata_entries(PC.METADATA, "benchmarks.toml", "runs") if run["model_id"] == model_id)
	numeric(table, target) = all(n == Symbol(target) || (eltype(table[n]) <: Real && !any(isnan, table[n])) for n in keys(table))
	chosen = String[]
	for kind in ("Binary", "Continuous", "Multiclass")
		candidates = sort([id for id in runs if PC.competition(id)["response_type"] == kind]; by = id -> PC.info(id)["tables"]["data"])
		i = findfirst(id -> numeric(PC.query("SELECT * FROM $(PC.table_sql(id, "data.parquet"))"), PC.info(id)["target"]), candidates)
		push!(chosen, candidates[i])
	end
	@test sort(chosen) == sort(collect(keys(published)))
	@testset "$(id)" for id in chosen
		fields, kind = PC.info(id), PC.competition(id)["response_type"]
		data = PC.query("SELECT * FROM $(PC.table_sql(id, "data.parquet"))")
		splits = PC.query("SELECT row, repeat, fold FROM $(PC.table_sql(id, "splits.parquet"))")
		target = Symbol(fields["target"])
		scores = Float64[]
		for repeat in 1:fields["repeats"], fold in 1:fields["folds"]
			test = sort(splits.row[(splits.repeat .== repeat) .& (splits.fold .== fold)])
			train = setdiff(eachindex(data[target]), test)
			model = fit(map(v -> v[train], data), target; classification = kind != "Continuous")
			p = MZ.predict(model, map(v -> v[test], data))
			y = data[target][test]
			push!(scores, kind == "Continuous" ? MZ.rmse(Float64.(y), p) : kind == "Binary" ? MZ.roc_auc(y .== model.classes[2], p[:, 2]) :
				MZ.log_loss([findfirst(==(v), model.classes) for v in y], Float32.(p)))
		end
		d = scores .- published[id]["scores"]
		mean = sum(d) / length(d)
		standard_error = sqrt(sum(abs2, d .- mean) / (length(d) - 1) / length(d))
		@test published[id]["measure"] == MZ.TABULAR_MEASURES[kind] && abs(mean) <= 4 * standard_error
	end
end

# The data, the splits and the published scores of the competitions and benchmarks come from PollisCompetitions.jl.
@testset "ModelZoo" begin
	# The measures of the Low-Dimensional Survival Benchmark: Kaplan-Meier and Nelson-Aalen, which have nothing to tune and
	# nothing random, fitted by Survival.jl on the training rows of every split of the 34 survival datasets and scored by
	# survival_scores, give, as means over the splits, the scores the benchmark published for them in
	# benchmarks.toml. The training times are first tied (tie_times), as the benchmark's curves tie them.
	@testset "Survival measures" begin
		runs = Dict(run["run_id"] => run["scores"] for run in PC.metadata_entries(PC.METADATA, "benchmarks.toml", "runs"))
		ids = [id for id in readdir(PC.BUILTIN) if PC.competition(id)["response_type"] == "Survival"]
		@test length(ids) == 34
		@testset "$(id)" for id in ids
			fields = PC.info(id)
			data = PC.query("SELECT $(fields["target"]) AS time, $(fields["event"]) AS event FROM $(PC.table_sql(id, "data.parquet"))")
			splits = PC.query("SELECT row, repeat, fold FROM $(PC.table_sql(id, "splits.parquet"))")
			time, event = Float64.(data.time), Int.(data.event)
			for (model, curve) in (("survival_benchmark_km", (t, e) -> (f = Survival.fit(Survival.KaplanMeier, t, e); (f.events.time, f.survival))),
				("survival_benchmark_nel", (t, e) -> (f = Survival.fit(Survival.NelsonAalen, t, e); (f.events.time, exp.(-f.chaz)))))
				scores = [begin
					test = sort(splits.row[(splits.repeat .== repeat) .& (splits.fold .== fold)])
					train = setdiff(eachindex(time), test)
					times, survival = curve(MZ.tie_times(time[train]), event[train])
					MZ.survival_scores(time[train], event[train], time[test], event[test], times, Base.repeat(permutedims(survival), length(test)))
				end for repeat in 1:fields["repeats"] for fold in 1:fields["folds"]]
				published = runs["$(model).$(id)"]
				@test all(isapprox(sum(s[m] for s in scores) / length(scores), published[m]; rtol = 1e-9, atol = 1e-12) for m in MZ.SURVIVAL_MEASURES)
			end
		end
	end

	# The measures of the tabular datasets against the values scikit-learn gives on fixed predictions, in
	# tabular_measures.toml (written by scripts/tabular_measures.py, which says how the predictions are made): every
	# built-in tabular dataset whose Response Type has a measure, on its first split. Ties in the scores of roc_auc and
	# probabilities of 0 in those of log_loss are among them.
	@testset "Tabular measures" begin
		expected = TOML.parsefile(joinpath(@__DIR__, "tabular_measures.toml"))
		ids = [id for id in readdir(PC.BUILTIN) if get(something(PC.info(id), Dict()), "format", nothing) == "tabular" &&
			haskey(MZ.TABULAR_MEASURES, something(PC.competition(id)["response_type"], ""))]
		@test sort(collect(keys(expected))) == ids
		@testset "$(id)" for id in ids
			kind = PC.competition(id)["response_type"]
			target = PC.query("SELECT \"$(PC.info(id)["target"])\" AS y FROM $(PC.table_sql(id, "data.parquet"))").y
			splits = PC.query("SELECT row, repeat, fold FROM $(PC.table_sql(id, "splits.parquet"))")
			test = sort(splits.row[(splits.repeat .== 1) .& (splits.fold .== 1)])
			train = setdiff(eachindex(target), test)
			mine = if MZ.TABULAR_MEASURES[kind] == "rmse"
				y = Float64.(target)
				mean = sum(y[train]) / length(train)
				Dict("mean" => MZ.rmse(y[test], fill(mean, length(test))), "rows" => MZ.rmse(y[test], [mean + r % 7 - 3 for r in test]))
			elseif MZ.TABULAR_MEASURES[kind] == "roc_auc"
				classes = sort(unique(target))
				truth = target[test] .== classes[2]
				share = count(==(classes[2]), target[train]) / length(train)
				Dict("frequency" => MZ.roc_auc(truth, fill(share, length(test))), "rows" => MZ.roc_auc(truth, [(r % 10) / 10 for r in test]))
			else
				classes = sort(unique(target))
				k = length(classes)
				truth = [findfirst(==(y), classes) for y in target[test]]
				shares = [count(==(c), target[train]) / length(train) for c in classes]
				Dict("frequency" => MZ.log_loss(truth, Base.repeat(permutedims(shares), length(test))),
					"rows" => MZ.log_loss(truth, [Float64(r % k + 1 == c) for r in test, c in 1:k]))
			end
			@test expected[id]["response_type"] == kind
			@test all(isapprox(value, expected[id][name]; rtol = 1e-12) for (name, value) in mine) && length(mine) == length(expected[id]) - 1
		end
	end

	# The measures by Response Type, and the ranking of a score, on values worked out by hand
	@testset "Scoring by Response Type" begin
		@test [MZ.measure_loss(m, v) for (m, v) in (("rmse", 3), ("roc_auc", 0.75), ("alpha_calib", 0.8), ("alpha_calib", 1.25), ("isbs_erv", 0.1))] ≈ [3, -0.75, 0.2, 0.25, -0.1]
		@test MZ.tabular_scores("Continuous", [1, 2, 3], [1, 2, 5]) == Dict("rmse" => sqrt(4 / 3))
		@test MZ.tabular_scores("Binary", [true, false, true, false], [0.9, 0.2, 0.4, 0.4]) == Dict("roc_auc" => 0.875)
		@test only(values(MZ.tabular_scores("Multiclass", [1, 3], [0.5 0.25 0.25; 0.2 0.3 0.5]))) ≈ log(2)
		@test_throws ErrorException MZ.tabular_scores("Ordered", [1, 2], [1, 2])
	end

	# Cox's model, fitted by Survival.jl (Efron's ties, as R's coxph), on every outer split of the survival datasets,
	# scored by Harrell's C and Uno's C and compared with the benchmark's CPH means, the covariates prepared by
	# covariates, all entering linearly. A dataset is compared when all its splits can be, as the published scores are
	# means over the splits. A split cannot be when:
	# - a test row has a level of a factor that no training row has: the benchmark imputes it at random;
	# - it is one of COX_DIFFERENT: the published score differs from ours, while R's coxph on the same training rows
	#   gives our value (checked on one of them) and the benchmark logged no error.
	@testset "Cox's model" begin
		COX_DIFFERENT = Dict("aids_id" => [9], "kidtran" => [15, 21], "liver" => [3, 5, 9], "ova" => [24, 30], "std" => [12], "tumor" => [8])
		runs = Dict(run["run_id"] => run["scores"] for run in PC.metadata_entries(PC.METADATA, "benchmarks.toml", "runs"))
		compared = String[]
		@testset "$(id)" for id in [id for id in readdir(PC.BUILTIN) if PC.competition(id)["response_type"] == "Survival"]
			fields = PC.info(id)
			data = PC.query("SELECT * FROM $(PC.table_sql(id, "data.parquet"))")
			splits = PC.query("SELECT row, repeat, fold FROM $(PC.table_sql(id, "splits.parquet"))")
			time, event = Float64.(getproperty(data, Symbol(fields["target"]))), Int.(getproperty(data, Symbol(fields["event"])))
			scores, comparable = Dict{String, Float64}[], !haskey(COX_DIFFERENT, id)
			for repeat in 1:fields["repeats"], fold in 1:fields["folds"]
				test = sort(splits.row[(splits.repeat .== repeat) .& (splits.fold .== fold)])
				train = setdiff(eachindex(time), test)
				prepared = covariates(data, fields, train, test)
				prepared.comparable || (comparable = false)
				X, Xt = prepared.X, prepared.Xt
				# Survival.jl's search reaches its iteration limit on some splits as the covariates are and on others once
				# they are centred: the fit as they are, else centred, which leaves the ranking of the test rows unchanged
				y = Survival.EventTime.(time[train], event[train] .== 1)
				model = try
					Survival.coxph(X, y; tol = 1e-10)
				catch
					centre = sum(X; dims = 1) ./ size(X, 1)
					X, Xt = X .- centre, Xt .- centre
					Survival.coxph(X, y; tol = 1e-10)
				end
				crank = Xt * Survival.coef(model)
				push!(scores, Dict("harrell_c" => MZ.concordance(time[test], event[test], crank, :harrell),
					"uno_c" => MZ.concordance(time[test], event[test], crank, :uno; train_time = time[train], train_event = event[train])))
			end
			comparable || continue
			push!(compared, id)
			published = runs["survival_benchmark_cph.$(id)"]
			@test all(isapprox(sum(s[m] for s in scores) / length(scores), published[m]; rtol = 1e-9, atol = 1e-12) for m in ("harrell_c", "uno_c"))
		end
		@test length(compared) == 24
	end

	# Cox's model with smooth terms (CoxGAM) on every outer split of the survival datasets, compared, as means over the
	# splits, with the benchmark's GAM, mgcv 1.9-1's gam with the cox.ph family and REML, with the formula of
	# mlr3extralearners (a smooth term for each numeric covariate with 10 or more distinct training values), the survival
	# from Breslow's estimate. The covariates are prepared by covariates. A dataset is compared when all its splits can
	# be. A split cannot be when:
	# - a test row has a level of a factor that no training row has: the benchmark imputes it at random;
	# - a smooth covariate has more than 2000 distinct training values: mgcv builds its spline on 2000 of them drawn at
	#   random;
	# - it is one of GAM_DIFFERENT: mgcv stopped at smoothing parameters whose REML criterion is higher than at those
	#   CoxGAM reaches, which the published scores reflect. The criterion can have more than one local minimum: mgcv
	#   searches from one start, CoxGAM from three, keeping the lowest minimum, and as Wood (2011) describes mgcv's
	#   start only in outline, a search from one start cannot be sure to begin where mgcv's does and end where it
	#   ends. Checked on every one of them with mgcv 1.9.4 on the same training rows, by evaluating CoxGAM's criterion
	#   at mgcv's smoothing parameters, put on CoxGAM's scale by the wiggliness each penalty gives mgcv's fitted smooth
	#   terms; on 4 of them (carpenter_fda_data 6, 9 and 14, tumor 30) mgcv 1.9.4 does not give the published scores
	#   either.
	# The REML criterion is flat near its minimum, so smoothing parameters that leave it unchanged to 1e-6 can move the
	# measures by more than that, D-calibration most, as it counts the test rows in bins of their survival.
	@testset "Cox GAM" begin
		GAM_DIFFERENT = Dict("carpenter_fda_data" => [4, 6, 9, 14, 28], "channing" => [11], "e1684" => [11], "metabric" => [1],
			"std" => [25], "tumor" => [30], "uis" => [9], "veteran" => [17, 20, 25, 29], "wbc1" => [4, 7, 24])
		runs = Dict(run["run_id"] => run["scores"] for run in PC.metadata_entries(PC.METADATA, "benchmarks.toml", "runs"))
		compared = String[]
		@testset "$(id)" for id in [id for id in readdir(PC.BUILTIN) if PC.competition(id)["response_type"] == "Survival"]
			haskey(GAM_DIFFERENT, id) && continue
			fields = PC.info(id)
			data = PC.query("SELECT * FROM $(PC.table_sql(id, "data.parquet"))")
			splits = PC.query("SELECT row, repeat, fold FROM $(PC.table_sql(id, "splits.parquet"))")
			time, event = Float64.(getproperty(data, Symbol(fields["target"]))), Int.(getproperty(data, Symbol(fields["event"])))
			# The splits' covariates first, as a dataset that cannot be compared need not be fitted
			prepared = [begin
				test = sort(splits.row[(splits.repeat .== repeat) .& (splits.fold .== fold)])
				train = setdiff(eachindex(time), test)
				(; train, test, covariates(data, fields, train, test; smooth_from = 10)...)
			end for repeat in 1:fields["repeats"] for fold in 1:fields["folds"]]
			all(p -> p.comparable && all(j -> length(unique(view(p.X, :, j))) <= 2000, p.smooth), prepared) || continue
			scores = map(prepared) do p
				model = MZ.CoxGAM(time[p.train], event[p.train], p.X; smooth = p.smooth)
				times = sort!(unique(MZ.tie_times(time[p.train])))
				MZ.survival_scores(time[p.train], event[p.train], time[p.test], event[p.test], times, MZ.survival(model, p.Xt, times);
					crank = MZ.linear_predictor(model, p.Xt))
			end
			push!(compared, id)
			published = runs["survival_benchmark_gam.$(id)"]
			@test all(isapprox(sum(s[m] for s in scores) / length(scores), published[m]; rtol = m == "dcalib" ? 5e-2 : 1e-3, atol = 1e-6)
				for m in MZ.SURVIVAL_MEASURES)
		end
		@test length(compared) == 15
	end

	# ExtraTrees: whether BaggedExtraTrees is TabArena's ExtraTrees (default), scikit-learn's ExtraTrees fitted by
	# AutoGluon as a bag of 8 forests, whose trees also draw other random numbers.
	@testset "ExtraTrees" begin
		compare_with_tabarena(MZ.BaggedExtraTrees, "tabarena_extratrees_default", "extratrees_tabarena.toml")
	end

	# APLR: whether BaggedAPLR is TabArena's APLR (default), aplr 10.26 fitted by AutoGluon as a bag of 8 models, each
	# with one fold of the training rows as its validation rows.
	@testset "APLR" begin
		compare_with_tabarena(MZ.BaggedAPLR, "tabarena_aplr_default", "aplr_tabarena.toml")
	end

	# FlexibleSurvival: whether it is the Low-Dimensional Survival Benchmark's Flex, flexsurv 2.3.2's flexsurvspline with
	# its defaults, its number of knots k tuned from 1 to 10 on inner splits that were not published. On the three
	# smallest survival datasets whose covariates are all numbers, every outer split is fitted with each k, the covariates
	# prepared by covariates, and compared with the scores published for that split (flexible_benchmark.toml, written by
	# scripts/flexible_benchmark.py). Of Flex tuned on Harrell's C, Harrell's and Uno's C must both equal those of one k,
	# to the single precision the benchmark keeps them in. Of Flex tuned on the integrated survival Brier score, the
	# integrated Brier score and log loss must both be within 1e-3 of those of one k, the survival predicted at every
	# distinct training time, as flexsurv predicts it. The log likelihood is concave and FlexibleSurvival reaches its
	# maximum, while flexsurv's search (BFGS) stops near it, which moves the integrated scores by up to about 2e-4, and on
	# the splits of C_DIFFERENT enough to rank some pairs of test rows the other way. Checked on each of them with flexsurv
	# 2.3.2 on the same training rows: its log likelihood is below FlexibleSurvival's at every k, and its fit at one k gives
	# the published C; except on carpenter_fda_data 9, where flexsurv stops with a singular Hessian at every k and the
	# published C of 0.5 is that of equal predictions for every test row (so its integrated scores are not compared
	# either), and carpenter_fda_data 20, where it stops so at k 9 and 10 and no other k gives the published C.
	@testset "Flexible parametric survival model" begin
		C_DIFFERENT = Dict("carpenter_fda_data" => [1, 7, 9, 16, 20, 28, 30], "e1684" => [2, 15, 23, 30], "wbc1" => [4, 18, 29])
		published = TOML.parsefile(joinpath(@__DIR__, "flexible_benchmark.toml"))
		candidates = [id for id in readdir(PC.BUILTIN) if PC.competition(id)["response_type"] == "Survival" && !haskey(PC.info(id), "factors")]
		chosen = sort(candidates; by = id -> PC.info(id)["tables"]["data"])[1:3]
		@test sort(chosen) == sort(collect(keys(published)))
		compared = [0, 0]
		@testset "$(id)" for id in chosen
			fields = PC.info(id)
			data = PC.query("SELECT * FROM $(PC.table_sql(id, "data.parquet"))")
			splits = PC.query("SELECT row, repeat, fold FROM $(PC.table_sql(id, "splits.parquet"))")
			time, event = Float64.(getproperty(data, Symbol(fields["target"]))), Int.(getproperty(data, Symbol(fields["event"])))
			by_c, by_isbs = published[id]["harrell_c"], published[id]["isbs"]
			agree = map(enumerate((r, f) for r in 1:fields["repeats"] for f in 1:fields["folds"])) do (s, (repeat, fold))
				test = sort(splits.row[(splits.repeat .== repeat) .& (splits.fold .== fold)])
				train = setdiff(eachindex(time), test)
				p = covariates(data, fields, train, test)
				times = sort!(unique(time[train]))
				scores = map(1:10) do k
					model = MZ.FlexibleSurvival(time[train], event[train], p.X; k)
					MZ.survival_scores(time[train], event[train], time[test], event[test], times, MZ.survival(model, p.Xt, times);
						crank = MZ.linear_predictor(model, p.Xt))
				end
				c = s in get(C_DIFFERENT, id, Int[]) || any(sc -> all(abs(sc[m] - by_c[m][s]) <= 1e-6 for m in ("harrell_c", "uno_c")), scores)
				integrated = (id, s) == ("carpenter_fda_data", 9) ||
					any(sc -> all(abs(sc[m] / by_isbs[m][s] - 1) <= 1e-3 for m in ("isbs", "isll")), scores)
				compared .+= [!(s in get(C_DIFFERENT, id, Int[])), (id, s) != ("carpenter_fda_data", 9)]
				c && integrated
			end
			@test all(agree)
		end
		@test compared == [76, 89]
	end
end
