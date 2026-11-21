using Test
import LinearAlgebra, Survival, TOML
import ModelZoo as MZ
import PollisCompetitions as PC

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
	# scored by Harrell's C and Uno's C and compared with the benchmark's CPH means. The covariates are prepared as the
	# benchmark prepares them for CPH (mlr3pipelines 0.7.1 in its bl(): fixfactors, imputesample of factors,
	# collapsefactors at 5% and 5 levels, removeconstants), the factors of info.toml coded as R's model.matrix codes them
	# (a column per level after the first), and the columns coxph drops as collinear dropped. A dataset is compared when
	# all its splits can be, as the published scores are means over the splits. A split cannot be when:
	# - a test row has a level of a factor that no training row has: the benchmark imputes it at random;
	# - it is one of COX_DIFFERENT: the published score differs from ours, while R's coxph on the same training rows
	#   gives our value (checked on one of them) and the benchmark logged no error.
	@testset "Cox's model" begin
		COX_DIFFERENT = Dict("aids_id" => [9], "kidtran" => [15, 21], "liver" => [3, 5, 9], "ova" => [24, 30], "std" => [12], "tumor" => [8])
		runs = Dict(run["run_id"] => run["scores"] for run in PC.metadata_entries(PC.METADATA, "benchmarks.toml", "runs"))
		label(x::AbstractString) = String(x)
		label(x::Real) = isinteger(x) ? string(Int(x)) : string(x)
		# collapsefactors: with more than 5 levels, the levels of 5% or more of the training rows and the 5 most
		# frequent are kept, and the others merge into the last level kept
		function collapse(levels, counts)
			length(levels) <= 5 && return Dict(l => l for l in levels)
			order = sortperm(counts; rev = true, alg = MergeSort)
			keep = unique([[levels[k] for k in order if counts[k] / sum(counts) >= 0.05]; levels[order[1:5]]])
			Dict(l => (l in keep ? l : keep[end]) for l in levels)
		end
		compared = String[]
		@testset "$(id)" for id in [id for id in readdir(PC.BUILTIN) if PC.competition(id)["response_type"] == "Survival"]
			fields = PC.info(id)
			factors = get(fields, "factors", Dict{String, Any}())
			data = PC.query("SELECT * FROM $(PC.table_sql(id, "data.parquet"))")
			splits = PC.query("SELECT row, repeat, fold FROM $(PC.table_sql(id, "splits.parquet"))")
			time, event = Float64.(getproperty(data, Symbol(fields["target"]))), Int.(getproperty(data, Symbol(fields["event"])))
			features = [String(n) for n in keys(data) if !(String(n) in (fields["target"], fields["event"]))]
			scores, comparable = Dict{String, Float64}[], !haskey(COX_DIFFERENT, id)
			for repeat in 1:fields["repeats"], fold in 1:fields["folds"]
				test = sort(splits.row[(splits.repeat .== repeat) .& (splits.fold .== fold)])
				train = setdiff(eachindex(time), test)
				columns, test_columns = Vector{Float64}[], Vector{Float64}[]
				for n in features
					values = getproperty(data, Symbol(n))
					if haskey(factors, n)
						x = label.(values)
						present = [l for l in factors[n] if any(==(l), x[train])]
						all(v -> v in present, x[test]) || (comparable = false)
						merging = collapse(present, [count(==(l), x[train]) for l in present])
						for l in unique(merging[l] for l in present)[2:end]
							push!(columns, [merging[v] == l ? 1.0 : 0.0 for v in x[train]])
							push!(test_columns, [get(merging, v, "") == l ? 1.0 : 0.0 for v in x[test]])
						end
					else
						push!(columns, Float64.(values[train]))
						push!(test_columns, Float64.(values[test]))
					end
				end
				# removeconstants; then the columns coxph keeps, in order, by a pivoted QR with the intercept
				kept = [k for k in eachindex(columns) if length(unique(columns[k])) > 1]
				X, Xt = reduce(hcat, columns[kept]), reduce(hcat, test_columns[kept])
				F = LinearAlgebra.qr([ones(length(train)) X], LinearAlgebra.ColumnNorm())
				independent = sort(filter(>(1), F.p[1:LinearAlgebra.rank(F.R; rtol = 1e-12)])) .- 1
				X, Xt = X[:, independent], Xt[:, independent]
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
end
