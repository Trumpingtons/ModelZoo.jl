# Extremely randomized trees (ExtraTrees) of Geurts, Ernst and Wehenkel (2006): each tree splits a node at the best of
# randomly drawn splits, one for each of `max_features` covariates drawn at random, at a threshold drawn uniformly
# between the smallest and the largest value of that covariate in the node; the forest averages its trees. The settings
# are those of scikit-learn's ExtraTreesClassifier and ExtraTreesRegressor as AutoGluon fits them (Pedregosa and 18
# coauthors, 2011; Erickson and 6 coauthors, 2020): 300 trees, each on a bootstrap sample of the rows, grown until its
# leaves are pure or it has 15000 leaves (splitting first the node whose split decreases the impurity the most), the
# Gini impurity for classes and the squared error for numbers, √p covariates drawn for classes and all p for numbers.
# A missing value is NaN: the threshold is drawn between the smallest and the largest value that is not missing, and the
# rows whose value is missing go to a side drawn at random, as scikit-learn's random splits send them.
# BaggedExtraTrees fits a forest on each of 8 folds' complement and averages them, as TabArena's ExtraTrees (default).

"""
	ExtraTree

One tree: for each node, the covariate it splits on (0 for a leaf), the threshold (rows with a value at or below it go
left), the side of the rows whose value is missing (left when true), its two children, and its value, the share of each
class or the mean of the response of its training rows.
"""
struct ExtraTree
	feature::Vector{Int}
	threshold::Vector{Float64}
	missing_left::Vector{Bool}
	left::Vector{Int}
	right::Vector{Int}
	value::Matrix{Float64}
end

# The impurity of the rows rows[lo:hi]: the Gini impurity of their classes (y holds class numbers 1:K) or the variance of
# their response (K = 0), with the node's value
function node_value(y, rows, lo, hi, K)
	n = hi - lo + 1
	if K > 0
		counts = zeros(K)
		for r in lo:hi
			counts[y[rows[r]]] += 1
		end
		shares = counts ./ n
		return 1 - sum(abs2, shares), shares
	end
	total = 0.0
	for r in lo:hi
		total += y[rows[r]]
	end
	mean = total / n
	spread = 0.0
	for r in lo:hi
		spread += abs2(y[rows[r]] - mean)
	end
	spread / n, [mean]
end

# Whether a row goes left: its value at or below the threshold, or, when it is missing (NaN), the node's missing side
goes_left(v, threshold, missing_left) = isnan(v) ? missing_left : v <= threshold

# The best of the random splits of rows[lo:hi]: covariates drawn at random without replacement until `max_features`
# that vary in the node have been tried (those whose values that are not missing do not vary are passed over), each at a
# threshold drawn uniformly in [smallest, largest) of the values that are not missing, the rows whose value is missing
# going to a side drawn at random (when there are none, to the side with more rows); the best maximizes the sum over the
# two sides of (Σ count²)/n for classes, or (Σ y)²/n for numbers, which is to minimize the weighted impurity of the two
# sides. Returns the covariate (0 when none varies), the threshold and the missing side.
function random_split(X, y, rows, lo, hi, K, max_features, features, rng, left_counts, right_counts)
	f = length(features)
	tried = 0
	best, best_feature, best_threshold, best_missing = -Inf, 0, 0.0, false
	while f > 0 && tried < max_features
		k = rand(rng, 1:f)
		features[k], features[f] = features[f], features[k]
		j = features[f]
		f -= 1
		low, high, missing = Inf32, -Inf32, 0
		for r in lo:hi
			v = X[rows[r], j]
			if isnan(v)
				missing += 1
			else
				low, high = min(low, v), max(high, v)
			end
		end
		high <= low + 1e-7 && continue
		tried += 1
		threshold = low + rand(rng) * (Float64(high) - low)
		threshold >= high && (threshold = Float64(low))
		missing_left = missing > 0 && rand(rng, Bool)
		proxy, nl, nr = if K > 0
			fill!(left_counts, 0)
			fill!(right_counts, 0)
			for r in lo:hi
				i = rows[r]
				if goes_left(X[i, j], threshold, missing_left)
					left_counts[y[i]] += 1
				else
					right_counts[y[i]] += 1
				end
			end
			nl, nr = sum(left_counts), sum(right_counts)
			sum(abs2, left_counts) / nl + sum(abs2, right_counts) / nr, nl, nr
		else
			sl, sr, nl, nr = 0.0, 0.0, 0, 0
			for r in lo:hi
				i = rows[r]
				if goes_left(X[i, j], threshold, missing_left)
					sl += y[i]
					nl += 1
				else
					sr += y[i]
					nr += 1
				end
			end
			sl^2 / nl + sr^2 / nr, nl, nr
		end
		if proxy > best
			best, best_feature, best_threshold = proxy, j, threshold
			best_missing = missing > 0 ? missing_left : nl >= nr
		end
	end
	best_feature, best_threshold, best_missing
end

# A pending split of a node, ordered by the decrease of impurity it brings, weighted by the node's share of the rows
struct PendingSplit
	gain::Float64
	node::Int
	lo::Int
	hi::Int
	impurity::Float64
	feature::Int
	threshold::Float64
	missing_left::Bool
	mid::Int
end

# A max-heap of the pending splits by gain
function heap_push!(heap, item)
	push!(heap, item)
	k = length(heap)
	while k > 1 && heap[k >> 1].gain < heap[k].gain
		heap[k >> 1], heap[k] = heap[k], heap[k >> 1]
		k >>= 1
	end
	heap
end

function heap_pop!(heap)
	top = heap[1]
	last = pop!(heap)
	if !isempty(heap)
		heap[1] = last
		k, n = 1, length(heap)
		while true
			c = 2k
			c > n && break
			c < n && heap[c + 1].gain > heap[c].gain && (c += 1)
			heap[c].gain > heap[k].gain || break
			heap[k], heap[c] = heap[c], heap[k]
			k = c
		end
	end
	top
end

"""
Grows one tree on the rows `rows` (with repeats, a bootstrap sample) of `X`, with responses `y` (class numbers 1:K, or
numbers when K = 0).
"""
function grow_tree(X, y, rows, K, max_features, max_leaf_nodes, rng)
	features = collect(1:size(X, 2))
	left_counts, right_counts = zeros(Int, K), zeros(Int, K)
	feature, threshold, missing_left, left, right, values = Int[], Float64[], Bool[], Int[], Int[], Vector{Float64}[]
	total = length(rows)
	function add_node!(lo, hi)
		impurity, value = node_value(y, rows, lo, hi, K)
		push!(feature, 0); push!(threshold, 0.0); push!(missing_left, false); push!(left, 0); push!(right, 0); push!(values, value)
		length(feature), impurity
	end
	heap = PendingSplit[]
	# The split of a node that is neither pure nor a single row, with its partition of the rows and its gain
	function consider!(node, lo, hi, impurity)
		(hi - lo + 1 >= 2 && impurity > eps()) || return
		j, t, m = random_split(X, y, rows, lo, hi, K, max_features, features, rng, left_counts, right_counts)
		j == 0 && return
		# Partition rows[lo:hi]: those that go left first
		a, b = lo, hi
		while a <= b
			if goes_left(X[rows[a], j], t, m)
				a += 1
			else
				rows[a], rows[b] = rows[b], rows[a]
				b -= 1
			end
		end
		mid = a - 1
		il, _ = node_value(y, rows, lo, mid, K)
		ir, _ = node_value(y, rows, mid + 1, hi, K)
		n, nl, nr = hi - lo + 1, mid - lo + 1, hi - mid
		gain = (n / total) * (impurity - (nl / n) * il - (nr / n) * ir)
		heap_push!(heap, PendingSplit(gain, node, lo, hi, impurity, j, t, m, mid))
	end
	root, impurity = add_node!(1, total)
	consider!(root, 1, total, impurity)
	leaves = 1
	while !isempty(heap) && leaves < max_leaf_nodes
		s = heap_pop!(heap)
		l, il = add_node!(s.lo, s.mid)
		r, ir = add_node!(s.mid + 1, s.hi)
		feature[s.node], threshold[s.node], missing_left[s.node], left[s.node], right[s.node] = s.feature, s.threshold, s.missing_left, l, r
		leaves += 1
		consider!(l, s.lo, s.mid, il)
		consider!(r, s.mid + 1, s.hi, ir)
	end
	ExtraTree(feature, threshold, missing_left, left, right, reduce(hcat, values))
end

function leaf(tree::ExtraTree, x)
	node = 1
	while tree.feature[node] != 0
		node = goes_left(x[tree.feature[node]], tree.threshold[node], tree.missing_left[node]) ? tree.left[node] : tree.right[node]
	end
	node
end

"""
	ExtraTrees

A forest of extremely randomized trees: its trees, and the classes in order (`nothing` for a numeric response), with
the coding of a table's columns when it was fitted to a table.
"""
struct ExtraTrees
	trees::Vector{ExtraTree}
	classes::Union{Nothing, Vector}
	coding::Any
end

"""
	ExtraTrees(X, y; classification = !(eltype(y) <: AbstractFloat), trees = 300, max_features, max_leaf_nodes = 15000,
		bootstrap = true, seed = 0)

Fits a forest of `trees` extremely randomized trees (Geurts, Ernst and Wehenkel, 2006) to the rows of the covariate
matrix `X`, in which a missing value is NaN, and the responses `y`, classes or numbers. Each tree grows on a bootstrap sample of the rows (all the rows
with `bootstrap = false`) until its leaves are pure or it has `max_leaf_nodes` leaves, splitting each node at the best
of random splits of `max_features` covariates, by default √p for classes and all p for numbers, and splitting first the
node whose split decreases the impurity (Gini, or squared error) the most. The trees are grown on Julia's threads, each
with its own random numbers from `seed`, so a fit does not depend on the number of threads.

```julia
import ModelZoo
X = [Float64(i % 7 + j) for i in 1:60, j in 1:3]
model = ModelZoo.ExtraTrees(X, [x > 4 ? "high" : "low" for x in X[:, 1]]; trees = 50)
ModelZoo.predict(model, X[1:3, :])
```
"""
function ExtraTrees(X::AbstractMatrix{<:Real}, y::AbstractVector; classification::Bool = !(eltype(y) <: AbstractFloat),
	trees::Integer = 300, max_features::Integer = classification ? max(1, floor(Int, sqrt(size(X, 2)))) : size(X, 2),
	max_leaf_nodes::Integer = 15000, bootstrap::Bool = true, seed::Integer = 0, coding = nothing)
	data = Float32.(X)
	classes, response, K = if classification
		levels = sort!(unique(y))
		levels, [searchsortedfirst(levels, v) for v in y], length(levels)
	else
		nothing, Float64.(y), 0
	end
	n = size(data, 1)
	forest = Vector{ExtraTree}(undef, trees)
	Threads.@threads for t in 1:trees
		rng = Random.Xoshiro(seed * 1_000_003 + t)
		rows = bootstrap ? rand(rng, 1:n, n) : collect(1:n)
		forest[t] = grow_tree(data, response, rows, K, max_features, max_leaf_nodes, rng)
	end
	ExtraTrees(forest, classes, coding)
end

"""
	predict(model, X)

The predictions of a forest (`ExtraTrees` or `BaggedExtraTrees`) for the rows of `X`: for classes, a matrix with the
probability of each class (a column each, in the order of `model.classes`), the mean of the trees' shares; for numbers,
the mean of the trees' values.
"""
function predict(model::ExtraTrees, X::AbstractMatrix{<:Real})
	data = Float32.(X)
	outputs = size(first(model.trees).value, 1)
	result = zeros(size(data, 1), outputs)
	for tree in model.trees, i in axes(data, 1)
		result[i, :] .+= view(tree.value, :, leaf(tree, view(data, i, :)))
	end
	result ./= length(model.trees)
	model.classes === nothing ? vec(result) : result
end

"""
	BaggedExtraTrees

Forests fitted to the complements of the folds of the training rows, whose predictions are averaged.
"""
struct BaggedExtraTrees
	forests::Vector{ExtraTrees}
	classes::Union{Nothing, Vector}
	coding::Any
end

"""
	bag_folds(y, folds, seed; classification)

The fold, from 1 to `folds`, of each row, as AutoGluon's bags split the rows: at random (with the generator Xoshiro
seeded with `seed`), stratified by class for classes, so that each fold holds an equal share of each class.
"""
function bag_folds(y::AbstractVector, folds::Integer, seed::Integer; classification::Bool)
	rng = Random.Xoshiro(seed)
	fold = zeros(Int, length(y))
	groups = classification ? [findall(==(c), y) for c in sort!(unique(y))] : [collect(eachindex(y))]
	next = 0
	for group in groups
		for i in Random.shuffle(rng, group)
			fold[i] = next % folds + 1
			next += 1
		end
	end
	fold
end

"""
	BaggedExtraTrees(X, y; folds = 8, seed = 0, classification, kwargs...)

Splits the rows into `folds` folds at random, stratified by class for classes, fits an `ExtraTrees` forest (with
`kwargs`) to the rows outside each fold, and averages their predictions: TabArena's ExtraTrees (default), which
AutoGluon fits as a bag of 8 forests.
"""
function BaggedExtraTrees(X::AbstractMatrix{<:Real}, y::AbstractVector; folds::Integer = 8, seed::Integer = 0,
	classification::Bool = !(eltype(y) <: AbstractFloat), coding = nothing, kwargs...)
	fold = bag_folds(y, folds, seed; classification)
	forests = [ExtraTrees(X[fold .!= k, :], y[fold .!= k]; classification, seed = seed * folds + k, kwargs...) for k in 1:folds]
	BaggedExtraTrees(forests, classification ? sort!(unique(y)) : nothing, coding)
end

function predict(model::BaggedExtraTrees, X::AbstractMatrix{<:Real})
	sum(predict(forest, X) for forest in model.forests) ./ length(model.forests)
end

# Tables: a numeric column as it is, any other column by the place of its value among the training rows' sorted levels,
# from 0, and a missing value, or a level the training rows lack, as NaN.

struct OrdinalCoding
	names::Vector{Symbol}
	levels::Dict{Symbol, Vector{Any}}
end

function ordinal_coding(table, names)
	numeric(v) = nonmissingtype(eltype(v)) <: Real
	OrdinalCoding(names, Dict{Symbol, Vector{Any}}(n => sort!(unique(skipmissing(table[n]))) for n in names if !numeric(table[n])))
end

function encode(coding::OrdinalCoding, table)
	X = zeros(Float32, length(table[first(coding.names)]), length(coding.names))
	for (j, name) in enumerate(coding.names)
		values = table[name]
		if haskey(coding.levels, name)
			levels = coding.levels[name]
			for (i, v) in enumerate(values)
				k = ismissing(v) ? 0 : searchsortedfirst(levels, v)
				X[i, j] = (k == 0 || k > length(levels) || levels[k] != v) ? NaN32 : k - 1
			end
		else
			for (i, v) in enumerate(values)
				X[i, j] = ismissing(v) ? NaN32 : v
			end
		end
	end
	X
end

function table_fit(fit, train::NamedTuple, target; kwargs...)
	names = [n for n in keys(train) if String(n) != String(target)]
	coding = ordinal_coding(train, names)
	y = collect(skipmissing(train[Symbol(target)]))
	length(y) == length(train[Symbol(target)]) || error("The response $(target) has missing values.")
	fit(encode(coding, train), y; coding, kwargs...)
end

"""
	ExtraTrees(train, target; kwargs...)
	BaggedExtraTrees(train, target; kwargs...)

Fits to a table of columns (a NamedTuple of vectors) whose column `target` is the response and whose other columns are
the covariates: a numeric column as it is, any other column by the place of its value among the training rows' sorted
levels, from 0, and a missing value, or a level the training rows lack, as NaN.
`predict(model, test)` takes a table with the same columns.

```julia
function predict(train, test, target)
	model = ModelZoo.BaggedExtraTrees(train, target)
	ModelZoo.predict(model, test)
end
```
"""
ExtraTrees(train::NamedTuple, target; kwargs...) = table_fit(ExtraTrees, train, target; kwargs...)
BaggedExtraTrees(train::NamedTuple, target; kwargs...) = table_fit(BaggedExtraTrees, train, target; kwargs...)

predict(model::Union{ExtraTrees, BaggedExtraTrees}, table::NamedTuple) = predict(model, encode(model.coding, table))
