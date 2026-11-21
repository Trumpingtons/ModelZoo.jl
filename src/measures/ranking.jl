"""
Which value of each measure of the tabular and survival datasets is best: "lower", "higher", or "one" for
alpha-calibration, the ratio of observed to expected events, which is best at 1.
"""
const MEASURE_BEST = Dict("rmse" => "lower", "roc_auc" => "higher", "log_loss" => "lower",
	"harrell_c" => "higher", "uno_c" => "higher", "isbs" => "lower", "isbs_erv" => "higher", "isll" => "lower",
	"isll_erv" => "higher", "dcalib" => "lower", "alpha_calib" => "one")

"""
	measure_loss(measure, value)

`value` of `measure` as a number of which lower is better, to rank scores by: the value when lower is better, minus
the value when higher is better, and its distance from 1 for alpha-calibration.
"""
function measure_loss(measure, value)
	best = get(MEASURE_BEST, measure, nothing)
	best === nothing && error("No measure $(measure): one of $(join(sort(collect(keys(MEASURE_BEST))), ", ")).")
	best == "lower" ? Float64(value) : best == "higher" ? -Float64(value) : abs(Float64(value) - 1)
end
