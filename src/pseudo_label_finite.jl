module pseudo_label_finite

using Reexport

# --- for test ---
greet() = print("I am pseudo_label_finite")
export greet

# for numerical experiments ------
include("ExperimentUtil.jl")
@reexport   using   .ExperimentUtil


# --- simple supervised learning -----
include("SupervisedSolver.jl")
@reexport   using   .SupervisedSolver

# --- iterative training ---
# 単発更新のアルゴリズムはもうないので、



end # module pseudo_label_finite
