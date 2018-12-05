#See JuliaSmoothOptimizers/NLPModels.jl/issues/113
__precompile__()

module NLPModels

using LinearAlgebra, LinearOperators, Printf, SparseArrays, FastClosures

export AbstractNLPModelMeta, NLPModelMeta, AbstractNLPModel, Counters,
       acceptsObjective, acceptsNLS, acceptsLS, acceptsMultipleObjective,
       NotImplementedError

mutable struct NotImplementedError <: Exception
  name :: Union{Symbol,Function,String}
end

Base.showerror(io::IO, e::NotImplementedError) = print(io, e.name, " not implemented")

include("nlp_utils.jl")
include("nlp_types.jl")
include("api.jl")
include("dercheck.jl")

# Traits
acceptsObjective(::AbstractNLPModel)         = false
acceptsNLS(::AbstractNLPModel)               = false
acceptsLS(::AbstractNLPModel)                = false
acceptsMultipleObjective(::AbstractNLPModel) = false

# Models
include("autodiff_model.jl")
include("lls_model.jl")

# Derived models
include("slack_model.jl")
include("qn_model.jl")
include("feasibility_residual.jl")

end # module
