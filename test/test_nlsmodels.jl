mutable struct DummyNLSModel <: AbstractNLSModel
end

model = DummyNLSModel()

for mtd in [:residual!, :jac_structure_residual!, :jac_coord_residual!, :hess_structure_residual!]
  @eval @test_throws(NotImplementedError, $mtd(model, [0], [1]))
end
for mtd in [:jprod_residual!, :jtprod_residual!, :hess_coord_residual!]
  @eval @test_throws(NotImplementedError, $mtd(model, [0], [1], [2]))
end
@test_throws(NotImplementedError, jth_hess_residual(model, [0], 1))
@test_throws(NotImplementedError, hprod_residual!(model, [0], 1, [2], [3]))

include("test_lls_model.jl")
