using Documenter, NLPModels

makedocs(
  modules = [NLPModels],
  doctest = true,
  strict = true,
  assets = ["assets/style.css"],
  format = :html,
  sitename = "NLPModels.jl",
  pages = ["Home" => "index.md",
           "Models" => "models.md",
           "Tools" => "tools.md",
           "Tutorial" => "tutorial.md",
           "API" => "api.md",
           "Reference" => "reference.md"
          ]
)

deploydocs(deps = nothing, make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/NLPModels.jl.git",
  target = "build",
  devbranch = "master"
)