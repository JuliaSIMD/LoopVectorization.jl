using Documenter, LoopVectorization

makedocs(;
  modules = [LoopVectorization],
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  pages = [
    "Home" => "index.md",
    "Getting Started" => "getting_started.md",
    "Examples" => [
      "examples/multithreading.md",
      "examples/matrix_multiplication.md",
      "examples/array_interface.md",
      "examples/matrix_vector_ops.md",
      "examples/dot_product.md",
      "examples/datetime_arrays.md",
      "examples/special_functions.md",
      "examples/sum_of_squared_error.md",
      "examples/filtering.md",
    ],
    "Vectorized Convenience Functions" => "vectorized_convenience_functions.md",
    "Future Work" => "future_work.md",
    "API reference" => "api.md",
    "Developer Documentation" => [
      "devdocs/overview.md",
      "devdocs/loopset_structure.md",
      "devdocs/constructing_loopsets.md",
      "devdocs/evaluating_loops.md",
      "devdocs/lowering.md",
      "devdocs/reference.md",
    ],
  ],
  # repo="https://github.com/JuliaSIMD/LoopVectorization.jl/blob/{commit}{path}#L{line}",
  sitename = "LoopVectorization.jl",
  authors = "Chris Elrod",
  # assets=[],
)

deploydocs(; repo = "github.com/JuliaSIMD/LoopVectorization.jl")
