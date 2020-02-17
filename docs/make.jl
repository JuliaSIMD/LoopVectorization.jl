using Documenter, LoopVectorization

makedocs(;
    modules=[LoopVectorization],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => Any[
            "examples/matrix_multiplication.md",
            "examples/matrix_vector_ops.md",
            "examples/dot_product.md",
            "examples/sum_of_squared_error.md"
        ],
        "Vectorized Convenience Functions" => "vectorized_convenience_functions.md",
        "Future Work" => "future_work.md"
    ],
    # repo="https://github.com/chriselrod/LoopVectorization.jl/blob/{commit}{path}#L{line}",
    sitename="LoopVectorization.jl",
    authors="Chris Elrod"
    # assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/LoopVectorization.jl",
)
