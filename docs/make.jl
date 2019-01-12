using Documenter, LoopVectorization

makedocs(;
    modules=[LoopVectorization],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/LoopVectorization.jl/blob/{commit}{path}#L{line}",
    sitename="LoopVectorization.jl",
    authors="Chris Elrod",
    assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/LoopVectorization.jl",
)
