using PrettyTables

function Base.show(io::IO, br::BenchmarkResult)
    hb = Highlighter(
        (br,i,j) -> (j > 1 && maximum(@view(br.results[:, i])) == br.results[j-1,i]),
        foreground = :green
    );
    pretty_table(
        io, br.sizedresults, br.tests, crop = :none, highlighters = (hb,)
    )
end

using VegaLite, IndexedTables
function plot(br::BenchmarkResult)
    res = vec(br.sizedresults.results)
    brsizes = br.sizedresults.sizes
    sizes = Vector{eltype(brsizes)}(undef, length(res))
    ntests = length(br.tests) - 1
    for i ∈ 0:length(brsizes)-1
        si = brsizes[i+1]
        for j ∈ 1:ntests
            sizes[j + i*ntests] = si
        end
    end
    names = ["$(i > 9 ? string(i) : "0$i"). $test" for (i,test) ∈ enumerate(@view(br.tests[2:end]))]
    tests = reduce(vcat, (names for _ ∈ eachindex(brsizes)))
    t = table((GFLOPS = res, Size = sizes, Method = tests))
    t |> @vlplot(
        :line,
        x = :Size,
        y = :GFLOPS,
        width = 900,
        height = 600,
        color={
            :Method,
            scale={scheme="category20"}
        }
    )
end

