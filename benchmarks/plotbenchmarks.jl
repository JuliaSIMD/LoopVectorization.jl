
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
    tests = vcat((@view(br.tests[2:end]) for _ ∈ eachindex(brsizes))...)
    t = table((GFLOPS = res, Size = sizes, Method = tests))
    t |> @vlplot(
        :line,
        x = :Size,
        y = :GFLOPS,
        color = :Method
    )
end

