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


using Colors, Gadfly
const COLORS = distinguishable_colors(21, [RGB(1,1,1), RGB(0,0,0)])
const COLOR_MAP = Dict{String,RGB{Colors.N0f8}}()
function getcolor(s::String)
    get!(COLOR_MAP, s) do
        COLORS[length(COLOR_MAP) + 2]
    end
end


function Gadfly.plot(br::BenchmarkResult)
    res = br.sizedresults.results
    sizes = br.sizedresults.sizes
    # sizes = Vector{eltype(brsizes)}(undef, length(res))
    tests = @view(br.tests[2:end])
    ntests = length(tests)
    colors = getcolor.(tests)
    
    xt = 0:20:260
    maxres = maximum(res)
    maxtick = 10round(Int, 0.1maxres)
    yt = if iszero(maxtick)
        maxtick = 10round(0.1maxres)
        range(0, maxtick, length = 20)
    elseif maxtick < 50
        0:5:maxtick
    elseif maxtick < 20
        0:2:maxtick
    elseif maxtick < 10
        0:1:maxtick
    else
        0:10:maxtick
    end
    p = Gadfly.plot(
        Gadfly.Guide.manual_color_key("Methods", tests, colors),
        Guide.xlabel("Size"), Guide.ylabel("GFLOPS"),
        Guide.xticks(ticks=collect(xt)), Guide.yticks(ticks=collect(yt))
    )
    for i ∈ eachindex(tests)
        push!(p, layer(x = sizes, y = res[i,:], Geom.line, Theme(default_color=colors[i])))
    end
    p
end

# using VegaLite, IndexedTables
# function plot(br::BenchmarkResult)
#     res = vec(br.sizedresults.results)
#     brsizes = br.sizedresults.sizes
#     sizes = Vector{eltype(brsizes)}(undef, length(res))
#     ntests = length(br.tests) - 1
#     for i ∈ 0:length(brsizes)-1
#         si = brsizes[i+1]
#         for j ∈ 1:ntests
#             sizes[j + i*ntests] = si
#         end
#     end
#     names = ["$(i > 9 ? string(i) : "0$i"). $test" for (i,test) ∈ enumerate(@view(br.tests[2:end]))]
#     tests = reduce(vcat, (names for _ ∈ eachindex(brsizes)))
#     t = table((GFLOPS = res, Size = sizes, Method = tests))
#     t |> @vlplot(
#         :line,
#         x = :Size,
#         y = :GFLOPS,
#         width = 900,
#         height = 600,
#         color={
#             :Method,
#             scale={scheme="category20"}
#         }
#     )
# end

