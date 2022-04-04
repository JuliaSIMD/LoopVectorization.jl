using PrettyTables

function Base.show(io::IO, br::BenchmarkResult)
  hb = Highlighter(
    (br, i, j) -> (j > 1 && maximum(@view(br.results[:, i])) == br.results[j-1, i]),
    foreground = :green,
  )
  pretty_table(io, br.sizedresults, br.tests, crop = :none, highlighters = (hb,))
end


if (Sys.ARCH === :aarch64) && Sys.isapple()
  nothing
else

  using Colors, ColorSchemes, Gadfly
  const COLORS = [RGB(0.0, 0.0, 0.0), RGB(1.0, 0.0, 0.0)]
  # const COLORS = [RGB(0.0,0.0,0.0),RGB(0.0,1.0,0.0)]
  # const COLORS = distinguishable_colors(14, pushfirst!(get(ColorSchemes.Paired_12, (0.5:11.5) ./ 12), RGB(0.0,0.0,0.0)))
  for i ∈ 1:12 # 11 is number of tested libs - 2
    push!(COLORS, get(ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25, i / 12))
    # push!(COLORS, get(ColorSchemes.vikO, (i-0.5)/12))
  end
  # const COLOR_MAP = Dict{String,RGB{Float64}}()
  # const COLOR_MAP = Dict{String,RGB{Colors.N0f8}}()
  const COLOR_MAP64 = Dict{String,RGB{Float64}}()
  function getcolor(s::String)
    get!(COLOR_MAP64, s) do
      COLORS[length(COLOR_MAP64)+1]
    end
  end
  replace_and(str) = replace(str, '&' => "with")

  function Gadfly.plot(br::BenchmarkResult)
    res = br.sizedresults.results
    sizes = br.sizedresults.sizes
    # sizes = Vector{eltype(brsizes)}(undef, length(res))
    tests = replace_and.(@view(br.tests[2:end]))
    colors = getcolor.(tests)
    addlabel = false

    maxxval, maxxind = findmax(sizes)
    maxxtick = 10cld(maxxval, 10) + (addlabel ? 20 : 0)
    xt = 0:20:maxxtick
    maxres = maximum(res)
    maxtick = 10round(Int, 0.1maxres)
    yt = if iszero(maxtick)
      maxtick = 10round(0.1maxres)
      range(0, maxres, length = 20)
    elseif maxtick < 10
      0:1:maxtick
    elseif maxtick < 20
      0:2:maxtick
    elseif maxtick < 50
      0:5:maxtick
    else
      0:10:maxtick
    end
    p = Gadfly.plot(
      Gadfly.Guide.manual_color_key("Methods", tests, colors),
      Guide.xlabel("Size"),
      Guide.ylabel("GFLOPS"),
      Guide.xticks(ticks = collect(xt)),
      Guide.yticks(ticks = collect(yt)),
    )
    for i ∈ eachindex(tests)
      push!(p, layer(x = sizes, y = res[i, :], Geom.line, Theme(default_color = colors[i])))
    end
    addlabel && push!(
      p,
      layer(
        x = fill(maxxtick - 10, length(tests)),
        y = res[:, maxxind],
        label = tests,
        Geom.label(position = :centered),
      ),
    )
    p
  end

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
