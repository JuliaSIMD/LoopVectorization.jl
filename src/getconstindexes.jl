for (i, f) ∈ enumerate((
  :second,
  :third,
  :fourth,
  :fifth,
  :sixth,
  :seventh,
  :eighth,
  :ninth,
  :tenth,
  :eleventh,
  :twelfth,
  :thirteenth,
))
  @eval @inline $f(x) = @inbounds getindex(x, $(i + 1))
end
