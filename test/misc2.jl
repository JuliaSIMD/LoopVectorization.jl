
using LoopVectorization, Test

function tuptestturbo(x,y,t)
  s = 0.0
  @turbo for i = eachindex(x,y)
    s += x[i]*t[1] - y[i]*t[2]
  end
  s
end
function tuptest(x,y,t)
  s = 0.0
  @inbounds @fastmath for i = eachindex(x,y)
    s += x[i]*t[1] - y[i]*t[2]
  end
  s
end
x = rand(127); y = rand(127); t = (rand(),rand());
@test tuptestturbo(x,y,t) ≈ tuptest(x,y,t)

