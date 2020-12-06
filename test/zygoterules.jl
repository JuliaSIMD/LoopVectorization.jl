using Zygote

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

function gradcheck(f, xs...)
  grad_zygote = gradient(f, xs...)
  grad_finite_difference = ngradient(f, xs...)
  return all(isapprox.(grad_zygote, grad_finite_difference; rtol = 1e-5, atol = 1e-5))
end

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

@testset "Zygote adjoint for $mapfunc" for mapfunc in [map, vmap] begin
    @test gradtest(xs -> sum(mapfunc(x -> x^2, xs)), rand(2,3))
    @test gradtest((xss...) -> sum(mapfunc((xs...) -> sqrt(sum(xs.^2)), xss...)), [rand(5) for _ in 1:6]...)
    function foo(y)
      bar = (x) -> x*y
      sum(mapfunc(bar, 1:5))
    end
    @test gradtest(foo, 3)
    # @test gradient(v -> sum([x for x in v]), [1.1,2.2,3.3]) == ([1, 1, 1],)
  # end

  # @testset "Tuple adjoint" begin
    x = randn(3)
    _, pb = Zygote.pullback(x -> mapfunc(abs2, x), x)
    Δy = randn(3)
    @test first(pb((Δy..., ))) ≈ first(pb(Δy))
  # end
end
end
