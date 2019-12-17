using Test
using LoopVectorization

stride1(x) = stride(x, 1)
@testset "LoopVectorization.jl" begin

    
    @generated function logsumexp!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
        quote
            n = length(x)
            length(r) == n || throw(DimensionMismatch())
            isempty(x) && return -T(Inf)
            1 == stride1(r) == stride1(x) || throw(error("Arrays not strided"))

            u = maximum(x)                                       # max value used to re-center
            abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values

            s = zero(T)
            @vectorize $T for i = 1:n
                tmp = exp(x[i] - u)
                r[i] = tmp
                s += tmp
            end

            invs = inv(s)
            r .*= invs

            return log1p(s-1) + u
        end
    end

    x = collect(1:1_000) ./ 10;
    r = similar(x);

    @test logsumexp!(r, x) ≈ 102.35216846104409

end


using LoopVectorization
q = :(for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
      Cᵢⱼ = 0.0
      for k ∈ 1:size(A,2)
      Cᵢⱼ += A[i,k] * B[k,j]
      end
      C[i,j] = Cᵢⱼ
      end)

ls = LoopVectorization.LoopSet(q);
ls.operations

lssi = last(ls.operations)
lssi.dependencies
lssi.reduced_deps

lssi.parents

LoopVectorization.num_loops(ls)
LoopVectorization.choose_tile(ls)
LoopVectorization.choose_order(ls)

lo = LoopVectorization.LoopOrders(ls);
new_order, state = iterate(lo)
LoopVectorization.evaluate_cost_tile(ls, new_order)

iter = iterate(lo, state)
new_order, state = iter
LoopVectorization.evaluate_cost_tile(ls, new_order)

X = [1.8000554666666666e8, 1.073741824e9, 1.7895697066666666e8, 0.0];
R = [1, 0, 0, 0];
LoopVectorization.solve_tilesize(X, R)

using BenchmarkTools
@benchmark LoopVectorization.choose_order($ls)



