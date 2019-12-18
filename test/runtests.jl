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

exit()
@time using LoopVectorization
using Test
gemmq = :(for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
      Cᵢⱼ = 0.0
      for k ∈ 1:size(A,2)
          Cᵢⱼ += A[i,k] * B[k,j]
      end
      C[i,j] = Cᵢⱼ
      end)

lsgemm = LoopVectorization.LoopSet(gemmq);
U, T = if LoopVectorization.VectorizationBase.REGISTER_COUNT == 16
    (3,4)
else
    (5,5)
end
@test LoopVectorization.choose_order(lsgemm) == (Symbol[:j,:i,:k], U, T)
LoopVectorization.choose_order(lsgemm)
LoopVectorization.lower(lsgemm)
lsgemm.operations

LoopVectorization.choose_tile(lsgemm)
LoopVectorization.choose_unroll_order(lsgemm)

ops = LoopVectorization.oporder(lsgemm);
findall(length.(ops) .!= 0)

dotq = :(for i ∈ eachindex(a)
         s += a[i]*b[i]
         end)
lsdot = LoopVectorization.LoopSet(dotq);
@test LoopVectorization.choose_order(lsdot) == (Symbol[:i], 8, -1)
LoopVectorization.lower(lsdot)

vexpq = :(for i ∈ eachindex(a)
          b[i] = exp(a[i])
          end)
lsvexp = LoopVectorization.LoopSet(vexpq);
@test LoopVectorization.choose_order(lsvexp) == (Symbol[:i], 1, -1)
LoopVectorization.lower(lsvexp)

vexpsq = :(for i ∈ eachindex(a)
          s += exp(a[i])
          end)
lsvexps = LoopVectorization.LoopSet(vexpsq);
@test LoopVectorization.choose_order(lsvexps) == (Symbol[:i], 1, -1)
LoopVectorization.lower(lsvexps)

gemvq = :(for i ∈ eachindex(y)
          yᵢ = 0.0
          for j ∈ eachindex(x)
          yᵢ += A[i,j] * x[j]
          end
          y[i] = yᵢ
          end)
lsgemv = LoopVectorization.LoopSet(gemvq);
@test LoopVectorization.choose_order(lsgemv) == (Symbol[:i, :j], 4, -1)
LoopVectorization.lower(lsgemv)

lsgemv.preamble
LoopVectorization.lower(lsgemv)
LoopVectorization.lower_unrolled(lsgemv, 4);

lsgemv.operations

@code_warntype LoopVectorization.choose_order(lsgemm)

lsgemm.operations

lssi = last(ls.operations)
lssi.dependencies
lssi.reduced_deps

lssi.parents

LoopVectorization.num_loops(ls)
LoopVectorization.choose_tile(ls)

LoopVectorization.unitstride(ls.operations[2], :i)

order1 = [:i,:j,:k];
order2 = [:j,:i,:k];
LoopVectorization.evaluate_cost_tile(ls, order1)
LoopVectorization.evaluate_cost_tile(ls, order2)

@code_warntype LoopVectorization.vector_cost(:getindex, 3, 8)
@code_warntype LoopVectorization.cost(first(ls.operations), :i, 3, 8)


LoopVectorization.determine_unroll_factor(ls, [:i,:j])


LoopVectorization.choose_unroll_order(lsgemv)
LoopVectorization.evaluate_cost_unroll(lsgemv, [:i,:j])
LoopVectorization.evaluate_cost_unroll(lsgemv, [:j,:i])

lsgemv.operations

lsvexp.operations

lo = LoopVectorization.LoopOrders(ls);
new_order, state = iterate(lo)
LoopVectorization.evaluate_cost_tile(ls, new_order)

iter = iterate(lo, state)
new_order, state = iter
LoopVectorization.evaluate_cost_tile(ls, new_order)

X = [1.8000554666666666e8, 1.073741824e9, 1.7895697066666666e8, 0.0];
R = [1, 1, 1, 0];
X = [1.79306496e8, 1.7895697066666666e8, 1.7895697066666666e8, 0.0];
R = [1, 1, 1, 0];
LoopVectorization.solve_tilesize(X, R)

using BenchmarkTools
@benchmark LoopVectorization.choose_order($ls)



