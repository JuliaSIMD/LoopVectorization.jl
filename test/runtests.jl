using Test
using LoopVectorization, VectorizationBase, SIMDPirates

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
LoopVectorization.lower(lsgemm)


function mygemm!(C, A, B)
    @inbounds for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
        Cᵢⱼ = 0.0
        @simd ivdep for k ∈ 1:size(A,2)
            Cᵢⱼ += A[i,k] * B[k,j]
        end
        C[i,j] = Cᵢⱼ
    end
end
function mygemmavx!(C, A, B)
    @avx for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
        Cᵢⱼ = 0.0
        for k ∈ 1:size(A,2)
            Cᵢⱼ += A[i,k] * B[k,j]
        end
        C[i,j] = Cᵢⱼ
    end
end

M, K, N = rand(70:81, 3);
C = Matrix{Float64}(undef, M, N); A = randn(M, K); B = randn(K, N);
C2 = similar(C);
mygemmavx!(C, A, B)
mygemm!(C2, A, B)
@test all(C .≈ C2)

using BenchmarkTools
@benchmark mygemmavx!($C, $A, $B)
@benchmark mygemm!($C, $A, $B)
using LinearAlgebra
BLAS.set_num_threads(1)
@benchmark mul!($C2, $A, $B)

LoopVectorization.choose_order(lsgemm)
lsgemm.operations

LoopVectorization.choose_tile(lsgemm)
LoopVectorization.choose_unroll_order(lsgemm)

ops = LoopVectorization.oporder(lsgemm);
findall(length.(ops) .!= 0)

dotq = :(for i ∈ eachindex(a,b)
         s += a[i]*b[i]
         end)
lsdot = LoopVectorization.LoopSet(dotq);
@test LoopVectorization.choose_order(lsdot) == (Symbol[:i], 4, -1)
LoopVectorization.lower(lsdot)
lsdot.operations

function mydot(a, b)
    s = 0.0
    @inbounds @simd for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    s
end
function mydotavx(a, b)
    s = 0.0
    @avx for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    s
end
a = rand(400); b = rand(400);
@test mydotavx(a,b) ≈ mydot(a,b)

@benchmark mydotavx($a,$b)
@benchmark mydot($a,$b)

a = rand(43); b = rand(43);
@benchmark mydotavx($a,$b)
@benchmark mydot($a,$b)

selfdotq = :(for i ∈ eachindex(a)
         s += a[i]*a[i]
         end)
lsselfdot = LoopVectorization.LoopSet(selfdotq);
@test LoopVectorization.choose_order(lsselfdot) == (Symbol[:i], 8, -1)
LoopVectorization.lower(lsselfdot)

function myselfdot(a)
    s = 0.0
    @inbounds @simd for i ∈ eachindex(a)
        s += a[i]*a[i]
    end
    s
end
function myselfdotavx(a)
    s = 0.0
    @avx for i ∈ eachindex(a)
        s += a[i]*a[i]
    end
    s
end

a = rand(400);
@test myselfdotavx(a) ≈ myselfdot(a)

@benchmark myselfdotavx($a)
@benchmark myselfdot($a)

@benchmark myselfdotavx($b)
@benchmark myselfdot($b)


vexpq = :(for i ∈ eachindex(a)
          b[i] = exp(a[i])
          end)
lsvexp = LoopVectorization.LoopSet(vexpq);
@test LoopVectorization.choose_order(lsvexp) == (Symbol[:i], 1, -1)
LoopVectorization.lower(lsvexp)

function myvexp!(b, a)
    @inbounds for i ∈ eachindex(a)
        b[i] = exp(a[i])
    end
end
function myvexpavx!(b, a)
    @avx for i ∈ eachindex(a)
        b[i] = exp(a[i])
    end
end
a = randn(127);
b1 = similar(a);
b2 = similar(a);

myvexp!(b1, a)
myvexpavx!(b2, a)
b1'
b2'
all(b1 .≈ b2)
@test all(b1 .≈ b2)

@benchmark myvexp!($b1, $a)
@benchmark myvexpavx!($b2, $a)


vexpsq = :(for i ∈ eachindex(a)
          s += exp(a[i])
          end)
lsvexps = LoopVectorization.LoopSet(vexpsq);
@test LoopVectorization.choose_order(lsvexps) == (Symbol[:i], 1, -1)
LoopVectorization.lower(lsvexps)
lsvexps.operations

function myvexp(a)
    s = 0.0
    @inbounds for i ∈ eachindex(a)
        s += exp(a[i])
    end
    s
end
function myvexpavx(a)
    s = 0.0
    @avx for i ∈ eachindex(a)
        s += exp(a[i])
    end
    s
end

@test myvexp(a) ≈ myvexpavx(a)

@benchmark myvexp($a)
@benchmark myvexpavx($a)

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


function mygemv!(y, A, x)
    @inbounds for i ∈ eachindex(y)
        yᵢ = 0.0
        @simd for j ∈ eachindex(x)
            yᵢ += A[i,j] * x[j]
        end
        y[i] = yᵢ
    end
end
function mygemvavx!(y, A, x)
    @avx for i ∈ eachindex(y)
        yᵢ = 0.0
        for j ∈ eachindex(x)
            yᵢ += A[i,j] * x[j]
        end
        y[i] = yᵢ
    end
end
A = randn(51, 49);
x = randn(49);
y1 = Vector{Float64}(undef, 51); y2 = similar(y1);
mygemv!(y1, A, x)
mygemvavx!(y2, A, x)

@test all(y1 .≈ y2)

@benchmark mygemv!($y1, $A, $x)
@benchmark mygemvavx!($y2, $A, $x)

subcolq = :(for i ∈ 1:size(A,2), j ∈ eachindex(x)
            B[j,i] = A[j,i] - x[j]
            end)
lssubcol = LoopVectorization.LoopSet(subcolq);
@test LoopVectorization.choose_order(lssubcol) == (Symbol[:j,:i], 4, -1)
LoopVectorization.lower(lssubcol)

function mysubcol!(B, A, x)
    @inbounds for i ∈ 1:size(A,2)
        @simd for j ∈ eachindex(x)
            B[j,i] = A[j,i] - x[j]
        end
    end
end
function mysubcolavx!(B, A, x)
    @avx for i ∈ 1:size(A,2), j ∈ eachindex(x)
        B[j,i] = A[j,i] - x[j]
    end
end
A = randn(199, 498); x = randn(size(A,1));
B1 = similar(A); B2 = similar(A);

mysubcol!(B1, A, x)
mysubcolavx!(B2, A, x)

@test all(B1 .≈ B2)

@benchmark mysubcol!($B1, $A, $x)
@benchmark mysubcolavx!($B2, $A, $x)

@code_native debuginfo=:none mysubcol!(B1, A, x)
@code_native debuginfo=:none mysubcolavx!(B2, A, x)


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



