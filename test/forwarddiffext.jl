using Base: Forward

using NNlib, LoopVectorization, VectorizationBase, ForwardDiff, Test
randnvec() = Vec(ntuple(_ -> randn(), pick_vector_width(Float64))...)

tovec(x::Vec{W,T}) where {W,T} = T[Tuple(x)...]
tovec(x::VecUnroll) = reduce(vcat, map(tovec, VectorizationBase.data(x)))
function tovec(x::ForwardDiff.Dual{T,V,N}) where {T,V,N}
  v = tovec(ForwardDiff.value(x))
  dv = map(tovec, Tuple(ForwardDiff.partials(x)))
  D = ForwardDiff.Dual{T,eltype(v),N}
  ret = Vector{D}(undef, length(v))
  for i in eachindex(v)
    ret[i] = ForwardDiff.Dual(v[i], map(Base.Fix2(Base.getindex, i), dv)...)
  end
  return ret
end

vx0 = randnvec()
vx1 = randnvec()
vx2 = randnvec()
vx3 = randnvec()
vx4 = randnvec()
vx5 = randnvec()

vd0 = ForwardDiff.Dual(vx0, vx1, vx2, vx3, vx4, vx5)

vu0 = VecUnroll((vx0, vx1))
vu1 = VecUnroll((vx2, vx3))
vu2 = VecUnroll((vx4, vx5))

vud = ForwardDiff.Dual(vu0, vu1, vu2)


@test reinterpret(Float64, tovec(VectorizationBase.relu(vd0))) ≈
      reinterpret(Float64, VectorizationBase.relu.(tovec(vd0)))
@test reinterpret(Float64, tovec(VectorizationBase.relu(vud))) ≈
      reinterpret(Float64, VectorizationBase.relu.(tovec(vud)))

@test reinterpret(Float64, tovec(VectorizationBase.leakyrelu(vd0))) ≈
      reinterpret(Float64, VectorizationBase.leakyrelu.(tovec(vd0)))
@test reinterpret(Float64, tovec(VectorizationBase.leakyrelu(vud))) ≈
      reinterpret(Float64, VectorizationBase.leakyrelu.(tovec(vud)))
