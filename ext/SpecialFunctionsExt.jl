using SpecialFunctions
using LoopVectorization: VectorizationBase
@inline SpecialFunctions.erf(x::AbstractSIMD) = VectorizationBase.verf(float(x))
