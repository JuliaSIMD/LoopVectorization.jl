module SpecialFunctionsExt
using SpecialFunctions
using LoopVectorization: VectorizationBase
using LoopVectorization: AbstractSIMD
@inline SpecialFunctions.erf(x::AbstractSIMD) = VectorizationBase.verf(float(x))
end
