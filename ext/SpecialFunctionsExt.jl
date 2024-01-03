module SpecialFunctionsExt
if VERSION < v"1.11-DEV"
using SpecialFunctions
using LoopVectorization: VectorizationBase
using LoopVectorization: AbstractSIMD
@inline SpecialFunctions.erf(x::AbstractSIMD) = VectorizationBase.verf(float(x))
end
end
