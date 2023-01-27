using SpecialFunctions
@inline SpecialFunctions.erf(x::AbstractSIMD) = VectorizationBase.verf(float(x))
