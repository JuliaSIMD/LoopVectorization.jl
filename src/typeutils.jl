# @nospecialize  # don't specialize anything until @specialize

"""
    params = abstractparameters(T::Type, AT::Type)

For a type `T` which is a subtype of `AT{params...}`, return `params`.
The purpose of this function is to allow one to use `@nospecialize` on type-arguments
and still extract the parameters of the corresponding abstract type.
"""
function abstractparameters(T::Type, AT::Type)
    @nospecialize T AT
    @assert T <: AT
    Tst = supertype(T)
    while Tst <: AT
        T = Tst
        Tst = supertype(T)
    end
    return T.parameters
end

# @specialize
