"""
    isscopedname(ex, modpath, name::Symbol)

Test whether expression `ex` is a module-scoped `name`. Both of these return `true`:

```
isscopedname(:(Base.OneTo), :Base, :OneTo)
isscopedname(:(Base.Checked.checked_add), (:Base, :Checked), :checked_add)
```
"""
function isscopedname(ex, modpath, name::Symbol)
  isexpr(ex, :(.), 2) &&
    (a = ex.args[2]; isa(a, QuoteNode) && a.value === name) &&
    hasscope(ex.args[1], modpath)
end
hasscope(modex, mod::Symbol) = modex === mod
hasscope(modex, mod::Tuple{Symbol}) = hasscope(modex, mod[1])
hasscope(modex, modpath::Tuple{Vararg{Symbol}}) =
  isscopedname(modex, Base.front(modpath), modpath[end])

"""
    isglobalref(g, mod, name)

Return true if `g` is equal to `GlobalRef(mod, name)`.
"""
isglobalref(g, mod, name) = isa(g, GlobalRef) && g.mod === mod && g.name === name
