function extract_all_lnns(x)
  lnns = Vector{LineNumberNode}(undef, 0)
  return extract_all_lnns!(lnns, x)
end

function extract_all_lnns!(
  lnns::AbstractVector{<:LineNumberNode},
  lnn::LineNumberNode
)
  push!(lnns, lnn)
  return lnns
end
function extract_all_lnns!(lnns::AbstractVector{<:LineNumberNode}, ex::Expr)
  for arg in ex.args
    extract_all_lnns!(lnns::AbstractVector{<:LineNumberNode}, arg)
  end
  return lnns
end
extract_all_lnns!(lnns::AbstractVector{<:LineNumberNode}, ::Any) = lnns

function prepend_lnns!(ex::Expr, lnns::AbstractVector{<:LineNumberNode})
  return prepend_lnns!(ex, lnns, Val(ex.head))
end
function prepend_lnns!(
  ex::Expr,
  lnns::AbstractVector{<:LineNumberNode},
  ::Val{:block}
)
  for lnn in lnns
    pushfirst!(ex.args, Expr(:block, lnn, :(nothing)))
  end
  return ex
end
