


# Auxiliary functions
const _uint_bit_length = sizeof(UInt) * 8
const _div_uint_size_shift = Int(log2(_uint_bit_length))
@inline _mul2(i::Integer) = i << 1
@inline _div2(i::Integer) = i >> 1
@inline _map_to_index(i::Integer) = _div2(i - 1)
@inline _map_to_factor(i::Integer) = _mul2(i) + 1
@inline _mod_uint_size(i::Integer) = i & (_uint_bit_length - 1)
@inline _div_uint_size(i::Integer) = i >> _div_uint_size_shift
@inline _get_chunk_index(i::Integer) = _div_uint_size(i + (_uint_bit_length - 1))
@inline _get_bit_index_mask(i::Integer) = UInt(1) << _mod_uint_size(i - 1)


function clear_factors_while!(arr::Vector{UInt}, factor_index::Integer, max_index::Integer)
  factor = _map_to_factor(factor_index)
  index = _div2(factor * factor)
  while index <= max_index
    @inbounds arr[_get_chunk_index(index)] |= _get_bit_index_mask(index)
    index += factor
  end
  return arr
end
function clear_factors_turbo!(arr::Vector{UInt}, factor_index::Integer, max_index::Integer)
  factor = _map_to_factor(factor_index)
  factor < _uint_bit_length &&
    error("Factor must be greater than UInt bit length to avoid memory pendencies")
  @turbo for index = _div2(factor * factor):factor:max_index
    @inbounds arr[(index+63)>>_div_uint_size_shift] |= 1 << ((index - 1) & 63)
  end
  return arr
end
function clear_factors_turbo_u4!(
  arr::Vector{UInt},
  factor_index::Integer,
  max_index::Integer,
)
  factor = _map_to_factor(factor_index)
  factor < _uint_bit_length &&
    error("Factor must be greater than UInt bit length to avoid memory pendencies")
  @turbo unroll = 4 for index = _div2(factor * factor):factor:max_index
    @inbounds arr[(index+63)>>_div_uint_size_shift] |= 1 << ((index - 1) & 63)
  end
  return arr
end

@testset "steprange" begin
  x0 = rand(UInt, cld(500_000, sizeof(UInt) * 8))
  x1 = copy(x0)
  x2 = copy(x0)
  clear_factors_while!(x0, 202, 500_000)
  clear_factors_turbo!(x1, 202, 500_000)
  clear_factors_turbo_u4!(x2, 202, 500_000)
  @test x0 == x1 == x2
end
