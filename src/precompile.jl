
using SnoopPrecompile

@precompile_setup begin
  x = rand(10)
  @precompile_all_calls begin
    _vreduce(+, x)
  end
end
