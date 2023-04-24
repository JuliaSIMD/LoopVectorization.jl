
using PrecompileTools

@setup_workload begin
  x = rand(10)
  @compile_workload begin
    _vreduce(+, x)
  end
end
