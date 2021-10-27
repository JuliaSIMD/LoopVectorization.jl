
using LoopVectorization, Test

function reference_mul4!(target_arr, src, range_a, range_b, padded_axis_a, padded_axis_b)
  @inbounds @fastmath for a1i ∈ eachindex(range_a), 
    a2i ∈ eachindex(range_a), 
    b1i ∈ eachindex(range_b), 
    b2i ∈ eachindex(range_b)
    a1 = range_a[a1i]; 
    a2 = range_a[a2i] 
    b1 = range_b[b1i]; b2 = range_b[b2i]
    contribution = zero(eltype(target_arr))
    for i_a ∈ padded_axis_a, i_b ∈ padded_axis_b
      contribution += src[i_a, i_b] * src[i_a+a1,i_b+b1] * src[i_a+a2,i_b+b2]
    end
    target_arr[b1i, b2i] += contribution
  end  
end
function mul4_turbo_v1!(target_arr, src, range_a, range_b, padded_axis_a, padded_axis_b)
  @turbo for a1i ∈ eachindex(range_a), 
    a2i ∈ eachindex(range_a), 
    b1i ∈ eachindex(range_b), 
    b2i ∈ eachindex(range_b)
    a1 = range_a[a1i]; 
    a2 = range_a[a2i] 
    b1 = range_b[b1i]; b2 = range_b[b2i]
    contribution = zero(eltype(target_arr))
    for i_a ∈ padded_axis_a, i_b ∈ padded_axis_b
      contribution += src[i_a, i_b] * src[i_a+a1,i_b+b1] * src[i_a+a2,i_b+b2]
    end
    target_arr[b1i, b2i] += contribution
  end
  target_arr
end
function mul4_turbo_v2!(target_arr, src, range_a, range_b, padded_axis_a, padded_axis_b)
  @turbo for a1i ∈ eachindex(range_a), 
    a2i ∈ eachindex(range_a), 
    b1i ∈ eachindex(range_b), 
    b2i ∈ eachindex(range_b)
    a1 = range_a[a1i]; 
    a2 = range_a[a2i] 
    b1 = range_b[b1i]; b2 = range_b[b2i]
    contribution = target_arr[b1i, b2i]
    for i_a ∈ padded_axis_a, i_b ∈ padded_axis_b
      contribution += src[i_a, i_b] * src[i_a+a1,i_b+b1] * src[i_a+a2,i_b+b2]
    end
    target_arr[b1i, b2i] = contribution
  end
end
function mul4_turbo_v3!(target_arr, src, range_a, range_b, padded_axis_a, padded_axis_b)
  @turbo for a1i ∈ eachindex(range_a), 
    a2i ∈ eachindex(range_a), 
    b1i ∈ eachindex(range_b), 
    b2i ∈ eachindex(range_b),
    i_a ∈ padded_axis_a,
    i_b ∈ padded_axis_b
    a1 = range_a[a1i]; 
    a2 = range_a[a2i] 
    b1 = range_b[b1i]; b2 = range_b[b2i]
    target_arr[b1i, b2i] += src[i_a, i_b] * src[i_a+a1,i_b+b1] * src[i_a+a2,i_b+b2]
  end
end
function mul4_turbo_v4!(target_arr, src, range_a, range_b, padded_axis_a, padded_axis_b)
  @turbo for  b1i ∈ eachindex(range_b), b2i ∈ eachindex(range_b)
    b1 = range_b[b1i]; b2 = range_b[b2i]
    contribution = zero(eltype(target_arr))
    for i_a ∈ padded_axis_a, i_b ∈ padded_axis_b, a1i ∈ eachindex(range_a), a2i ∈ eachindex(range_a)
      a1 = range_a[a1i]; a2 = range_a[a2i] 
      contribution += src[i_a, i_b] * src[i_a+a1,i_b+b1] * src[i_a+a2,i_b+b2]
    end
    target_arr[b1i, b2i] += contribution
  end
end

@testset "Inner reductions" begin
  src = ones(19, 101);
  max_a = 7; max_b = 9;
  range_a = -max_a:max_a;
  range_b = -max_b:max_b;

  target_arr = zeros(Float64, length(range_b), length(range_b));

  padded_axis_b = (first(axes(src,2)) .+ max_b):(last(axes(src,2)) - max_b);
  padded_axis_a = (first(axes(src,1)) .+ max_a):(last(axes(src,1)) - max_a);

  target_ref = zero(target_arr);
  reference_mul4!(target_ref, src, range_a, range_b, padded_axis_a, padded_axis_b)

  mul4_turbo_v1!(target_arr, src, range_a, range_b, padded_axis_a, padded_axis_b)
  @test target_arr ≈ target_ref
  target_arr .= 0;
  mul4_turbo_v2!(target_arr, src, range_a, range_b, padded_axis_a, padded_axis_b)
  @test target_arr ≈ target_ref
  target_arr .= 0;
  mul4_turbo_v3!(target_arr, src, range_a, range_b, padded_axis_a, padded_axis_b)
  @test target_arr ≈ target_ref
  target_arr .= 0;
  mul4_turbo_v4!(target_arr, src, range_a, range_b, padded_axis_a, padded_axis_b)
  @test target_arr ≈ target_ref

end

