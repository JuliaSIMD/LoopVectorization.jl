
@enum Unrolled::Int8 NotUnrolled=0 ParallelUnrolled=1 TemporalUnrolled=2
struct Schedule
    statements::Vector{Pair{Int32,Int32}}
    nested::Vector{Schedule}
    vectorized::Bool
    unrolled::Unrolled
end

isunrolled(s::Schedule) = s.u !== NotUnrolled
isu₁unrolled(s::Schedule) = s.u === ParallelUnrolled
isu₂unrolled(s::Schedule) = s.u === TemporalUnrolled



