


struct ArrayReference
    loops::ByteVector # 0s mean not loops / loopindex = false
    staticoffsets::ByteVector
    strideorder::ByteVector
    isdense::ByteVector
    staticmultipliers::WordVector
    dynamicoffsets::Vector{Symbol}
    dynamicmultipliers::Vector{Symbol}
    name::Symbol
end



