module SaddlePointUtil

export StateL, StateU

struct StateL
    qhat::Float64
    chihat::Float64
    mhat::Float64
    
    q::Float64
    chi::Float64
    m::Float64
    B::Float64

    iter_num::Int64
end

# TODO: iterative版で解くバージョンの解のstructを作る（もともろ対称行列で定義することになるはず）
# struct StatePath
#     qhat::Float64
#     chihat::Float64
#     mhat::Float64
#     rhat::Float64

#     q::Float64
#     chi::Float64
#     m::Float64
#     R::Float64
#     B::Float64


#     iter_num::Int64
# end

end