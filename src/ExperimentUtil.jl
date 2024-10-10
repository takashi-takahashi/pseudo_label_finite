module ExperimentUtil

using Optim
using Distributions
using LinearAlgebra

export make_data, fit_squared, fit_squared_nobias


function calc_squared_loss(w, y, Xp, λ)
    ŷ = Xp * w
    sum(
        (y .- ŷ).^2
    ) .+ 0.5 .* λ .* sum(w[2:end].^2.0)
end

function calc_squared_dloss(w, y, Xp, λ_v)
    ŷ = Xp * w
    return Xp' * (ŷ .- y) + λ_v .* w
end


# この場合、yは0,1ではなくて、-1,1であることを仮定している
function fit_squared(X, y, λ)
    @assert size(X)[1] == size(y)[1] "size mismatch size(X)=$(size(X)), size(y)=$(size(y))"
    @assert length(size(y)) == 1 "dimension mismatch size(y)=$(size(y))"
    m, n = size(X);
    Xp = hcat(ones(m), X);
    λ_v = ones(n+1) .* λ;
    λ_v[1] = 0;

    l(x) = calc_squared_loss(x, y, Xp, λ);
    dl(x) = calc_squared_dloss(x, y, Xp, λ_v);
    x0 = ones(n+1)
    x0[1] = 0.0
    result = optimize(l, dl, x0, LBFGS(), inplace=false);

    return Optim.minimizer(result);  # includes the intercept
end

# この場合、yは0,1ではなくて、-1,1であることを仮定している
function fit_squared_nobias(X, y, λ)
    @assert size(X)[1] == size(y)[1] "size mismatch size(X)=$(size(X)), size(y)=$(size(y))"
    @assert length(size(y)) == 1 "dimension mismatch size(y)=$(size(y))"
    m, n = size(X);
    λ_v = ones(n) .* λ;

    l(x) = calc_squared_loss(x, y, X, λ);
    dl(x) = calc_squared_dloss(x, y, X, λ_v);
    x0 = ones(n)
    result = optimize(l, dl, x0, LBFGS(), inplace=false);

    return Optim.minimizer(result);  # includes the intercept
end

function make_data(m, n, ρ ; Δ=1.0)
    det_index = Int(round(m * ρ))
    y = zeros(Int64, m);
    for i in 1:det_index
        y[i] = 1
    end
    # y = rand(Binomial(1, ρ), m);
    ϵ = randn((m, n));
    X = ones(m, n) .* (2.0 .* y .- 1.0) ./ n .+ Δ.^0.5 .* ϵ ./ n^0.5;
    return X, y
end


end 