using Revise

using pseudo_label_finite
using FastGaussQuadrature, Distributions
using ProgressMeter
using LinearAlgebra
using Plots


function normal_cdf(x::Float64)
    cdf(Normal(0.0, 1.0), x)
end

function normal_sf(x::Float64)
    1.0 - cdf(Normal(0.0, 1.0), x)
end

function calc_gen_err(ρ, Δ, b, m, q; eps=0.0)
    ρ * normal_cdf(
        -(b + m)/((q * Δ)^0.5 + eps)
    ) + (1.0 - ρ) * normal_sf(
        (m-b) / ((q * Δ)^0.5 + eps)
    )
end

α = 0.5  # number of samples
ρ = 0.5  # bias （当分1/2しかやらない）
λ = 0.01  # regularization parameter
Δ = 0.75^2  # size of the clusters

t_max = 10000
η_d = 0.5
tol = eps(Float64) * 4.0

n_α_rs = 1000
α_array_rs = range(0.5, stop=5.0, length=n_α_rs);
q_array_rs = zeros(n_α_rs);
m_array_rs = zeros(n_α_rs);
B_array_rs = zeros(n_α_rs);
egen_array_rs = zeros(n_α_rs);

prog = Progress(n_α_rs)
@time for i in 1:n_α_rs
    α = α_array_rs[i]

    stateL = squared_noBias_rs(
        α, ρ, Δ, λ,
        t_max, η_d, tol
    );

    egen = calc_gen_err(ρ, Δ, stateL.B, stateL.m, stateL.q)

    q_array_rs[i] = stateL.q
    m_array_rs[i] = stateL.m
    egen_array_rs[i] = egen
    B_array_rs[i] = stateL.B
    next!(prog)
end

# p = plot()
plot!(α_array_rs, q_array_rs, label="q")
plot!(α_array_rs, m_array_rs, label="m")
p1 = plot!(α_array_rs, B_array_rs, label="B")


