using Revise

using pseudo_label_finite
using FastGaussQuadrature, Distributions
using ProgressMeter
using LinearAlgebra
using Plots


ρ = 0.5  # bias 
λ = 0.01  # regularization parameter
Δ = 0.75^2  # size of the clusters

t_max = 10000
η_d = 0.5
tol = 1.0e-15

n = 1024
n_experiment = 64

n_α = 10
α_array = range(0.5, stop=5.0, length=n_α);
q_array = zeros(n_experiment, n_α);
m_array = zeros(n_experiment, n_α);
B_array = zeros(n_experiment, n_α);

prog = Progress(n_α * n_experiment)
for i in 1:n_α
    α = α_array[i]
    mL = Int(round(n * α))

    for j in 1:n_experiment
        X, y = make_data(mL, n, ρ, Δ=Δ);
        y = 2 .* y .- 1
        ŵ = fit_squared_nobias(X, y, λ);
        q = sum(ŵ.^2)/n
        m = sum(ŵ)/n
        B = 0.0;

        q_array[j,i] = q
        m_array[j,i] = m
        B_array[j,i] = B
        next!(prog)
    end
end


qave = vec(mean(q_array, dims=1));
qerr = vec(std(q_array, dims=1))./sqrt(n_experiment);
mave = vec(mean(m_array, dims=1));
merr = vec(std(m_array, dims=1))./sqrt(n_experiment);
Bave = vec(mean(B_array, dims=1));
Berr = vec(std(B_array, dims=1))./sqrt(n_experiment);

plot(α_array, qave, yerror=qerr, label="q", seriestype=:scatter)
plot!(α_array, Bave, yerror=Berr, label="B", seriestype=:scatter)
p1 = plot!(α_array, mave, yerror=merr, label="m", seriestype=:scatter)

