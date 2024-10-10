module SupervisedSolver

include("SaddlePointUtil.jl")
using .SaddlePointUtil

export squared_noBias_rs, squared_rs

# derivative of logistic loss
function dl(u, χ, y, h)
    u /χ - y /(1 + exp(y * (u + h))) 
end

# dd logistic loss
function ddl(u, χ, h, γ=1)
    1/χ + 0.25 / cosh((u+h)*0.5)^2
end

function logistic_uhat(y, h, χ)
    u = -h
    pre_u = 9999.0
    for i in 1:30
        pre_u = u
        u = u - dl(u, χ, y, h) / ddl(u, χ, h)
        if abs(u - pre_u) < 1.0e-6
            # println("$i, $(abs(u - pre_u))")
            return u
        end
    end
    println("aho, $(abs(u - pre_u)), $y, $h, $χ")
    return u
end

function squared_rs(
    α, ρ, Δ, λ,
    t_max=100, η_d=0.5, tol=1.0e-12,
    qhat=1.0,mhat=1.0, chihat=1.0, B=1.0e-2,
    clamp_min=1.0e-14, clamp_max=1.0e14
)
    α_p = α * ρ
    α_m = α * (1-ρ)
    V = 4ρ*(1-ρ)  # the variance of the label y
    # initialization
    q = 1.0
    m = 1.0
    chi = 1.0

    diff_array = zeros(4);
    diff = 9999.0
    for t in 1:t_max
        pre_B = B
        pre_q = q
        pre_m = m
        pre_chi = chi
        # update order parameters ---------------------------------------------------------------------------
        q = (mhat^2 + chihat)/(qhat + λ)^2
        m = mhat / (qhat + λ)
        chi = 1/(qhat + λ)

        chi = clamp(chi, clamp_min, clamp_max)

        # update conjugate variables 
        qhat_temp = α * Δ / (1 + chi * Δ)
        mhat_temp = α * V * (1-m) / (1 + chi * Δ)
        chihat_temp = α * Δ / (1 + chi * Δ)^2 * (
            (1-m)^2*V + q * Δ
        )
        B_temp = (1-m) * (2ρ-1)

        qhat = η_d * qhat + (1-η_d) * qhat_temp
        mhat = η_d * mhat + (1-η_d) * mhat_temp
        chihat = η_d * chihat + (1-η_d) * chihat_temp
        B = η_d * B + (1-η_d) * B_temp

        # check convergence
        diff_array[1] = abs(pre_B - B)
        diff_array[2] = abs(pre_m - m)
        diff_array[3] = abs(pre_q - q)
        diff_array[4] = abs(pre_chi - chi)
        diff = maximum(diff_array)

        if t%20000 == 0
            println("$t, diff = $diff, $B, $λ, $chi, $(diff_array)")
        end
        if diff < tol
            # return q, chi, m, B, qhat, chihat, mhat, t
            return  StateL(
                qhat, chihat, mhat,
                q, chi, m, B, t
            )
        end
        # println("\t $B")
    end
    println("not converged, diff=$(diff), $(α_p +α_m), $(α_p), $(α_m), $λ")
    # return q, chi, m, B, qhat, chihat, mhat, t_max
    return StateL(
        qhat, chihat, mhat,
        q, chi, m, B, t_max
    )
end

# このバージョンでは結果がρに依存しない
function squared_noBias_rs(
    α, ρ, Δ, λ, 
    t_max=100, η_d=0.5, tol=1.0e-12,
    qhat=1.0,mhat=1.0, chihat=1.0, B=1.0e-2,
    clamp_min=1.0e-14, clamp_max=1.0e14
)
    # initialization
    q = 1.0
    m = 1.0
    chi = 1.0
    B = 0.0

    diff_array = zeros(3);
    diff = 9999.0
    for t in 1:t_max
        pre_q = q
        pre_m = m
        pre_chi = chi
        # update order parameters ---------------------------------------------------------------------------
        q = (mhat^2 + chihat)/(qhat + λ)^2
        m = mhat / (qhat + λ)
        chi = 1/(qhat + λ)

        chi = clamp(chi, clamp_min, clamp_max)

        # update conjugate variables 
        qhat_temp = α * Δ / (1 + chi * Δ)
        mhat_temp = α * (1-m) / (1 + chi * Δ)
        chihat_temp = α * Δ / (1 + chi * Δ)^2 * (
            (1-m)^2 + q * Δ
        )

        qhat = η_d * qhat + (1-η_d) * qhat_temp
        mhat = η_d * mhat + (1-η_d) * mhat_temp
        chihat = η_d * chihat + (1-η_d) * chihat_temp

        # check convergence
        diff_array[1] = abs(pre_m - m)
        diff_array[2] = abs(pre_q - q)
        diff_array[3] = abs(pre_chi - chi)
        diff = maximum(diff_array)

        if t%20000 == 0
            println("$t, diff = $diff, $B, $λ, $chi, $(diff_array)")
        end
        if diff < tol
            # return q, chi, m, B, qhat, chihat, mhat, t
            return  StateL(
                qhat, chihat, mhat,
                q, chi, m, B, t
            )
        end
        # println("\t $B")
    end
    println("not converged, diff=$(diff), $(ρ), $(α), $λ")
    # return q, chi, m, B, qhat, chihat, mhat, t_max
    return StateL(
        qhat, chihat, mhat,
        q, chi, m, B, t_max
    )
end

end  # module SupervisedSolver