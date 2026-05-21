using Distributions

abstract type DistGenerator end

@kwdef struct LikertGen{T<:ContinuousMultivariateDistribution} <: DistGenerator
    μ_M::Float64 = 0.0
    σ²_M::Float64 = 1.0
    σ²_X::Float64 = 25.0
    α::Float64 = 3.0
    β::Float64 = 1.0
    distType::Type{T} = LikertDist
end


@kwdef mutable struct LikertDist  <: ContinuousMultivariateDistribution
    M_Q_D::Matrix{Float64}
    shift_Q::Vector{Float64}
    σ²_D::Vector{Float64}
    _tmp_vec::Vector{Float64}
    _μ_vec::Vector{Float64}
end

function calc_μ!(dist::LikertDist, z::AbstractVector{<:Integer}) 
    tmp = dist._tmp_vec
    μ   = dist._μ_vec

    @simd for i in eachindex(z)
        tmp[i] = z[i] - dist.shift_Q[i]
    end
    tmp[length(z) + 1] = 1.0

    mul!(μ, dist.M_Q_D', tmp)
end


function calc_μ!(dist::LikertDist, z::CartesianIndex)
    tmp = dist._tmp_vec
    μ   = dist._μ_vec
    @simd for i in 1:length(z.I)
        tmp[i] = z.I[i] - dist.shift_Q[i]
    end
    tmp[length(z.I)+1] = 1.0
    mul!(μ, dist.M_Q_D', tmp)
end

function Distributions.rand(dist::LikertDist, z::Vector{<:Integer})
    calc_μ!(dist, z)
    μ  = dist._μ_vec
    σ² = dist.σ²_D

    out = similar(μ)
    @simd for i in eachindex(μ)
        out[i] = μ[i] + sqrt(σ²[i]) * randn()
    end
    return out
end


function Distributions.logpdf(dist::LikertDist, z, x::AbstractVector{Float64})
    calc_μ!(dist, z)
    μ    = dist._μ_vec
    σ²   = dist.σ²_D
    # Inline logpdf calculation
    s = 0.0
    for i in eachindex(x)
        invσ = 1 / sqrt(σ²[i])
        diff = (x[i] - μ[i]) * invσ
        s += -0.5 * (log(2π) + log(σ²[i])) - 0.5 * diff^2
    end

    return s
end

function logpdf_un_normalized(dist::LikertDist, z, x::AbstractVector{Float64})
    calc_μ!(dist, z)
    μ    = dist._μ_vec
    σ²   = dist.σ²_D
    # Inline logpdf calculation
    s = 0.0
    for i in eachindex(x)
        s += -(x[i] - μ[i])^2 / (2* σ²[i])
    end
    return s
end


struct CenterCutout <: ContinuousUnivariateDistribution
    base::UnivariateDistribution
    l::Float64
    u::Float64
end


function Distributions.rand(d::CenterCutout)
    log_p_left = logcdf(d.base, d.l)
    log_p_right = logccdf(d.base, d.u)

    p_left =  exp(log_p_left - logsumexp(log_p_left,log_p_right))
    if rand(Bernoulli(p_left)) 
        # Sample from (-∞, l]
        return rand(Truncated(d.base, -Inf, d.l))
    else
        # Sample from [u, ∞)
        return rand(Truncated(d.base, d.u, Inf))
    end
end

function PG_post(X_E_P, Θ_P_K ,Z_E_K, N_E, k,Λ_P_K_K;ε = 1e-8 )
    P,K = size(Θ_P_K)
    E = length(N_E)

    @assert size(X_E_P) == (E,P)
    @assert size(Z_E_K) == (E,K)

    ϕ_E_K = X_E_P * Θ_P_K
    ind = [i for i  in 1:K if i != k]

    ζ_E = [logsumexp(ϕ_E_K[e,ind]) for e in 1:E]
    ρ_E = ϕ_E_K[:,k] .- ζ_E

    ω_E = rand.(PolyaGammaHybridSampler.(N_E,ρ_E))
    κ_E = Z_E_K[:,k] .- N_E./2

    μ_0 = [-Θ_P_K[p, ind]' * Λ_P_K_K[p][ind, k] / Λ_P_K_K[p][k, k] for p in 1:P]
    Λ_0 = Diagonal([Λ_P_K_K[p][k, k] for p in 1:P]) 



    A = X_E_P' * (Diagonal(ω_E) * X_E_P) + Matrix(Λ_0)
    A = Symmetric(A + ε*LinearAlgebra.I)
    b = X_E_P' * (κ_E .+ ω_E .* ζ_E) + Λ_0 * μ_0

    F = cholesky(A; check=true)   
    μ_post = F \ b
    Σ_post = Symmetric(inv(F))   

    return (μ_post,Σ_post)
end



@inline function flat_idx(z, K)
    idx = 0
    stride = 1
    for j in 1:length(z)
        idx += (z[j] - 1) * stride
        stride *= K[j]
    end
    return idx + 1
end

@inline function un_flat_idx!(z,lin_ind, K)
    rem = lin_ind - 1
    for d in 1:length(K)
        z[d] = (rem % K[d]) + 1
        rem = div(rem, K[d])
    end
end

@inline function flat_idx(e, i, PA, K_pa, Z)
    idx = 0
    stride = 1
    for j in 1:length(PA)
        idx += (Z[e][i, PA[j]] - 1) * stride
        stride *= K_pa[j]
    end
    return idx + 1
end
