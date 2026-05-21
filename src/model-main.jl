using Distributions
using StatsBase
using LinearAlgebra
using PolyaGammaHybridSamplers
using LogExpFunctions


include("samplers.jl")
include("model-utils.jl")


@kwdef mutable struct CausalState{T<:ContinuousMultivariateDistribution} <: AbstractState
    c_Q::Array{Float64,1}
    I_Q_E::Array{Bool,2}
    Θ_Q_PA_K::Array{Array{Float64},1}
    Δ_Q_E_PA_K::Array{Array{Float64},2}
    Z_E_Nₑ_Q::Array{Array{Int8,2},1 }
    X_dist::T
    X_E_Nₑ_D::Array{Array{Float64,2},1 }
end

function copy_without_X_dist(state::CausalState)
    return CausalState(
        c_Q            = deepcopy(state.c_Q),
        I_Q_E          = deepcopy(state.I_Q_E),
        Θ_Q_PA_K       = deepcopy(state.Θ_Q_PA_K),
        Δ_Q_E_PA_K     = deepcopy(state.Δ_Q_E_PA_K),
        Z_E_Nₑ_Q       = deepcopy(state.Z_E_Nₑ_Q),
        X_dist         = deepcopy(state.X_dist),  # shallow copy (reference)
        X_E_Nₑ_D       = state.X_E_Nₑ_D
    )
end

@kwdef mutable struct CausalModel_Partial_Temper{DG <: DistGenerator} <: AbstractModel
    state::Union{Nothing,CausalState} = nothing
    G::Array{Array{Int,1},1}                           #Causal Graph as list of parents
    K_Q::Array{Int8,1}                                  #Size of each latent classes
    Q::Int                                             #num latent classes
    E::Int                                             #num environments
    D::Int                                             #Data dimention
    N_E::Array{Int,1}                                  #num points per environment
    α_p::Float64 = 2                                   #hyperparams
    β_p::Float64 = 6
    gen::DG
    KL_Q::Array{Float64,1}
    Σ_Θ_Q::Array{Matrix{Float64},1}
    Σ_Δ_Q::Array{Matrix{Float64},1}
    T::Float64 = 1.0
end


function copy_without_X_dist(m::CausalModel_Partial_Temper{DG}) where {DG<:DistGenerator}
    new_state =  isnothing(m.state) ? nothing : copy_without_X_dist(m.state)
    return CausalModel_Partial_Temper{DG}(
        state = new_state,
        G     = deepcopy(m.G),
        K_Q   = deepcopy(m.K_Q),
        Q     = m.Q,
        E     = m.E,
        D     = m.D,
        N_E   = deepcopy(m.N_E),
        α_p   = m.α_p,
        β_p   = m.β_p,
        gen   = deepcopy(m.gen),      # change to `m.gen` if you want to share it
        KL_Q  = deepcopy(m.KL_Q),
        Σ_Θ_Q = deepcopy(m.Σ_Θ_Q),
        Σ_Δ_Q = deepcopy(m.Σ_Δ_Q),
        T     = m.T
    )
end

deep_copy(m::CausalModel_Partial_Temper{DG}) where {DG<:DistGenerator} = copy_without_X_dist(m)

deep_copy(s::CausalState{T}) where {T<:ContinuousMultivariateDistribution} = copy_without_X_dist(s)


stateType(::CausalModel_Partial_Temper) = CausalState

CausalState(model::CausalModel_Partial_Temper) = begin
    
    state = CausalState(;
        c_Q         = Array{Float64,1}(undef,model.Q),
        I_Q_E       = Array{Bool,2}(undef,model.Q,model.E),
        Θ_Q_PA_K    = Array{Array{Float64},1}(undef,model.Q),
        Δ_Q_E_PA_K  = Array{Array{Float64},2}(undef,model.Q,model.E),
        Z_E_Nₑ_Q    = Array{Array{Int8,2},1 }(undef,model.E),
        X_dist      = allocDist(model),
        X_E_Nₑ_D    = Array{Array{Float64,2},1 }(undef,model.E)
    )

    for q in 1:model.Q
        dims = tuple(model.K_Q[model.G[q]]...,model.K_Q[q] )

        state.Θ_Q_PA_K[q] = Array{Float64}(undef,dims)
        for e in 1:model.E
            state.Δ_Q_E_PA_K[q,e]  = Array{Float64}(undef,dims)
        end
    end
    for e in 1:model.E
        state.Z_E_Nₑ_Q[e] = Array{Int8,2}(undef,model.N_E[e],model.Q)
        state.X_E_Nₑ_D[e] = Array{Float64,2}(undef,model.N_E[e],model.D)
    end
    return state
end

allocDist(model::CausalModel_Partial_Temper) = model.gen.distType(model)

function LikertDist(model::CausalModel_Partial_Temper{LikertGen})
    M_Q_D = Array{Float64,2}(undef,tuple(model.Q+1,model.D))
    σ²_D = Array{Float64,1}(undef,model.D)
    shift_Q = Array{Float64,1}(undef,model.Q)
    tmp_vec = zeros(Float64, model.Q +1)
    μ_vec   = zeros(Float64, model.D)
    return LikertDist(;M_Q_D = M_Q_D,σ²_D = σ²_D,shift_Q=shift_Q,_tmp_vec = tmp_vec,_μ_vec = μ_vec )
end

modelToSample(model::CausalModel_Partial_Temper{LikertGen}) = copy_without_X_dist(model.state)
model_to_sample(model::CausalModel_Partial_Temper,::SMCS) = copy_without_X_dist(model)
#temporary stubs until funcitons re-factored for new design
generateParam!(param::Val,model::CausalModel_Partial_Temper)  = generateParam!(param,model,model.state)
updateParam!(param::Val,model::CausalModel_Partial_Temper)  = updateParam!(param,model,model.state)


#initalize paramters from priors:
generateParam!(::Val{:c_Q},model::CausalModel_Partial_Temper,state::CausalState) = begin
    state.c_Q = rand(Beta(model.α_p,model.β_p), model.Q)
end

generateParam!(::Val{:I_Q_E},model::CausalModel_Partial_Temper,state::CausalState) = begin
    for q in 1:model.Q
        state.I_Q_E[q,:] = rand(Bernoulli(state.c_Q[q]), model.E)
    end
end

generateParam!(::Val{:Θ_Q_PA_K},model::CausalModel_Partial_Temper,state::CausalState) = begin  
    for q in 1:model.Q
        parentItter = Iterators.product((1:k for k in model.K_Q[model.G[q]])...)
        μ = zeros(model.K_Q[q])
        Σ = model.Σ_Θ_Q[q]
        for PA in parentItter
            state.Θ_Q_PA_K[q][PA...,:] .= rand(MvNormal(μ,Σ))
        end
    end
end

function get_H(model::CausalModel_Partial_Temper,q)
    p = fill(1/model.K_Q[q], model.K_Q[q])
    return diagm(p) - p*p'
end
function KL_norm(H::AbstractMatrix, Δ::AbstractVector)
    return (Δ' * H * Δ ) / 2
end

generateParam!(::Val{:Δ_Q_E_PA_K},model::CausalModel_Partial_Temper,state::CausalState) = begin  
    for e in 1:model.E
        for q in 1:model.Q
            parentItter = Iterators.product((1:k for k in model.K_Q[model.G[q]])...)
            μ = zeros(model.K_Q[q])
            Σ = model.Σ_Δ_Q[q]
            L = cholesky(Σ).L 
            K = model.K_Q[q]
            dist = Normal()
            for PA in parentItter
                H = get_H(model,q)
                z = rand(dist, K)
                Δ = L * z
                samples = 0
                while KL_norm(H,Δ) <= model.KL_Q[q]
                    z = rand(dist, K)
                    Δ = L * z
                    if samples==10000
                        @warn "prior truncation slow"
                    end
                    samples +=1
                end
                state.Δ_Q_E_PA_K[q,e][PA...,:] .= Δ
            end
        end
    end
end


generateParam!(::Val{:X_dist}, model::CausalModel_Partial_Temper{LikertGen}, state::CausalState) = begin
    gen = model.gen
    state.X_dist.M_Q_D .= rand(Normal(gen.μ_M,sqrt(gen.σ²_M)), tuple(model.Q+1,model.D))

    state.X_dist.σ²_D .=  model.T .* clamp.(rand(InverseGamma(gen.α,gen.β),model.D),0,gen.σ²_X)
    #state.X_dist.σ²_D .= fill(model.T,model.D)
    state.X_dist.shift_Q .= (model.K_Q.+1)./2

end

generateParam!(v::Val{:X_E_Nₑ_D}, m::CausalModel_Partial_Temper, s::CausalState) = begin
    updateParam!(v,m,s)
end

generateParam!(::Val{:Z_E_Nₑ_Q},model::CausalModel_Partial_Temper,state::CausalState) = begin
    for e in 1:model.E
        for q in 1:model.Q
            Θ_PA_Z = state.Θ_Q_PA_K[q] .+ (state.I_Q_E[q,e] ? state.Δ_Q_E_PA_K[q,e] : 0)
            for i in 1:model.N_E[e]
                PA = model.G[q]
                z_PA = state.Z_E_Nₑ_Q[e][i,PA]
                p = softmax(Θ_PA_Z[z_PA...,:]./model.T)
                state.Z_E_Nₑ_Q[e][i,q] = rand(Categorical(p))          
            end
        end
    end
end



updateParam!(::Val{:c_Q},model::CausalModel_Partial_Temper,state::CausalState) = begin
    α_Q = vec(model.α_p .+ sum(state.I_Q_E; dims=2))
    β_Q = vec(model.β_p .+ sum(.!state.I_Q_E; dims=2))
    state.c_Q .= rand.(Beta.(α_Q, β_Q))
end

updateParam!(::Val{:I_Q_E},model::CausalModel_Partial_Temper,state::CausalState) = begin

    for q in 1:model.Q
    
        PA = model.G[q]
        Kq = model.K_Q[q]
        K_pa = model.K_Q[PA]
        Θ = state.Θ_Q_PA_K[q]
        idx = zeros(Int,Kq)
        for e in 1:model.E
            Δ = state.Δ_Q_E_PA_K[q,e]
            
            logp0 = log1p(-state.c_Q[q])
            logp1 = log(state.c_Q[q])
            for i in 1:model.N_E[e]
                z = state.Z_E_Nₑ_Q[e][i,q]
 
                pe = flat_idx(e, i, PA, K_pa, state.Z_E_Nₑ_Q)
                idx .= (0:Kq-1).*prod(K_pa) .+ pe
                Θ_K = Θ[idx]
                Δ_K = Δ[idx]

                #=
                z_PA = state.Z_E_Nₑ_Q[e][i,PA]
                @assert Θ_K == Θ[z_PA..., :]
                @assert Δ_K == Δ[z_PA..., :]
                =#

                Θ_val = Θ_K[z] / model.T
                Δ_val = Δ_K[z]/ model.T  # only used for logp1



                ls_Θ_K = logsumexp(Θ_K ./ model.T)
                ls_ΘΔ_K = logsumexp((Θ_K .+ Δ_K) ./ model.T)

                logp0 += Θ_val - ls_Θ_K
                logp1 += Θ_val + Δ_val - ls_ΘΔ_K
                if !isfinite(logp1)
                    @show Δ_K,Δ[z_PA...,:], state.Δ_Q_E_PA_K
                end    

            end
            p1 =  exp(logp1 - logsumexp(logp0,logp1 ))
            state.I_Q_E[q,e] = rand(Bernoulli( p1 ))

        end   
    end 
end


function quadratic_solve(a, b, c;eps = 1e-10#=,H,θ,Δ,Δ_all,T,μ_post,Σ_post,μ_0,Σ_0=#)
    @assert a >= 0
    if a == 0.0
        if abs(b) < eps
            if c <= 0
                l,u =  (-Inf, Inf)  
            else
                l,u =  (0.0, 0.0) 
            end
        else
            x_cut = -c / b
            if b > 0
                l,u =  -Inf, x_cut
            else
                l,u =  x_cut, Inf
            end
        end
    else
        discriminant = b^2 - 4a*c
        if discriminant < 0
            l,u =  (0.0, 0.0) 
        else
            l = (-b - sqrt(discriminant)) / (2a)
            u = (-b + sqrt(discriminant)) / (2a)
        end
    end     
    return (l,u)
end




updateParam!(::Val{:Θ_Q_PA_K},model::CausalModel_Partial_Temper,state::CausalState) = begin  
    for q in 1:model.Q
        parentItter = Iterators.product((1:k for k in model.K_Q[model.G[q]])...)

        PA = model.G[q]
        Kq = model.K_Q[q]
        K_pa = model.K_Q[PA]
        n_parent_configs = prod(K_pa)
        
        counts_PA_K_E_flat = zeros(Int, n_parent_configs, Kq, model.E)

        for e in 1:model.E
            for i in 1:model.N_E[e]
                pe = flat_idx(e, i, PA, K_pa, state.Z_E_Nₑ_Q)
                z  = state.Z_E_Nₑ_Q[e][i, q]
                counts_PA_K_E_flat[pe, z, e] +=1
            end
        end

        X = zeros(model.E,model.E+1)
        X[:,1] .= 1
        for e in 1:model.E
            X[e,e+1] = state.I_Q_E[q,e]
        end
        X./=model.T
        Λ_Θ_0 = inv(model.Σ_Θ_Q[q])
        Λ_Δ_0 = inv(model.Σ_Δ_Q[q])
        Λ_P_K_K = [Λ_Θ_0 , fill(Λ_Δ_0,model.E)...]


        for PA in parentItter
            counts_K_E = counts_PA_K_E_flat[flat_idx(PA,K_pa),:,:]
            N_E = sum(counts_K_E,dims = 1)[1,:]

            for k in 1:model.K_Q[q]
        
                ind = setdiff(1:model.K_Q[q],k)
                Θ_K = state.Θ_Q_PA_K[q][PA...,:]
                Δ_E_K = [state.Δ_Q_E_PA_K[q,e][PA...,:] for e in 1:model.E]
                Θ_P_K = hcat(Θ_K,Δ_E_K...)'

                (μ_post,Σ_post) = PG_post(X, Θ_P_K ,counts_K_E', N_E, k,Λ_P_K_K)

                θ₁ = rand(Normal(μ_post[1], sqrt(Σ_post[1, 1])))

                state.Θ_Q_PA_K[q][PA..., k] = θ₁

                μ_cond = μ_post[2:end] + Σ_post[2:end, 1] * ((θ₁ - μ_post[1]) / Σ_post[1, 1])
                Σ_cond = Σ_post[2:end, 2:end] - Σ_post[2:end, 1] * Σ_post[1, 2:end]' / Σ_post[1, 1]

                Θ = state.Θ_Q_PA_K[q][PA..., :]
                H = get_H(model, q)
                for e in 1:model.E
                    Δ = state.Δ_Q_E_PA_K[q, e][PA...,ind]

                    a = H[k,k]
                    b = 2* Δ' * H[ind,k]
                    c = Δ' * H[ind,ind] * Δ - 2*model.KL_Q[q]
                    l,u = quadratic_solve(a, b, c)
                    
                    if !isfinite(l) || !isfinite(u)
                        @show l u a b c H Θ q Δ model.T μ_post Σ_post state.Δ_Q_E_PA_K  state.Θ_Q_PA_K state.I_Q_E
                    end
                    
                    μ = μ_cond[e]
                    σ = sqrt(Σ_cond[e, e])
                    new_delta = rand(CenterCutout(Normal(μ, σ),l,u))
                    if abs(new_delta) > 1000
                        @show new_delta k Θ_P_K model.T μ_post Σ_post state.Δ_Q_E_PA_K[q, e][PA...,:] 
                        @show l u a b c H Θ q Δ 
                        return
                    end

                    state.Δ_Q_E_PA_K[q, e][PA...,k] = new_delta
                    if !isfinite(state.Δ_Q_E_PA_K[q, e][PA...,k]) 
                        @show state.Δ_Q_E_PA_K[q, e][PA...,k],state.Δ_Q_E_PA_K[q, e][PA...,:]
                    end
                end
            end
        end
    end
end



updateParam!(::Val{:Δ_Q_E_PA_K},model::CausalModel_Partial_Temper,state::CausalState) = begin  
    ### handled in the Θ_Q_PA_K update
    updateParam!(Val{:Θ_Q_PA_K}(),model,state)
end

function updateParam!(::Val{:Z_E_Nₑ_Q}, model::CausalModel_Partial_Temper, state::CausalState)
    dims = Tuple(model.K_Q)
    indices = CartesianIndices(dims)
    NZ  = prod(dims)
    logp0 = zeros(Float64, dims)
    logp = zeros(Float64, dims)

    for e in 1:model.E

        X = state.X_E_Nₑ_D[e]
        Z = state.Z_E_Nₑ_Q[e]
        Nₑ = model.N_E[e]

        
        fill!(logp0, 0.0)
           
        for q in 1:model.Q
            PA = model.G[q]

            Θ_PA_Z = state.Θ_Q_PA_K[q] 
            if state.I_Q_E[q, e]
                Θ_PA_Z = state.Θ_Q_PA_K[q].+ state.Δ_Q_E_PA_K[q, e]
            end
            for z_ind in indices
                z_tuple = z_ind.I 
                z_PA = ntuple(p -> z_tuple[p], length(PA))

                Θ = @view Θ_PA_Z[z_PA...,:]                
                logp0[z_ind] += Θ[z_tuple[q]] - logsumexp(Θ)

            end  
        end
       


        z = zeros(Int,model.Q)
        for i in 1:Nₑ

            xᵢ = @view X[i, :]
            logp.= logp0./model.T
            z = zeros(Int,model.Q)
            for lin_ind in 1:NZ
                un_flat_idx!(z,lin_ind, dims)
                a = logpdf_un_normalized(state.X_dist, z, xᵢ)

                logp[lin_ind] += a
            end

            flat_logp = reshape(logp, :)
            offset = logsumexp(flat_logp)
            prob = exp.(flat_logp .- offset)
            z_idx = rand(Categorical(prob))
            z_sample = indices[z_idx]
            Z[i, :] .= z_sample.I

        end        
    end

end


function calc_ols(model::CausalModel_Partial_Temper{LikertGen},state::CausalState)
    gen = model.gen
    dist = state.X_dist
    N = sum(model.N_E)
    dists = MvNormal[]
    Z_N_Q = hcat(vcat([ state.Z_E_Nₑ_Q[e].-dist.shift_Q'  for e in 1:model.E]...),ones(N) )
    X_N_D = vcat([ state.X_E_Nₑ_D[e]  for e in 1:model.E]...)
 
    Σ_0 = Diagonal(fill(gen.σ²_M,model.Q+1))
    Λ_0 = inv(Σ_0)
    ZᵀZ = Z_N_Q' * Z_N_Q

    for d in 1:model.D
        Λ_data = ZᵀZ./dist.σ²_D[d]
        Σ_post = Symmetric(inv(Λ_0 + Λ_data + LinearAlgebra.I*1e-6))
        μ_post = Σ_post * ( (Z_N_Q' * X_N_D[:,d])./dist.σ²_D[d] + Λ_0 * fill(gen.μ_M,model.Q + 1) )
        push!(dists,MvNormal(μ_post,Σ_post))
       
    end 
    return dists
end


updateParam!(::Val{:X_dist},model::CausalModel_Partial_Temper{LikertGen},state::CausalState) = begin
    dists = calc_ols(model,state)
    for (d,dist) in zip(1:model.D,dists)
        state.X_dist.M_Q_D[:,d] .= rand(dist)
    end 

    gen = model.gen
    dist = state.X_dist
    N = sum(model.N_E)

    Z_N_Q = hcat(vcat([ state.Z_E_Nₑ_Q[e].-dist.shift_Q'  for e in 1:model.E]...),ones(N) )
    X_N_D = vcat([ state.X_E_Nₑ_D[e]  for e in 1:model.E]...)

    α = gen.α + N/2
    β = gen.β .+ sum((X_N_D .- Z_N_Q * state.X_dist.M_Q_D).^2,dims =1)[1,:]./(2 *  model.T)
    for d in 1:model.D
        σ² = try
            rand(InverseGamma(α, β[d]))
        catch err
            @warn "σ² draw failed at d=$d; using fallback" β=β[d] exception=(err, catch_backtrace())
            gen.σ²_X         
        end
        state.X_dist.σ²_D[d] = model.T * clamp(σ²,0, gen.σ²_X)
    end
    #state.X_dist.σ²_D .= model.T .* clamp.(rand.(InverseGamma.(α, β)),0,gen.σ²_X)
    #state.X_dist.σ²_D .= fill(model.T,model.D)
 
end


updateParam!(::Val{:X_E_Nₑ_D},model::CausalModel_Partial_Temper,state::CausalState) = begin
    for e in 1:model.E
        for i in 1:model.N_E[e]
            z = state.Z_E_Nₑ_Q[e][i,:]
            state.X_E_Nₑ_D[e][i,:] .=  rand(state.X_dist,z)          
        end
    end
end


Distributions.logpdf(dist::LikertDist, model::CausalModel_Partial_Temper{LikertGen}) = begin
    gen = model.gen
    ans =  sum(logpdf.(Normal(gen.μ_M,sqrt( gen.σ²_M)), dist.M_Q_D)) 
    ans += sum(logpdf.(InverseGamma(gen.α,gen.β), dist.σ²_D./model.T)) 
    return ans
end

jointlogpdf(model::CausalModel_Partial_Temper,state::CausalState) = begin
    
    logp = 0.0
    
    # ==== Mixture means ====== #
    logp += logpdf(state.X_dist,model)
    # ==== X ====== #
    for e in 1:model.E
        for n in 1:model.N_E[e]
            x = @view state.X_E_Nₑ_D[e][n,:]
            z = @view state.Z_E_Nₑ_Q[e][n,:]
            logp+=logpdf(state.X_dist,z,x)
        end
    end
    # ==== Z ====== #
    for q in 1:model.Q
    
        PA = model.G[q]
        Kq = model.K_Q[q]
        K_pa = model.K_Q[PA]

        idx = zeros(Int,Kq)
        for e in 1:model.E
            Θ_PA_Z = state.I_Q_E[q,e] ? state.Θ_Q_PA_K[q] .+ state.Δ_Q_E_PA_K[q,e] :
                              state.Θ_Q_PA_K[q]
            for i in 1:model.N_E[e]
                z = state.Z_E_Nₑ_Q[e][i,q]

                pe = flat_idx(e, i, PA, K_pa, state.Z_E_Nₑ_Q)
                idx .= (0:Kq-1).*prod(K_pa) .+ pe
                @assert maximum(idx) <= length(Θ_PA_Z)
                @assert minimum(idx) >= 1
                Θ_K = Θ_PA_Z[idx]
                Θ_val = Θ_K[z] /model.T
                ls_Θ_K = logsumexp(Θ_K./model.T)
                logp += Θ_val - ls_Θ_K
            end
        end   
    end 

     # ==== Θ ====== #
     for q in 1:model.Q
        parentItter = Iterators.product((1:k for k in model.K_Q[model.G[q]])...)
        μ = zeros(model.K_Q[q])
        Σ = model.Σ_Θ_Q[q]
        for PA in parentItter
            logp += logpdf(MvNormal(μ,Σ),state.Θ_Q_PA_K[q][PA...,:])
        end
    end

    # ==== Δ ====== #
    for e in 1:model.E
        for q in 1:model.Q
            parentItter = Iterators.product((1:k for k in model.K_Q[model.G[q]])...)
            μ = zeros(model.K_Q[q])
            Σ = model.Σ_Δ_Q[q]
            for PA in parentItter
                Δ = state.Δ_Q_E_PA_K[q,e][PA...,:]
                logp += logpdf(MvNormal(μ,Σ),Δ)
                if KL_norm(get_H(model,q),Δ) <= model.KL_Q[q]
                    return -Inf
                end
            end
        end
    end

    # ==== c  ====== #
    logp += sum(logpdf.(Beta(model.α_p, model.β_p), state.c_Q))

     # ==== I  ====== #
     for q in 1:model.Q
        logp += sum( logpdf.(Bernoulli(state.c_Q[q]), state.I_Q_E[q,:]))
    end
    return logp
end


sampleData(model::CausalModel_Partial_Temper;kwargs...) = begin
    @assert !isnothing(model.state)
    return model.state.X_E_Nₑ_D
end

#helper to make graph 
connectedG(n) = begin
    G = Array{Array{Int,1}}(undef, n)
    for i in 1:n
        G[i] = collect(1:(i-1))
    end
    return G
end



lgprb(z_vals,model,state) = begin
    logp = 0
    logp_E = zeros(model.E)
    for q in 1:model.Q
        PA = model.G[q]
        Θ = state.Θ_Q_PA_K[q]
        z_PA = ntuple(i -> z_vals[PA[i]], length(PA))
        z = z_vals[q]
        Θ_K = Θ[z_PA...,:]./model.T
        logp += (Θ_K[z]- logsumexp(Θ_K))
        for e in 1:model.E
            Δ = state.Δ_Q_E_PA_K[q,e]
            Δ_K = Δ[z_PA...,:]./model.T
            logp_E[e] += (Θ_K[z]+Δ_K[z] - logsumexp(Θ_K.+Δ_K))
        end   
    end 
    return (logp,logp_E)
end

function apply_full_permutation!(model::CausalModel_Partial_Temper, σ)
    
    state = model.state
    Q = model.Q
    K_dims = Tuple(model.K_Q)
    D = model.D

    # === 4. permute I ===
    model.state.I_Q_E.= model.state.I_Q_E[σ,:]

    # === 1. Relabel latent assignments ===
    for e in 1:model.E
        Z = state.Z_E_Nₑ_Q[e]
        for i in 1:model.N_E[e]
            Z[i, 1:model.Q] .= Z[i, σ]
        end
    end
    # === 2. Permute mixture means ===
    state.X_dist.M_Q_D[1:model.Q,:] .= state.X_dist.M_Q_D[σ,:]
    # === 3. Permute Theta,Delta ===
    new_log_prob_map  =  Dict(newIdx.I => lgprb(newIdx.I[σ],model,state) for newIdx in CartesianIndices(Tuple(model.K_Q)))
    for q in 1:model.Q
        dims = tuple(model.K_Q[model.G[q]]...,model.K_Q[q] )
        dims_E = tuple(model.K_Q[model.G[q]]...,model.K_Q[q],model.E )
        log_probs = fill(-Inf, dims)       # log(0)
        log_probs_E = fill(-Inf, dims_E)
        PA = model.G[q]

        for (idx,log_prb) in new_log_prob_map
            z_PA = ntuple(i -> idx[PA[i]], length(PA))
            z = idx[q]
            log_probs[z_PA..., z] = logaddexp(log_probs[z_PA..., z], log_prb[1])
            for e in 1:model.E
                log_probs_E[z_PA..., z, e] = logaddexp(log_probs_E[z_PA..., z, e], log_prb[2][e])
            end
        end

        parentItter = Iterators.product((1:k for k in model.K_Q[model.G[q]])...)
        for PA in parentItter
            new = log_probs[PA...,:] 
            state.Θ_Q_PA_K[q][PA...,:] .= new .-mean(new) .+mean(state.Θ_Q_PA_K[q][PA...,:])
            for e in 1:model.E
                new = log_probs_E[PA...,:,e].-state.Θ_Q_PA_K[q][PA...,:] 
                state.Δ_Q_E_PA_K[q,e][PA...,:] .= new .-mean(new) .+mean(state.Δ_Q_E_PA_K[q,e][PA...,:])
            end
        end

    end

end

log_proposal_prob(model::CausalModel_Partial_Temper,state::CausalState,σ) = 0.0


function attempt_full_label_permutation!(model::CausalModel_Partial_Temper)


    σ = shuffle(1:model.Q)

    
    logp_before = jointlogpdf(model, model.state)
    logp_bwd = log_proposal_prob(model,model.state,σ )



    state_backup = deepcopy(model.state)
    apply_full_permutation!(model, σ)




    logp_after = jointlogpdf(model, model.state)
    logp_fwd = log_proposal_prob(model,model.state,σ )



    α = logp_after - logp_before + logp_bwd - logp_fwd

    #@show α
    if log(rand()) > α
        model.state = state_backup
        return false
    end
    return true
end







@kwdef mutable struct PermutingSampler <: AbstractSampler
    baseSampler::AbstractSampler
    n_permute::Int = 1
end
get_n_step(s::PermutingSampler) = get_n_step(s.baseSampler)
get_n_burn(s::PermutingSampler) = get_n_burn(s.baseSampler)
get_n_thin(s::PermutingSampler) = get_n_thin(s.baseSampler)
get_n_chain(s::PermutingSampler) = get_n_chain(s.baseSampler)



PermutingSampler(model::AbstractModel; n_step, n_permute = 1, kwargs...) = begin
    PermutingSampler(;baseSampler = GibbsSampler(model; n_step, kwargs...),n_permute= n_permute)
end

update_state!(model::CausalModel_Partial_Temper,sampler::PermutingSampler;n,debug_struct,kwargs...) = begin
    update_state!(model,sampler.baseSampler;n,debug_struct=debug_struct[2])
    if n%sampler.n_permute == 0
        debug_struct[1][:total] +=1
        sawped = attempt_full_label_permutation!(model::CausalModel_Partial_Temper)
        debug_struct[1][:swap] += sawped
    end
end

alloc_debug(m::AbstractModel,s::PermutingSampler) = (Dict(:total => 0, 
                                                       :swap => 0) ,alloc_debug(m,s.baseSampler))

function show_debug(m::AbstractModel,s::PermutingSampler,debug_struct)
    @show debug_struct[1]
    @show debug_struct[1][:swap]./debug_struct[1][:total]
    show_debug(m,s.baseSampler,debug_struct[2])
end


