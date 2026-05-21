using Random
using Plots
using HypothesisTests
using ProgressMeter 
using Base.Threads

abstract type AbstractModel end
abstract type AbstractState end
abstract type AbstractSampler end

generateParam!(param::Val,model::AbstractModel; kwargs...)  = generateParam!(param,model,model.state; kwargs...)
updateParam!(param::Val,model::AbstractModel; kwargs...)  = updateParam!(param,model,model.state; kwargs...)


@kwdef mutable struct GibbsSampler <: AbstractSampler
    rng::AbstractRNG = Random.default_rng()
    n_step::Int
    n_burn::Int = 0
    n_thin::Int = 1
    n_chain::Int = 1
    updates::Tuple{Vararg{Symbol}} 
    schedule::Dict{Symbol,Tuple{Int,Int}} = Dict{Symbol,Tuple{Int,Int}}()
end
get_n_step(s::AbstractSampler) = s.n_step
get_n_burn(s::AbstractSampler) = s.n_burn
get_n_thin(s::AbstractSampler) = s.n_thin
get_n_chain(s::AbstractSampler) = s.n_chain

@kwdef mutable struct TemperedSampler <: AbstractSampler
    baseSampler::AbstractSampler
    n_swap::Int = 1
    save_all::Bool = false
    temps::Vector{Float64}
end
get_n_step(s::TemperedSampler) = get_n_step(s.baseSampler)
get_n_burn(s::TemperedSampler) = get_n_burn(s.baseSampler)
get_n_thin(s::TemperedSampler) = get_n_thin(s.baseSampler)
get_n_chain(s::TemperedSampler) = get_n_chain(s.baseSampler)



GibbsSampler(model::AbstractModel; n_step, kwargs...) = begin
    GibbsSampler(;n_step = n_step,
                  updates = stateFields(model),
                  kwargs...)
end

TemperedSampler(model::AbstractModel; n_step,temps,n_swap = 1, kwargs...) = begin
    TemperedSampler(;baseSampler = GibbsSampler(model; n_step, kwargs...),
                     temps = temps, n_swap = n_swap)
end




sample_posterior(model::AbstractModel;kwargs...) = begin
    sampler = GibbsSampler(model;kwargs...)
    sample_posterior(model::AbstractModel,sampler)
end


function sample_posterior(model::AbstractModel, s::AbstractSampler; progress_label ="Sampling...", kwargs...)
    model = deepcopy(model)
    n_chain = get_n_chain(s)

    if n_chain == 1
        return run_chain(model, s;progress_label=progress_label, kwargs...)
    else
        chains = Vector{Any}(undef, n_chain)
        @showprogress 1 progress_label for i in 1:n_chain
            chains[i] = run_chain(deepcopy(model), s;progress_label=progress_label, kwargs...)
        end
        return chains
    end
end


@kwdef mutable struct SMCS <: AbstractSampler
    n_particles::Int
    n_steps::Int = 1
    baseSampler::AbstractSampler
    temps::Vector{Float64}
end


function stratified_resample_indices(w::AbstractVector{<:Real})
    N = length(w)
    u = ((0:N-1) .+ rand(N)) ./ N
    bins = cumsum(w)
    idx = Vector{Int}(undef, N)
    j = 1
    @inbounds for i in 1:N
        ui = u[i]
        while ui > bins[j]
            j += 1
        end
        idx[i] = j
    end
    return idx
end

function deep_copy(m::T) where T<:AbstractModel
    return deepcopy(m)
end 

function deep_copy(s::T) where T<:AbstractState
    return deepcopy(s)
end 



function sample_posterior(model::T, s::SMCS; progress_label ="Sampling...",  ess_threshold=0.5,kwargs...) where T<:AbstractModel
    particles = [begin m = deep_copy(model); m.T = s.temps[1]; m end for _ in 1:s.n_particles]
    weights = fill(1.0/s.n_particles,s.n_particles)
    log_weights = log.(weights)
    debug_structs = [alloc_debug(model,s.baseSampler) for _ in 1:s.n_particles]
    for (particle,debug_struct) in zip(particles,debug_structs)
        generate_state!(particle,s.baseSampler;debug_struct=debug_struct,kwargs...)
    end
    prev_t = s.temps[1]
    @showprogress 1 progress_label for t in s.temps[1:end]
        ESS = 1/sum(weights.^2)
        if ESS < s.n_particles * ess_threshold
            idx = stratified_resample_indices(weights)
            particles = deep_copy.(particles[idx])
            log_weights .= 0.0
        end

        @threads for n in 1:s.n_particles
            particle = particles[n]
            debug_struct = debug_structs[n]
            
            log_prev = jointlogpdf(particle, particle.state)
            particle.T = t
            log_new = jointlogpdf(particle, particle.state)
            log_weights[n] += log_new - log_prev

            for step in 1:s.n_steps
                update_state!(particle,s.baseSampler;n=step,debug_struct=debug_struct)
            end
        end
        prev_t = t
        weights = exp.(log_weights.-logsumexp(log_weights))
    end
    return ([model_to_sample(p,s) for p in particles ],weights)
end


sample_type(model::AbstractModel,::AbstractSampler) = sampleType(model)

sample_type(model::AbstractModel,s::TemperedSampler) = begin
    s.save_all ? NTuple{length(s.temps),sampleType(model)} : sampleType(model)
end

alloc_model(model::AbstractModel,::AbstractSampler) = deepcopy(model)

alloc_model(model::AbstractModel,s::TemperedSampler) = begin
    models = [begin m = deepcopy(model); m.T = T; m end  for T in s.temps]
    return models
end

function generate_state!(model::AbstractModel,::AbstractSampler;kwargs...) 
    if isnothing(model.state)
        generateState!(model;kwargs...);
    else
        @warn "model has an initial state. Using that as the starting point of chain"
    end
end

function generate_state!(models::Vector{T},::TemperedSampler;kwargs...) where T <: AbstractModel
    for model in models
        if isnothing(model.state)
            generateState!(model;kwargs...);
        else
            @warn "model has an initial state. Using that as the starting point of chain"
        end
    end
end

update_state!(model::AbstractModel,sampler::GibbsSampler;n,debug_struct,kwargs...) = begin
    for param in sampler.updates
        if !haskey(sampler.schedule,param) || (n>=first(sampler.schedule[param]) && n%last(sampler.schedule[param])==0 )
            t0 = time()
            updateParam!(Val{param}(), model)
            debug_struct[param]+= time()-t0
        end
    end
    #@show n model.T maximum.(model.state.Θ_Q_PA_K) maximum(maximum.(model.state.Δ_Q_E_PA_K ))
end

update_state!(models::Vector{T},sampler::TemperedSampler;n,debug_struct,kwargs...) where T <: AbstractModel = begin
    @threads for i in 1:length(models) 
        model = models[i]
        update_state!(model,sampler.baseSampler;n,debug_struct=debug_struct[2][i])
    end

    if n%sampler.n_swap == 0
        for j in reverse(1:(length(models)-1))
            m1, m2 = models[j], models[j+1]
            s1, s2 = m1.state, m2.state
            log_accept = (jointlogpdf(m2, s1) + jointlogpdf(m1, s2)-
                        jointlogpdf(m2, s2) - jointlogpdf(m1, s1))
            debug_struct[1][:total][j]+=1
            if log(rand()) < log_accept
                debug_struct[1][:swap][j]+=1
                models[j].state, models[j+1].state = s2, s1
            end
        end
    end

end

model_to_sample(model::AbstractModel,::AbstractSampler) = modelToSample(model)
model_to_sample(models::Vector{T},s::TemperedSampler) where T <: AbstractModel  = begin
    s.save_all ?  Tuple(modelToSample(m) for m in models) : modelToSample(first(models)) 
end


alloc_debug(::AbstractModel,::AbstractSampler) = nothing
alloc_debug(::AbstractModel,s::GibbsSampler) = Dict(p => 0.0 for p in s.updates)
alloc_debug(m::AbstractModel,s::TemperedSampler) = (Dict(:total => zeros(Int,length(s.temps)-1), 
                                                       :swap => zeros(Int,length(s.temps)-1)) ,[alloc_debug(m,s.baseSampler) for _ in 1:length(s.temps) ])

show_debug(::AbstractModel,::AbstractSampler,debug_struct) = @show debug_struct
function show_debug(m::AbstractModel,s::GibbsSampler,debug_struct)
    for p in s.updates
        println(rpad("$p:", 20), debug_struct[p])
    end
end
function show_debug(m::AbstractModel,s::TemperedSampler,debug_struct)
    @show debug_struct[1]
    @show debug_struct[1][:swap]./debug_struct[1][:total]
    for (i, t) in enumerate(s.temps)
        println("-"^25 * "\nTemp $t debug:")
        show_debug(m,s.baseSampler,debug_struct[2][i])
    end
end
                    

                          

function run_chain(model::AbstractModel, s::AbstractSampler; debug = false, progress_label ="Sampling...",kwargs...)
    model_cpy = alloc_model(model,s);

    n_step = get_n_step(s);
    n_burn = get_n_burn(s);
    n_thin = get_n_thin(s);

    chain = Vector{sample_type(model,s)}(undef, n_step)

    debug_struct = alloc_debug(model,s)
    i = 1;
    @showprogress 3 progress_label for step in 0:(n_burn + (n_step-1) * n_thin)
        if step ==0 
            generate_state!(model_cpy,s;debug_struct=debug_struct,kwargs...)
        else 
            update_state!(model_cpy,s;n=step,debug_struct=debug_struct);
        end

        if step >= n_burn && (step - n_burn) % n_thin == 0
            chain[i] = model_to_sample(model_cpy,s);
            i += 1;
        end
    end
    if debug
        show_debug(model,s,debug_struct)
    end
    return chain
end



stateType(model::AbstractModel) =  typeof(model.state)


stateFields(model::AbstractModel) = fieldnames(stateType(model))

allocState!(model::AbstractModel) = begin 
    model.state = stateType(model)(model)  
end
allocState(model::AbstractModel) =  stateType(model)(model)  

generateState!(model::AbstractModel; kwargs...) = begin 
    if !isnothing(model.state)
        @warn "overwriting non null state with a new init"
    end
    allocState!(model)
    for param in stateFields(model)
        if param in keys(kwargs)
            #@warn "using passed in value for"  param
            setfield!(model.state, param, get(kwargs,param,nothing))
        else
            generateParam!(Val{param}(),model)
        end
    end
    model.state=deep_copy(model.state)
end

#sample data from the model

sample_model!(model::AbstractModel;kwargs...) = begin
    @warn "will modify model state, when sampling"
    if isnothing(model.state)
        @warn "state not initialized, sampling state from prior"
        generateState!(model;kwargs...)
    end
    sampleData(model;kwargs...)
end



#basic sample def. overload to add elements from model/prune elements from state
modelToSample(model::AbstractModel) = deepcopy(model.state)
sampleType(model::AbstractModel) = stateType(model)

