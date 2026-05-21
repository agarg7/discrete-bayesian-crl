include("model-main.jl")
using Plots
using JLD2
using Serialization
using Dates
using ProgressMeter 
using Base.Threads
using ArgParse
using CSV
using DataFrames
using JSON3, StructTypes
using Random

s = ArgParseSettings()

@add_arg_table s begin
    "--KL"
        help = "KL tuncation threshold"
        arg_type = Float64
        default = 0.1
    "--Q"
        help = "Number of latents"
        arg_type = Int
        required = true
    "--K"
        help = "Cardinality of latents"
        arg_type = Int
        required = true
    "--a"
        help = "Prior on sparsity; Beta(a,b)"
        arg_type = Float64
        default = 1.0
    "--b"
        help = "Prior on sparsity; Beta(a,b)"
        arg_type = Float64
        default = 9.0
    "--sd_theta"
        help = "Prior on shared logits, theta"
        arg_type = Float64
        default = 1.0
    "--sd_delta"
        help = "Prior on environment-specific deviations, delta"
        arg_type = Float64
        default = 1.0
    "--var_M"
        help = "Prior cariance of measurement model params, M"
        arg_type = Float64
        default = 1.0
    "--seed"
        help = "Random seed"
        arg_type = Int
        default = nothing
    "--n_steps"
        help = "MCMC steps per temperature"
        arg_type = Int
        default = 1
    "--n_samples"
        help = "Number of posterior samples"
        arg_type = Int
        default = 1000
    "--n_temps"
        help = "Number of Temperatures"
        arg_type = Int
        default = 100
    "--T_max"
        help = "Maximum Temperature"
        arg_type = Float64
        default = 1000.0
    "--inFile"
        help = "Input file"
        arg_type = String
        required = true
    "--outPath"
        help = "Output folder"
        arg_type = String
        default = "saved/"
    "--method"
        help = "Inference method: `SMCS`, `PT`, or `Gibbs`"
        arg_type = String
        default = "SMCS"
    "--debug"
        help = "Show debugging info"
        arg_type = Bool
        default = true
end


args = parse_args(s)
if !isnothing(args["seed"])
    Random.seed!(args["seed"])   
end


data = JSON3.read(args["inFile"])
X_list = [ reshape(Float64.(X),:,length(data.Question_list)) for X in data.X_list]


σ =  args["sd_theta"]
Σ_Θ_Q =  [1.1 * diagm(fill(σ,K).^2) - fill(σ,K)*fill(σ,K)'./K  for K in fill(args["K"],args["Q"])]
σ =  args["sd_delta"]
Σ_Δ_Q  =  [1.1 *diagm(fill(σ,K).^2) - fill(σ,K)*fill(σ,K)'./K  for K in fill(args["K"],args["Q"])]


model_init = CausalModel_Partial_Temper{LikertGen}(;Q=args["Q"], 
                                                    K_Q = fill(args["K"],args["Q"]), 
                                                    G = connectedG(args["Q"]), 
                                                    E = length(data.Environment_list), 
                                                    N_E = length.(data.X_list) .÷ length(data.Question_list),
                                                    Σ_Θ_Q=Σ_Θ_Q,
                                                    Σ_Δ_Q=Σ_Δ_Q,
                                                    KL_Q=fill(args["KL"],args["Q"]),
                                                    D= length(data.Question_list),
                                                    gen=LikertGen(;σ²_M = args["var_M"]),
                                                    T=1.0, 
                                                    α_p = args["a"], 
                                                    β_p = args["b"])






n_temps = args["n_temps"]
T_max = args["T_max"]  # proportional to number of samples
shared_temps = T_max .^ (range(0,1,length=n_temps))


SMCS_base = GibbsSampler(;n_step = args["n_steps"],
updates = (:c_Q, :I_Q_E, :Θ_Q_PA_K,:Z_E_Nₑ_Q,:X_dist)) 
SMCS_temps =  reverse(shared_temps)
SMCS_totalMCMC =  length(SMCS_temps)*args["n_samples"]*args["n_steps"]
SMCS_sampler = SMCS(;baseSampler = SMCS_base,
        n_particles = args["n_samples"],
        n_steps = args["n_steps"],
        temps = SMCS_temps)



PT_base = GibbsSampler(;n_step = args["n_samples"],
n_thin = args["n_steps"],
n_burn = args["n_samples"]*args["n_steps"],
updates = (:c_Q, :I_Q_E, :Θ_Q_PA_K,:Z_E_Nₑ_Q,:X_dist)) 
PT_permute_sampler = PermutingSampler(;baseSampler = PT_base)
PT_temps =  shared_temps[1:2:end]
PT_totalMCMC=  length(PT_temps)*(args["n_samples"]*args["n_steps"] + args["n_steps"]*args["n_samples"])
PT_sampler = TemperedSampler(;baseSampler = PT_permute_sampler,
n_swap = 5,
temps = PT_temps,
save_all = false)




Gibbs_mult = length(PT_temps)
Gibbs_sampler = GibbsSampler(;n_step = args["n_samples"],
n_thin = Gibbs_mult*args["n_steps"],
n_burn = Gibbs_mult*args["n_samples"]*args["n_steps"],
updates = (:c_Q, :I_Q_E, :Θ_Q_PA_K,:Z_E_Nₑ_Q,:X_dist)) 
Gibbs_totalMCMC =  2* Gibbs_mult*args["n_steps"]*args["n_samples"]


@info "Method: $(args["method"])"



mkpath(args["outPath"])
timestamp = Dates.format(Dates.now(), "mm-dd_HHMM");
function chain_filename(method;ext = "jld2")
    return joinpath(args["outPath"], "$(method)_Q$(args["Q"])_K$(args["K"])_KL$(args["KL"])_$(timestamp).$ext")
end

fname = chain_filename(args["method"])
fname_json = chain_filename(args["method"];ext="json")
if isfile(fname)
    @warn "File exisits already. Will overwrite "
end
if args["method"] == "SMCS"
    @info "Total MCMC Steps: $SMCS_totalMCMC"
    chain = sample_posterior(model_init,SMCS_sampler;debug = args["debug"],X_E_Nₑ_D = X_list);
    @save fname chain
    open(fname_json, "w") do io
        JSON3.write(io, chain)
    end
elseif args["method"] == "PT"
    @info "Total MCMC Steps: $PT_totalMCMC"
    chain = sample_posterior(model_init,PT_sampler;debug = args["debug"],X_E_Nₑ_D = X_list);
    @save fname chain
    open(fname_json, "w") do io
        JSON3.write(io, chain)
    end
elseif args["method"] == "Gibbs"
    @info "Total MCMC Steps: $Gibbs_totalMCMC"
    chain = sample_posterior(model_init,Gibbs_sampler;debug = args["debug"],X_E_Nₑ_D = X_list);
    @save fname chain 
    open(fname_json, "w") do io
        JSON3.write(io, chain)
    end
else
    @error "Invalid Method"
end







