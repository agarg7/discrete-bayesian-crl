# Bayesian Multi-Domain Causal Representation Learning

Source code for
**“Discrete Causal Representations from Heterogeneous Domains: A Bayesian Approach with Social Survey Applications”**

[paper link]


## Repository Structure

* `data/` contains all datasets used
* `src/` contains inference and model code


## Running the Model

### Setup

Clone the repository and instantiate the Julia environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

All required dependencies are listed in `Project.toml`.


### Example usage:

```bash
julia --project=. src/model-run.jl \
      --inFile pollfish_data/data_ER.json \
      --Q 3 \
      --K 2
```

### File Format:

The script expects a JSON file with the following structure:

```json
{
  "X_list": [...],
  "Environment_list": [...],
  "Question_list": [...]
}
```

* Each `X_list[e]` is a flattened matrix of size `(N_e × D)`, corresponding to environment `e`

It will save the posterior samples and weights  as both `.jld2` and `.json` files

---

### Key Arguments


```bash
  --inFile INFILE       Input file
  --KL KL               KL tuncation threshold (default: 0.1)
  --Q Q                 Number of latents 
  --K K                 Cardinality of latents 
  --n_steps N_STEPS     MCMC steps per temperature (default: 1)
  --n_samples N_SAMPLES
                        Number of posterior samples (default: 1000)
  --n_temps N_TEMPS     Number of Temperatures (default:
                        100)
  --T_max T_MAX         Maximum Temperature (1000.0)
```

For all available options:

```bash
julia --project=. src/model-run.jl --help
```
---

## Notes

* The code uses `Q` to denote the number of latent variables, whereas the paper uses `L`.

