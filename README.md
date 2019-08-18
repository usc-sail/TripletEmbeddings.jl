# TripletEmbeddings.jl
This package implements various triplet embedding algorithms.

## Usage
### Multithreading
This package uses threads to compute the gradients. To set the number of threads in Julia, open a terminal and run:

```bash
$ export JULIA_NUM_THREADS=n
```
where `n` is the number of threads you want to use.

### Installation
This package is currently not registered. To install, open a REPL and install directly from this repo:

```julia
julia> ]
(v1.0) pkg> add https://github.com/usc-sail/TripletEmbeddings.jl
```
To use this package, you can now use:

```julia
julia> using TripletEmbeddings
```



### Examples
#### 1D embeddings
We generate a random signal, compute its triplets, and then fit an embedding to those triplets:

```julia
using Plots
using Random
using TripletEmbeddings

Random.seed!(4)

n = 100
data = rand(100)

# Compute the triplets
triplets = Triplets(data)

# Initialize a loss function
loss = STE(σ = 1/sqrt(2))

# Initialize a random embedding
X = Embedding(length(data), 1)

# Fit the embedding
@time violations = fit!(loss, triplets, X; verbose=true, max_iterations=50)

plot(data)
plot!(scale(data, X))
```
This code generates the following embedding:

![1D example](figures/1D.svg)

#### Embeddings of dimensions 2 or greater

```julia
Random.seed!(4)

n = 100; d = 2;
data = rand(n, d)

triplets = Triplets(data)
loss = STE(σ = 1/sqrt(2))
X = Embedding(size(data))

@time violations = fit!(loss, triplets, X; verbose=true, max_iterations=200)
X, _ = procrustes(data, X)

scatter(data[:,1], data[:,2], label="Data")
scatter!(X[:,1], X[:,2], label="Embedding")
```
This code generates the following embedding:

![1D example](figures/2D.svg)