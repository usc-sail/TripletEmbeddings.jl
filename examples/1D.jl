using Random
using VegaLite
using StatsFuns
using DataFrames
using TableReader
using Distributions
using TripletEmbeddings

Random.seed!(4)

n = 100
Y = 2*rand(1,n) .- 1

triplets = Triplets(Y; f = x -> logistic(20x))
loss = STE(σ = 1/sqrt(2))
X = Embedding(size(Y))

@time violations = fit!(loss, triplets, X; verbose=true, max_iterations=1000)
procrustes!(X, Y)

dfX = DataFrame(embedding = "X", time = 1:n, value = X[1,:])
dfY = DataFrame(embedding = "Y", time = 1:n, value = Y[1,:])

vcat(dfX, dfY) |> @vlplot(:line, x = :time, y = :value, color=:embedding, width=600, height=400)

# savefig("figures/1D.svg")