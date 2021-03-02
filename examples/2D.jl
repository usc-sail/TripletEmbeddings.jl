using Plots
using Random
using VegaLite
using DataFrames
using LinearAlgebra
using TripletEmbeddings

Random.seed!(4)

# We define a random Embedding in ℜ²
d = 2
n = 100
Y = rand(d, n)

triplets = Triplets(Y)
loss = STE(σ = 1/sqrt(2))
X = Embedding(size(Y))

@time violations = fit!(loss, triplets, X; verbose=true, max_iterations=200)
procrustes!(X, Y)

scatter(Y[1,:], Y[2,:]); scatter!(X[1,:], X[2,:])

dfX = DataFrame(embedding = "X", time = 1:n, x = X[1,:], y = X[2,:])
dfY = DataFrame(embedding = "Y", time = 1:n, x = Y[1,:], y = Y[2,:])

vcat(dfX, dfY) |> @vlplot(:point, x = :x, y = :y, color=:embedding, width=600, height=400)