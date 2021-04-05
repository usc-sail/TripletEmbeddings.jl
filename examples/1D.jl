using Random
using VegaLite
using StatsFuns
using DataFrames
using Distributions
using TripletEmbeddings
using GreenVideoExperiments

Random.seed!(4)

# n = 100
# Y = 2*rand(1,n) .- 1
Y = GreenVideoExperiments.load("TaskA", "ground_truth", "30Hz").data[begin:30:end]
Y = Matrix(transpose(filter(x -> !isnan(x), Y)))

triplets = Triplets(Y; f = x -> logistic(2x))
loss = STE(σ = 1/sqrt(2))
X = Embedding(size(Y))

@time violations = fit!(loss, triplets, X; verbose=true, max_iterations=1000)
procrustes!(X, Y)

dfX = DataFrame(embedding = "X", time = 1:length(X), value = X[1,:])
dfY = DataFrame(embedding = "Y", time = 1:length(Y), value = Y[1,:])

vcat(dfX, dfY) |> @vlplot(:line, x = :time, y = :value, color=:embedding, width=600, height=400)

# savefig("figures/1D.svg")