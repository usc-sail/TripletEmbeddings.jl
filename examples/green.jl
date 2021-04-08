
using CSV
using Plots
using Random
using StatsFuns
using DataFrames
using Distributions
using LinearAlgebra
using TripletEmbeddings
using GreenVideoExperiments

# Y = Matrix(transpose(load("TaskA", "ground_truth", "30Hz")[begin:30:end,:data]))
Y = CSV.read(expanduser("~/Documents/Research/PRL2019/data/TaskA.csv"), DataFrame, header=false)
Y = Matrix(transpose(Y.Column1))[:,begin:30:end]

plt = plot(legend=:bottomleft)
plot!(plt, transpose(Y), label="Embedding", color=:black)

# for μ in [0.7, 0.8, 0.9]
for μ in [0.7]
    triplets = Triplets(Y; f = x -> logistic(10x))
    Ŷ = Embedding(size(Y))

    misclassifications = fit!(tSTE(α=30), triplets, Ŷ)
    Ŷ, tr = procrustes(Ŷ, Y)

    @show norm(Y - Ŷ).^2

    plot!(plt, transpose(Ŷ), label="Embedding μ = $(μ)")
end

display(plt)

################
### Debiased ###
################
let
    triplets = LabeledTriplets(Y; f = x -> logistic(10x))

    Ŷ = Embedding(size(Y))
    loss = ρSTE(ρ = Dict([-1, +1] .=> [0.3, 0.3]))
    misclassifications = fit!(loss, triplets, Ŷ; max_iterations=1000)

    Ŷ, tr = procrustes(Ŷ, Y)

    @show norm(Y - Ŷ).^2

    plot!(plt, transpose(Ŷ), label="Debiased")
end