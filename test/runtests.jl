using Printf
using Random
using Distances
using LinearAlgebra

using TripletEmbeddings
using Test

@testset "embeddings.jl" begin
    d = 2
    n = 10
    X = Embedding(d,n)

    @test size(X) == (d,n)
    @test size(Embedding([1 0; 0 1])) == (2,2)
    @test size(Embedding((1,10))) == (1,10)
    @test eltype(Embedding([1 1; 1 1])) <: AbstractFloat # This is needed for kernel(loss, X)

    @test issymmetric(X.G)
    
    @test nitems(X) == n
    @test ndims(X) == d
    
    X₀ = rand(n)
    X = Embedding(X₀)
    @test nitems(X) == n
    @test ndims(X) == 1

    @test J(d) == [0.5 -0.5; -0.5 0.5]
end

@testset "triplets.jl" begin
    data = [1 2 3]
    triplets = Triplets(data)
    @test triplets.triplets == [(3, 2, 1); (1, 2, 3)]

    data = [1 2 3; 1 2 3]
    triplets = Triplets(data)
    @test triplets.triplets == [(3, 2, 1); (1, 2, 3)]

    @test size(triplets) == (2,)
    @test length(triplets) == 2
end

@testset "STE.jl" begin
    σ = 1/sqrt(2)
    loss = STE(σ = σ)

    @test loss.σ == σ
    @test loss.constant == 1/σ^2

    data = [1 2 3; 1 2 3]
    triplets = Triplets(data)
end
