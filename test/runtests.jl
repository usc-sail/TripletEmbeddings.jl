using Printf
using Random
using Distances
using Distributions
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


function test_procrustes(d::Int)
    @assert d == 1 || d == 2

    n = 100
    X = rand(d, n)
    if d == 1
        Y = randn() * X .+ 5 * rand()
    else
        θ = rand(Uniform(0, 2π))
        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        Y = broadcast(+, randn() * R * X , 5 * rand(d))
    end
    X = Embedding(X)
    Z = procrustes(X, Y)

    return mse(Z,Y)
end

@testset "procrustes.jl" begin   
    @test isapprox(test_procrustes(1), 0.0; atol=eps())
    @test isapprox(test_procrustes(2), 0.0; atol=eps())
end