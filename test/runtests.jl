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

    @test nitems(X) == n
    @test ndims(X) == d

    X₀ = rand(n)
    X = Embedding(X₀)
    @test nitems(X) == n
    @test ndims(X) == 1

    @test V(d) == [0.5 -0.5; -0.5 0.5]

    @test  L(3, (1,2,3)) == [0.0 -1.0 1.0; -1.0 1.0 0.0; 1.0 0.0 -1.0]
end

@testset "triplets.jl" begin
    data = [1 2 3]
    vec_triplets = [Triplet(Int16, (3, 2, 1)); Triplet(Int16, (1, 2, 3))]
    test_triplets = Triplets(vec_triplets)
    triplets = Triplets(data)
    @test triplets.triplets == vec_triplets
    @test triplets == test_triplets

    data = [1 2 3; 1 2 3]
    triplets = Triplets(data)
    @test triplets == test_triplets

    @test size(triplets) == (2,)
    @test length(triplets) == 2

    triplets = Triplets([Triplet(Int32,(1,2,3)), Triplet(Int32, (1,2,5))])
    @test checktriplets(triplets) == false

    @test_throws ArgumentError Triplets(ones(1,10))
end

@testset "STE.jl" begin
    σ = 1/sqrt(2)
    loss = STE(σ = σ)

    @test loss.σ == σ
    @test loss.constant == 1/σ^2

    data = [1 2 3; 1 2 3]
    triplets = Triplets(data)
end

function test_insample(d::Int)
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
    Z, transform = procrustes(X, Y)

    return mse(Z, Y)
end

function test_outofsample()
    Y = broadcast(+, [1 1 -1 -1; 1 -1 1 -1], [2, 2])
    s = rand()
    θ = rand()
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    b = rand(2)
    X = [Y mean(Y, dims=2)]
    X = broadcast(+, s * R * X, b)

    _, transform = procrustes(X[:,1:end-1], Y)
    return mean(Y, dims=2) ≈ apply(transform, X[:,end])
end

@testset "procrustes.jl" begin
    @test isapprox(test_insample(1), 0.0; atol=eps())
    @test isapprox(test_insample(2), 0.0; atol=eps())
    @test test_outofsample()
end