module TripletEmbeddings

using Printf
using Random
using Zygote
using Logging
using ThreadsX
using Distances
using StatsFuns
using Statistics: mean
using SparseArrays
using LossFunctions
using LinearAlgebra

export Embedding, ndims, nitems, V, Gram, L,
       Triplet, Triplets, ntriplets, checktriplets,
       # STE, tSTE, kernel,
       STE, XSTE, tSTE, kernel,
       fit!, gradient, tgradient, tgradient!, cost, tcost,
       procrustes, procrustes!, mse, apply,
       show

import Base: size, getindex, length, iterate, ndims, broadcast, -, *, show
import Distances: pairwise

include("embedding.jl")
include("triplets.jl")
include("utilities.jl")
include("losses.jl")
include("STE.jl")
include("XSTE.jl")
include("tSTE.jl")
include("fit.jl")
include("procrustes.jl")

end # module