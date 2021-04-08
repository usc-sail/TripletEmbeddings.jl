module TripletEmbeddings

using Printf
using Random
using Logging
using Distances
using StatsFuns
using Statistics: mean
using SparseArrays
using LinearAlgebra
using InvertedIndices


export Embedding, ndims, nitems, V, Gram, L,
       Triplet, Triplets, checktriplets, sampletriplet, sampletriplets,
       LabeledTriplet, LabeledTriplets,
       STE, ρSTE, tSTE, kernel,
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
include("rhoSTE.jl")
include("tSTE.jl")
include("fit.jl")
include("procrustes.jl")

end # module