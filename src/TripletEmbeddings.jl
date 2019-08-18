module TripletEmbeddings

using Printf
using Random
using Distances
using Statistics: mean
using LinearAlgebra

export Embedding, ndims, nitems, J,
       Triplets, ntriplets,
       STE, tSTE, kernel,
       fit!, gradient, tgradient, tgradient!, cost, tcost,
       scale, scale!, procrustes, procrustes!

import Base: size, getindex, length, iterate, ndims, broadcast, -, *
import Distances: pairwise

include("embedding.jl")
include("triplets.jl")
include("utilities.jl")
include("losses.jl")
include("STE.jl")
include("tSTE.jl")
include("fit.jl")

end # module