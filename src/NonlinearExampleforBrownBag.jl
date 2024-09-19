module NonlinearExampleforBrownBag

using Polynomials
using LinearAlgebra

include("Chebyshev.jl")

export ChebyshevTPoints
export VandermondeMatrix, InverseVandermondeMatrix
export VandermondeMatrixAndInverse
export LVM, RVM, CollocationLeftVandermondeMatrix, CollocationRightVandermondeMatrix
export CoefficientDerivativeMatrix, CollocationDerivativeMatrix
export CollocationProjectionMatrix
export ChebyshevHelper

end
