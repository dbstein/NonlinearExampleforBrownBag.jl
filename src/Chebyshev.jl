################################################################################
# Construction of Chebyshev Nodes

"""
Provides chebyshev quadratures nodes on [-1, 1]
The nodes are reversed from traditional chebyshev nodes
    (so that the lowest valued node comes first)
"""
function ChebyshevTPoints(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat
    return cos.( (T(π)/2) .* (2 .* Vector{T}(order:-1:1) .- 1) ./ order )
end

################################################################################
# Construction of Vandermonde Matrices

function VandermondeMatrix(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat
    return vander(ChebyshevT, ChebyshevTPoints(order, T), order-1)
end
function VandermondeMatrix(order::Integer, x::Vector{T}) where T <: AbstractFloat
    return vander(ChebyshevT, x, order-1)
end
function VandermondeMatrix(order::Integer, x::T) where T <: AbstractFloat
    return vander(ChebyshevT, T[x], order-1)
end
function InverseVandermondeMatrix(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat
    return inv(VandermondeMatrix(order, T))
end
function VandermondeMatrixAndInverse(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat
    V = VandermondeMatrix(order, T)
    return V, inv(V)
end
# Left and Right Vandermonde Matrices
LVM(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat = VandermondeMatrix(order, -one(T))
RVM(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat = VandermondeMatrix(order,  one(T))
# Left and Right Vandermonde Matrices (Collocation)
CollocationLeftVandermondeMatrix(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat = VandermondeMatrix(order, -one(T))*InverseVandermondeMatrix(order, T)
CollocationRightVandermondeMatrix(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat = VandermondeMatrix(order,  one(T))*InverseVandermondeMatrix(order, T)

################################################################################
# Construction of derivative matrices

function CoefficientDerivativeMatrix(
    order::Integer,
    D::Integer,
    ::Type{T}=Float64,
) where T <: AbstractFloat
    DM = zeros(T, order-D, order)
    b = zeros(T, order)
    for i in eachindex(b)
        @. b *= zero(T)
        b[i] = one(T)
        w = derivative(ChebyshevT(b), D).coeffs
        DM[1:length(w), i] = w
    end
    return DM
end
function CollocationDerivativeMatrix(
    order::Integer,
    D::Integer,
    ::Type{T}=Float64,
) where T <: AbstractFloat
    CoefficientDM = CoefficientDerivativeMatrix(order, D, T)
    VMI = InverseVandermondeMatrix(order, T)
    VM = VandermondeMatrix(order-D, T)
    return VM*CoefficientDM*VMI
end
function CollocationProjectionMatrix(
    order_in::Integer,
    order_out::Integer,
    ::Type{T}=Float64,
) where T <: AbstractFloat
    VMI = InverseVandermondeMatrix(order_in, T)[1:order_out, :]
    VM = VandermondeMatrix(order_out, T)
    return VM*VMI
end

################################################################################
# Helper struct for these demonstrations

struct ChebyshevHelper
    P2::Matrix{Float64}
    D1::Matrix{Float64}
    D2::Matrix{Float64}
    CLVM::Vector{Float64}
    CRVM::Vector{Float64}
end
function ChebyshevHelper(N)
    D2 = CollocationDerivativeMatrix(N, 2);
    P2 = CollocationProjectionMatrix(N, N-2);
    _D1 = CollocationDerivativeMatrix(N, 1);
    _P1 = CollocationProjectionMatrix(N-1, N-2);
    D1 = _P1 * _D1;
    CLVM = CollocationLeftVandermondeMatrix(N);
    CRVM = CollocationRightVandermondeMatrix(N);
    return ChebyshevHelper(P2, D1, D2, CLVM[1,:], CRVM[1,:])
end
function (H::ChebyshevHelper)(u)
    Pu  = H.P2 * u;
    u′ = H.D1 * u;
    u′′ = H.D2 * u;
    uₗ = dot(H.CLVM, u)
    uᵣ = dot(H.CRVM, u)
    return Pu, u′, u′′, uₗ, uᵣ
end
function (H::ChebyshevHelper)(u, Pu, u′, u′′)
    mul!(Pu, H.P2, u)
    mul!(u′, H.D1, u)
    mul!(u′′, H.D2, u)
    uₗ = dot(H.CLVM, u)
    uᵣ = dot(H.CRVM, u)
    return uₗ, uᵣ
end
