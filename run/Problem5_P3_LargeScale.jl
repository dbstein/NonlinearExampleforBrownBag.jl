using NonlinearExampleforBrownBag
using LinearAlgebra
using Printf
using ForwardDiff
using Chairmarks
using Krylov
using SparseDiffTools

"""
Problem 3: Solve nonlinear problem:
u + uu′ + u′′ + uv = f
v + vv′ + v′′ + uv = g
with Dirichlet boundary conditions
where u(x) = exp(sin(x)) in [-1, 1]
and v(x) = cos(x²) in [-1, 1]
"""
u_function(s) = exp(sin(s));
u′_function(s) = exp(sin(s)) * cos(s);
u′′_function(s) = exp(sin(s)) * (cos(s)^2 - sin(s));
v_function(s) = cos(s^2);
v′_function(s) = -2s * sin(s^2);
v′′_function(s) = -2sin(s^2) - 4s^2*cos(s^2);

# order of discretization
N = 24;

# get Chebyhsev points and matrices
s = ChebyshevTPoints(N);

# construct analytical solution
uₐ = u_function.(s);
vₐ = v_function.(s);
# get boundary conditions
u₁ = u_function(1.0);
u₋₁ = u_function(-1.0);
v₁ = v_function(1.0);
v₋₁ = v_function(-1.0);
# compute forcing function
f = @. u_function(s) + u_function(s)*u′_function(s) + u′′_function(s) + u_function(s)*v_function(s);
g = @. v_function(s) + v_function(s)*v′_function(s) + v′′_function(s) + u_function(s)*v_function(s);

################################################################################
# Objective Functions

# struct to hold memory for non-allocating version
struct HelperMemory{T}
    Pu::Vector{T}
    u′::Vector{T}
    u′′::Vector{T}
end
function HelperMemory(N, T::Type=Float64)
    return HelperMemory{T}(zeros(T, N), zeros(T, N), zeros(T, N))
end
function (H::HelperMemory)(T)
    return H.Pu, H.u′, H.u′′
end
struct DualStruct{T, DT}
    StandardStruct::T
    DualStruct::DT
    DualStruct(StandardStruct, DualStruct) = new{typeof(StandardStruct), typeof(DualStruct)}(StandardStruct, DualStruct)
end
@inline (DS::DualStruct)(::Type{TT}) where TT = DS.StandardStruct(TT)
@inline (DS::DualStruct)(::Type{TT}) where TT <: ForwardDiff.Dual = DS.DualStruct(TT)
@inline ISDUAL(::Type) = false
@inline ISDUAL(::Type{TT}) where TT <: ForwardDiff.Dual = true

function objective1!(out, u, f, left_bc, right_bc, helper, helper_memory)
    # get memory
    Pu, u′, u′′ = helper_memory(eltype(u))
    # get derivatives and endpoint evals
    uₗ, uᵣ = helper(u, Pu, u′, u′′)
    # compute main equation
    out_main = @view out[begin:end-2]
    @. out_main = Pu + Pu*u′ + u′′ - f
    # boundary conditions
    out[end-1] = uₗ - left_bc
    out[end] = uᵣ - right_bc
    return out_main, Pu
end
coupling(Pu, Pv) = Pu*Pv
function coupler!(out, Pu, Pv)
    @. out += coupling(Pu,Pv)
    return nothing
end
function objective!(out, U, f, g, u_leftbc, u_rightbc, v_leftbc, v_rightbc, helper, u_helper_memory, v_helper_memory)
    # split out / U into parts corresponding to each equation
    N = length(U) ÷ 2
    u = @view U[1:N];
    v = @view U[N+1:2N];
    out_u = @view out[1:N];
    out_v = @view out[N+1:2N];
    # call objective1! for u/v
    out_u_main, Pu = objective1!(out_u, u, f, u_leftbc, u_rightbc, helper, u_helper_memory)
    out_v_main, Pv = objective1!(out_v, v, g, v_leftbc, v_rightbc, helper, v_helper_memory)
    # couple the equations
    coupler!(out_u_main, Pu, Pv)
    coupler!(out_v_main, Pv, Pu)
    return out
end

helper = ChebyshevHelper(N);
const ChunkSize = 1
const DT = ForwardDiff.Dual{Nothing, Float64, ChunkSize}
u_dual_memory = DualStruct(
                    HelperMemory(N-2),
                    HelperMemory(N-2, DT)
                );
v_dual_memory = DualStruct(
                    HelperMemory(N-2),
                    HelperMemory(N-2, DT)
                );

################################################################################
# Tech for solving Newton-Krylov problem

struct MyPreconditioner
    LU_Ju::LU{Float64, Matrix{Float64}, Vector{Int64}}
    LU_Jv::LU{Float64, Matrix{Float64}, Vector{Int64}}
end
function LinearAlgebra.mul!(out::Vector{T}, P::MyPreconditioner, in::Vector{T}) where T
    N = length(in) ÷ 2
    u = @view in[1:N]
    v = @view in[N+1:2N]
    out_u = @view out[1:N]
    out_v = @view out[N+1:2N]
    ldiv!(out_u, P.LU_Ju, u)
    ldiv!(out_v, P.LU_Jv, v)
    return out
end
function MyPreconditioner(Ju::Matrix{Float64}, Jv::Matrix{Float64})
    return MyPreconditioner(lu(Ju), lu(Jv))
end

# generate a preconditioner for u/V
Ju = zeros(N, N);
Jv = zeros(N, N);
cfg1 = ForwardDiff.JacobianConfig(nothing, zeros(N), zeros(N), ForwardDiff.Chunk(ChunkSize));
outu = zeros(N);
u = zeros(N);
obju!(outu, u) = objective1!(outu, u, zero(N-2), 0.0, 0.0, helper, u_dual_memory);
ForwardDiff.jacobian!(Ju, obju!, outu, u, cfg1);
objv!(outu, u) = objective1!(outu, u, zero(N-2), 0.0, 0.0, helper, v_dual_memory);
ForwardDiff.jacobian!(Jv, objv!, outu, u, cfg1);

P = MyPreconditioner(Ju, Jv);

# generate Jacobian Vector product
f2 = helper.P2*f;
g2 = helper.P2*g;
obj!(out, U) = objective!(out, U, f2, g2, u₋₁, u₁, v₋₁, v₁, helper, u_dual_memory, v_dual_memory);
U = [1.2*uₐ.+0.1*vₐ; 0.9*vₐ.+0.1*uₐ];
JV! = JacVec(obj!, U; tag=nothing);
# test the speed of this thing
out = zeros(2N);
@b mul!($out, $JV!, $U)
# note that this costs twice what it should cost...

# invert J using Krylov method
GS = GmresSolver(2N, 2N, 100, Vector{Float64});
out = zeros(2N);
obj!(out, U);
Krylov.gmres!(GS, JV!, out, N=P, verbose=3);
ΔU = GS.x;

# full Newton-Krylov solver
O = zeros(2N);
Ou = zeros(N);
Ov = zeros(N);
Ju = zeros(N, N);
Jv = zeros(N, N);
obj!(O, U);
residual = norm(O, Inf);
println("Initial residual: ", residual)
iteration = 1;
while residual > 1e-10
    # setup for Krylov
    u = @view U[1:N];
    v = @view U[N+1:2N];
    JV! = JacVec(obj!, U; tag=nothing);
    ForwardDiff.jacobian!(Ju, obju!, Ou, u, cfg1);
    ForwardDiff.jacobian!(Jv, objv!, Ov, v, cfg1);
    P = MyPreconditioner(Ju, Jv);
    Krylov.gmres!(GS, JV!, O, N=P, verbose=3, atol=1e-12, rtol=1e-12);
    U -= GS.x
    obj!(O, U)
    residual = norm(O, Inf)
    @printf "Residual; it %i: %0.2e\n" iteration residual
    iteration += 1
end;

# extract u/v
uₑ = U[1:N];
vₑ = U[N+1:2N];
# compute error
error_u = norm(uₑ - uₐ, Inf);
error_v = norm(vₑ - vₐ, Inf);
err = max(error_u, error_v);

@printf "Error: %0.2e\n" err

