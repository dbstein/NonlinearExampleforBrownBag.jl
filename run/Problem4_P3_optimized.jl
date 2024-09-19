using NonlinearExampleforBrownBag
using LinearAlgebra
using Printf
using ForwardDiff
using Chairmarks

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
# Problem 3's version

helper = ChebyshevHelper(N);

# very straightforward implementation of our problem
function objective(U, f, g, u_leftbc, u_rightbc, v_leftbc, v_rightbc, helper)
    u = U[1:N];
    v = U[N+1:2N];
    # get derivatives / boundary evaluations
    Pu, u′, u′′, uₗ, uᵣ = helper(u);
    Pv, v′, v′′, vₗ, vᵣ = helper(v);
    # compute main u/v equations
    u_eq = @. Pu + Pu*u′ + u′′ + Pu*Pv - f;
    v_eq = @. Pv + Pv*v′ + v′′ + Pu*Pv - g;
    # compute boundary conditions
    u_left_eq = uₗ - u_leftbc;
    u_right_eq = uᵣ - u_rightbc;
    v_left_eq = vₗ - v_leftbc;
    v_right_eq = vᵣ - v_rightbc;
    # smush together u/v equations
    full_u_eq = [u_eq; u_left_eq; u_right_eq];
    full_v_eq = [v_eq; v_left_eq; v_right_eq];
    # smush all together
    return [full_u_eq; full_v_eq]
end;
obj(U) = objective(U, helper.P2*f, helper.P2*g, u₋₁, u₁, v₋₁, v₁, helper);

U = [uₐ; vₐ];
@b obj($U)
@b ForwardDiff.jacobian(obj, $U)

################################################################################
# Optimize these two functions

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
    # call objective1! for u/v (easy to toss parallel loop over)
    out_u_main, Pu = objective1!(out_u, u, f, u_leftbc, u_rightbc, helper, u_helper_memory)
    out_v_main, Pv = objective1!(out_v, v, g, v_leftbc, v_rightbc, helper, v_helper_memory)
    # couple the equations (easy to toss parallel loop over)
    coupler!(out_u_main, Pu, Pv)
    coupler!(out_v_main, Pv, Pu)
    return out
end

u_helper_memory = HelperMemory(N-2);
v_helper_memory = HelperMemory(N-2);

f2 = helper.P2*f;
g2 = helper.P2*g;
obj!(out, U) = objective!(out, U, f2, g2, u₋₁, u₁, v₋₁, v₁, helper, u_helper_memory, v_helper_memory);

out = zeros(2N);
U = zeros(2N);
obj!(out, U);
@b obj!($out, $U)

# okay, this works awesome.  let's take its derivative
J = zeros(2N, 2N);
ForwardDiff.jacobian!(J, obj!, out, U);

# whoops, this did not work!  why?
# the problem is that we only have memory buffers for floats

struct DualStruct{T, DT}
    StandardStruct::T
    DualStruct::DT
    DualStruct(StandardStruct, DualStruct) = new{typeof(StandardStruct), typeof(DualStruct)}(StandardStruct, DualStruct)
end
@inline (DS::DualStruct)(::Type{TT}) where TT = DS.StandardStruct(TT)
@inline (DS::DualStruct)(::Type{TT}) where TT <: ForwardDiff.Dual = DS.DualStruct(TT)

const ChunkSize = 4
const DT = ForwardDiff.Dual{Nothing, Float64, ChunkSize}
u_dual_memory = DualStruct(
                    HelperMemory(N-2),
                    HelperMemory(N-2, DT)
                );
v_dual_memory = DualStruct(
                    HelperMemory(N-2),
                    HelperMemory(N-2, DT)
                );

obj!(out, U) = objective!(out, U, f2, g2, u₋₁, u₁, v₋₁, v₁, helper, u_dual_memory, v_dual_memory);
J = zeros(2N, 2N);
cfg = ForwardDiff.JacobianConfig(nothing, zeros(2N), zeros(2N), ForwardDiff.Chunk(ChunkSize));
ForwardDiff.jacobian!(J, obj!, out, U, cfg);
@b ForwardDiff.jacobian!($J, obj!, $out, $U, $cfg)
