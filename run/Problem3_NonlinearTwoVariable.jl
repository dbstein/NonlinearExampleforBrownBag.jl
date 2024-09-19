using NonlinearExampleforBrownBag
using LinearAlgebra
using Printf
using ForwardDiff

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
# Solve using Autodiff and Newton's Method

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

# solve for u (starting at a guess nearby)
Uₑ = [1.2*uₐ.+0.1*vₐ; 0.9*vₐ.+0.1*uₐ];
O = obj(Uₑ);
residual = norm(O, Inf);
println("Initial residual: ", residual)
iteration = 1;
while residual > 1e-12
    J = ForwardDiff.jacobian(obj, Uₑ)
    Uₑ -= J \ O
    O = obj(Uₑ)
    residual = norm(O, Inf)
    @printf "Residual; it %i: %0.2e\n" iteration residual
    iteration += 1
end;
# extract u/v
uₑ = Uₑ[1:N];
vₑ = Uₑ[N+1:2N];
# compute error
error_u = norm(uₑ - uₐ, Inf);
error_v = norm(vₑ - vₐ, Inf);
err = max(error_u, error_v);

@printf "Error: %0.2e\n" err



