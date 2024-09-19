using NonlinearExampleforBrownBag
using LinearAlgebra
using Printf
using ForwardDiff

"""
Problem 2: Solve linear problem:
u + u′ + u′′ + v = f
v + v′ + v′′ - u = g
with Dirichlet boundary conditions
where u(x) = exp(sin(x)) in [-1, 1]
and v(x) = cos(x²) in [-1, 1]
"""
u_function(s) = exp(sin(s))
u′_function(s) = exp(sin(s)) * cos(s)
u′′_function(s) = exp(sin(s)) * (cos(s)^2 - sin(s))
v_function(s) = cos(s^2)
v′_function(s) = -2s * sin(s^2)
v′′_function(s) = -2sin(s^2) - 4s^2*cos(s^2)

# order of discretization
N = 16

# get Chebyhsev points and matrices
s = ChebyshevTPoints(N);
helper = ChebyshevHelper(N);
# collocation projection matrix from N nodes to N-2 nodes
P2 = helper.P2; 
# collocation first derivative matrix, takes in N nodes, gives N-2 nodes
D1 = helper.D1;
# collocation second derivative matrix, takes in N nodes, gives N-2 nodes
D2 = helper.D2;
# left/right eval matrix from N nodes
CLVM = helper.CLVM;
CRVM = helper.CRVM;

# construct analytical solution
uₐ = u_function.(s);
vₐ = v_function.(s);
# get boundary conditions
u₁ = u_function(1.0);
u₋₁ = u_function(-1.0);
v₁ = v_function(1.0);
v₋₁ = v_function(-1.0);
# compute forcing function
f = @. u_function(s) + u′_function(s) + u′′_function(s) + v_function(s);
g = @. v_function(s) + v′_function(s) + v′′_function(s) + u_function(s);

################################################################################
# Solve using classical Linear Method

# construct Linear operator to solve
_LinearOp = P2 + D1 + D2;
LinearOpOne = [_LinearOp; CLVM'; CRVM'];
P2Full = [P2; zeros(2, N)];
LinearOp = [LinearOpOne P2Full; P2Full LinearOpOne];
# construct right hand side from data
rhs = [P2*f; u₋₁; u₁; P2*g; v₋₁; v₁];
# solve for u
Uₑ = LinearOp \ rhs;
uₑ = Uₑ[1:N];
vₑ = Uₑ[N+1:2N];
# compute error
error_u = norm(uₑ - uₐ, Inf);
error_v = norm(vₑ - vₐ, Inf);
err = max(error_u, error_v);

@printf "Error: %0.2e\n" err

################################################################################
# Solve using Autodiff and Newton's Method

# very straightforward implementation of our problem
function objective(U, f, g, u_leftbc, u_rightbc, v_leftbc, v_rightbc, helper)
    u = U[1:N];
    v = U[N+1:2N];
    # get derivatives / boundary evaluations
    Pu, u′, u′′, uₗ, uᵣ = helper(u);
    Pv, v′, v′′, vₗ, vᵣ = helper(v);
    # compute main u/v equations
    u_eq = Pu + u′ + u′′ + Pv - f;
    v_eq = Pv + v′ + v′′ + Pu - g;
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
end
obj(U) = objective(U, P2*f, P2*g, u₋₁, u₁, v₋₁, v₁, helper)

# compute the Jacobian
J = ForwardDiff.jacobian(obj, zeros(2N));
# solve for u
Uₑ = -J \ obj(zeros(2N));
uₑ = Uₑ[1:N];
vₑ = Uₑ[N+1:2N];
# compute error
error_u = norm(uₑ - uₐ, Inf);
error_v = norm(vₑ - vₐ, Inf);
err = max(error_u, error_v);

@printf "Error: %0.2e\n" err
