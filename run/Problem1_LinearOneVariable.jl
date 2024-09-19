using NonlinearExampleforBrownBag
using LinearAlgebra
using Printf
using ForwardDiff

"""
Problem 1: Solve linear problem:
u + u′ + u′′ = f
with Dirichlet boundary conditions
where u(x) = exp(sin(x)) in [-1, 1]
"""
u_function(s) = exp(sin(s))
u′_function(s) = exp(sin(s)) * cos(s)
u′′_function(s) = exp(sin(s)) * (cos(s)^2 - sin(s))

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
# get boundary conditions
u₁ = u_function(1.0);
u₋₁ = u_function(-1.0);
# compute forcing function
f = @. u_function(s) + u′_function(s) + u′′_function(s);

################################################################################
# Solve using classical Linear Method

# construct Linear operator to solve
_LinearOp = P2 + D1 + D2;
LinearOp = [_LinearOp; CLVM'; CRVM'];
# construct right hand side from data
rhs = [P2*f; u₋₁; u₁];
# solve for u
uₑ = LinearOp \ rhs;
# compute error
err = norm(uₑ - uₐ, Inf);

@printf "Error: %0.2e\n" err

################################################################################
# Solve using Autodiff and Newton's Method

# very straightforward implementation of our problem
function objective(u, f, leftbc, rightbc, helper)
    # get derivatives / boundary evaluations
    Pu, u′, u′′, uₗ, uᵣ = helper(u);
    # compute main equation
    main_eq = Pu + u′ + u′′ - f;
    # compute boundary condition
    left_eq = uₗ - leftbc;
    right_eq = uᵣ - rightbc;
    return [main_eq; left_eq; right_eq]
end
obj(u) = objective(u, P2*f, u₋₁, u₁, helper)

# compute the Jacobian
J = ForwardDiff.jacobian(obj, zeros(N));
# solve for u
uₑ = -J \ obj(zeros(N));
# compute error
err = norm(uₑ - uₐ, Inf)
@printf "Error: %0.2e\n" err

# also check to see if J was the same as LinearOp
@printf "Difference, Jacobian vs LinearOp: %0.2e\n" norm(J - LinearOp, Inf)

