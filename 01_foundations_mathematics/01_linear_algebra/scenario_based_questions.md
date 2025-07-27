# Linear Algebra Interview Questions - Scenario_Based Questions

## Question 1

**Discuss the importance of linear algebra in optimization problems, such as gradient descent.**

**Answer:** Linear algebra forms the mathematical foundation of optimization algorithms, particularly gradient descent, by providing the framework for efficient computation of gradients, direction vectors, and parameter updates in high-dimensional spaces. Understanding this relationship is crucial for developing and implementing effective optimization strategies in machine learning and scientific computing.

**1. Fundamental Role in Gradient Computation:**

**1.1 Gradient as Linear Transformation:**
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def demonstrate_gradient_computation():
    """Demonstrate gradient computation using linear algebra"""
    
    print("Linear Algebra in Gradient Computation")
    print("=" * 40)
    
    # Example: Quadratic function f(x) = x^T A x + b^T x + c
    # Gradient: ∇f(x) = 2Ax + b
    
    # Define problem
    A = np.array([[2, 1], [1, 3]])  # Positive definite matrix
    b = np.array([1, -2])
    c = 5
    
    def quadratic_function(x):
        """f(x) = x^T A x + b^T x + c"""
        return x.T @ A @ x + b.T @ x + c
    
    def gradient_function(x):
        """∇f(x) = 2Ax + b"""
        return 2 * A @ x + b
    
    def hessian_function():
        """H(x) = 2A (constant for quadratic)"""
        return 2 * A
    
    # Test point
    x_test = np.array([1.0, 0.5])
    
    print(f"Test point x = {x_test}")
    print(f"Function value f(x) = {quadratic_function(x_test):.4f}")
    print(f"Gradient ∇f(x) = {gradient_function(x_test)}")
    print(f"Hessian H(x) = \n{hessian_function()}")
    
    # Verify gradient using finite differences
    h = 1e-8
    grad_numerical = np.zeros(2)
    for i in range(2):
        x_plus = x_test.copy()
        x_minus = x_test.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad_numerical[i] = (quadratic_function(x_plus) - quadratic_function(x_minus)) / (2 * h)
    
    print(f"Numerical gradient = {grad_numerical}")
    print(f"Gradient error = {np.linalg.norm(gradient_function(x_test) - grad_numerical):.2e}")
    
    return A, b, c, quadratic_function, gradient_function, hessian_function

A, b, c, f, grad_f, hess_f = demonstrate_gradient_computation()
```

**1.2 Matrix-Vector Operations in Neural Networks:**
```python
def neural_network_gradients():
    """Demonstrate gradient computation in neural networks using linear algebra"""
    
    print("\nLinear Algebra in Neural Network Gradients:")
    print("-" * 45)
    
    # Simple 2-layer neural network
    # y = W2 * σ(W1 * x + b1) + b2
    # where σ is activation function (ReLU)
    
    np.random.seed(42)
    
    # Network parameters
    input_dim = 3
    hidden_dim = 4
    output_dim = 2
    
    W1 = np.random.randn(hidden_dim, input_dim) * 0.1
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(output_dim, hidden_dim) * 0.1
    b2 = np.zeros(output_dim)
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    def forward_pass(x):
        """Forward pass through network"""
        z1 = W1 @ x + b1  # Linear transformation
        a1 = relu(z1)      # Activation
        z2 = W2 @ a1 + b2  # Linear transformation
        return z1, a1, z2
    
    def backward_pass(x, y_true, z1, a1, z2):
        """Backward pass - compute gradients using chain rule and linear algebra"""
        
        # Output layer gradients
        dL_dz2 = 2 * (z2 - y_true)  # Assuming MSE loss
        dL_dW2 = np.outer(dL_dz2, a1)  # Outer product
        dL_db2 = dL_dz2
        
        # Hidden layer gradients
        dL_da1 = W2.T @ dL_dz2  # Matrix-vector multiplication
        dL_dz1 = dL_da1 * relu_derivative(z1)  # Element-wise multiplication
        dL_dW1 = np.outer(dL_dz1, x)  # Outer product
        dL_db1 = dL_dz1
        
        return dL_dW1, dL_db1, dL_dW2, dL_db2
    
    # Example computation
    x = np.array([1.0, -0.5, 2.0])
    y_true = np.array([1.0, 0.0])
    
    # Forward pass
    z1, a1, z2 = forward_pass(x)
    
    # Backward pass
    dW1, db1, dW2, db2 = backward_pass(x, y_true, z1, a1, z2)
    
    print("Network Architecture:")
    print(f"  Input: {input_dim}D → Hidden: {hidden_dim}D → Output: {output_dim}D")
    print(f"\nForward Pass:")
    print(f"  Input x = {x}")
    print(f"  Hidden z1 = W1 @ x + b1 = {z1}")
    print(f"  Hidden a1 = ReLU(z1) = {a1}")
    print(f"  Output z2 = W2 @ a1 + b2 = {z2}")
    print(f"  Target y = {y_true}")
    
    print(f"\nGradient Shapes (showing matrix operations):")
    print(f"  dL/dW2 shape: {dW2.shape} = outer_product({dL_dz2.shape}, {a1.shape})")
    print(f"  dL/dW1 shape: {dW1.shape} = outer_product({dL_dz1.shape}, {x.shape})")
    
    return W1, b1, W2, b2, dW1, db1, dW2, db2

W1, b1, W2, b2, dW1, db1, dW2, db2 = neural_network_gradients()
```

**2. Gradient Descent Algorithms:**

**2.1 Basic Gradient Descent:**
```python
def gradient_descent_linear_algebra():
    """Implement gradient descent using linear algebra operations"""
    
    print("\nGradient Descent Implementation:")
    print("-" * 35)
    
    # Problem: minimize f(x) = ||Ax - b||^2 (least squares)
    # Gradient: ∇f(x) = 2A^T(Ax - b)
    
    np.random.seed(42)
    m, n = 100, 10  # 100 equations, 10 variables
    
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true + 0.1 * np.random.randn(m)  # Add small noise
    
    def objective(x):
        """f(x) = ||Ax - b||^2"""
        residual = A @ x - b
        return residual.T @ residual
    
    def gradient(x):
        """∇f(x) = 2A^T(Ax - b)"""
        return 2 * A.T @ (A @ x - b)
    
    # Gradient descent
    x = np.random.randn(n)  # Initialize
    learning_rate = 0.001
    max_iterations = 1000
    tolerance = 1e-8
    
    objectives = []
    gradients_norms = []
    
    print(f"Problem size: {m} equations, {n} variables")
    print(f"Condition number of A^T A: {np.linalg.cond(A.T @ A):.2e}")
    
    for iteration in range(max_iterations):
        # Compute objective and gradient
        obj_val = objective(x)
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        objectives.append(obj_val)
        gradients_norms.append(grad_norm)
        
        # Check convergence
        if grad_norm < tolerance:
            print(f"Converged at iteration {iteration}")
            break
        
        # Update step: x = x - α∇f(x)
        x = x - learning_rate * grad
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: f(x) = {obj_val:.6f}, ||∇f|| = {grad_norm:.6f}")
    
    # Compare with analytical solution
    x_analytical = np.linalg.solve(A.T @ A, A.T @ b)
    
    print(f"\nFinal Results:")
    print(f"  Gradient descent solution error: {np.linalg.norm(x - x_analytical):.6f}")
    print(f"  True solution error: {np.linalg.norm(x_analytical - x_true):.6f}")
    
    return x, x_analytical, objectives, gradients_norms

x_gd, x_analytical, objectives, grad_norms = gradient_descent_linear_algebra()
```

**2.2 Newton's Method:**
```python
def newtons_method_linear_algebra():
    """Implement Newton's method using linear algebra"""
    
    print("\nNewton's Method Implementation:")
    print("-" * 30)
    
    # Same problem: minimize f(x) = ||Ax - b||^2
    # Hessian: H(x) = 2A^T A (constant)
    
    # Use same A, b from previous example
    np.random.seed(42)
    m, n = 50, 5  # Smaller problem for Newton's method
    
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true + 0.1 * np.random.randn(m)
    
    def gradient(x):
        return 2 * A.T @ (A @ x - b)
    
    def hessian():
        return 2 * A.T @ A
    
    # Newton's method
    x = np.random.randn(n)
    H = hessian()
    max_iterations = 50
    tolerance = 1e-12
    
    print(f"Hessian condition number: {np.linalg.cond(H):.2e}")
    
    for iteration in range(max_iterations):
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tolerance:
            print(f"Converged at iteration {iteration}")
            break
        
        # Newton step: x = x - H^(-1)∇f(x)
        # Solve Hd = -∇f instead of computing H^(-1)
        newton_direction = np.linalg.solve(H, -grad)
        x = x + newton_direction
        
        if iteration % 5 == 0:
            obj_val = np.linalg.norm(A @ x - b)**2
            print(f"Iteration {iteration}: f(x) = {obj_val:.8f}, ||∇f|| = {grad_norm:.8f}")
    
    # Compare convergence speed
    x_analytical = np.linalg.solve(A.T @ A, A.T @ b)
    final_error = np.linalg.norm(x - x_analytical)
    
    print(f"Final error: {final_error:.2e}")
    print("Newton's method shows quadratic convergence for well-conditioned problems")
    
    return x

x_newton = newtons_method_linear_algebra()
```

**3. Advanced Optimization Techniques:**

**3.1 Conjugate Gradient Method:**
```python
def conjugate_gradient_method():
    """Implement conjugate gradient method using linear algebra"""
    
    print("\nConjugate Gradient Method:")
    print("-" * 25)
    
    # Solve Ax = b where A is symmetric positive definite
    # This is equivalent to minimizing f(x) = (1/2)x^T A x - b^T x
    
    np.random.seed(42)
    n = 8
    
    # Create symmetric positive definite matrix
    Q = np.random.randn(n, n)
    A = Q.T @ Q + np.eye(n)  # Ensure positive definiteness
    
    x_true = np.random.randn(n)
    b = A @ x_true
    
    def conjugate_gradient(A, b, x0=None, max_iter=None, tol=1e-10):
        """
        Solve Ax = b using conjugate gradient method
        """
        
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        if max_iter is None:
            max_iter = n
        
        r = b - A @ x  # Initial residual
        p = r.copy()   # Initial search direction
        
        residuals = [np.linalg.norm(r)]
        
        for iteration in range(max_iter):
            # Check convergence
            if np.linalg.norm(r) < tol:
                print(f"CG converged at iteration {iteration}")
                break
            
            # Compute step size
            Ap = A @ p
            alpha = (r.T @ r) / (p.T @ Ap)
            
            # Update solution
            x = x + alpha * p
            
            # Update residual
            r_new = r - alpha * Ap
            
            # Compute new search direction
            beta = (r_new.T @ r_new) / (r.T @ r)
            p = r_new + beta * p
            
            r = r_new
            residuals.append(np.linalg.norm(r))
            
            if iteration % 2 == 0:
                print(f"Iteration {iteration}: ||r|| = {np.linalg.norm(r):.6e}")
        
        return x, residuals
    
    print(f"Problem size: {n} × {n}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    
    # Solve using CG
    x_cg, residuals = conjugate_gradient(A, b)
    
    # Compare with direct solution
    x_direct = np.linalg.solve(A, b)
    
    print(f"\nResults:")
    print(f"  CG solution error: {np.linalg.norm(x_cg - x_true):.8e}")
    print(f"  Direct solution error: {np.linalg.norm(x_direct - x_true):.8e}")
    print(f"  CG vs Direct error: {np.linalg.norm(x_cg - x_direct):.8e}")
    
    # CG should converge in at most n steps for exact arithmetic
    print(f"  CG iterations: {len(residuals) - 1} (max possible: {n})")
    
    return x_cg, residuals

x_cg, cg_residuals = conjugate_gradient_method()
```

**3.2 Preconditioned Gradient Methods:**
```python
def preconditioned_gradient_descent():
    """Demonstrate preconditioning for improved convergence"""
    
    print("\nPreconditioned Gradient Descent:")
    print("-" * 32)
    
    # Problem with poor conditioning
    np.random.seed(42)
    n = 5
    
    # Create ill-conditioned matrix
    U, _, Vt = np.linalg.svd(np.random.randn(n, n))
    singular_values = np.logspace(0, 3, n)  # Condition number ≈ 1000
    A = U @ np.diag(singular_values) @ Vt
    
    x_true = np.random.randn(n)
    b = A @ x_true
    
    def objective(x):
        return 0.5 * (x.T @ A @ x) - b.T @ x
    
    def gradient(x):
        return A @ x - b
    
    # Standard gradient descent
    def standard_gd(max_iter=1000, lr=0.001):
        x = np.random.randn(n)
        objectives = []
        
        for i in range(max_iter):
            obj_val = objective(x)
            objectives.append(obj_val)
            
            grad = gradient(x)
            if np.linalg.norm(grad) < 1e-10:
                break
                
            x = x - lr * grad
        
        return x, objectives
    
    # Preconditioned gradient descent
    def preconditioned_gd(max_iter=1000, lr=0.01):
        x = np.random.randn(n)
        objectives = []
        
        # Use inverse of A as preconditioner (ideal but expensive)
        # In practice, use approximations like diagonal preconditioning
        P_inv = np.linalg.inv(A)  # Ideal preconditioner
        
        for i in range(max_iter):
            obj_val = objective(x)
            objectives.append(obj_val)
            
            grad = gradient(x)
            if np.linalg.norm(grad) < 1e-10:
                break
            
            # Preconditioned gradient step
            preconditioned_grad = P_inv @ grad
            x = x - lr * preconditioned_grad
        
        return x, objectives
    
    print(f"Matrix condition number: {np.linalg.cond(A):.2e}")
    
    # Compare methods
    x_std, obj_std = standard_gd()
    x_prec, obj_prec = preconditioned_gd()
    x_exact = np.linalg.solve(A, b)
    
    print(f"\nConvergence Comparison:")
    print(f"  Standard GD iterations: {len(obj_std)}")
    print(f"  Preconditioned GD iterations: {len(obj_prec)}")
    print(f"  Standard GD final error: {np.linalg.norm(x_std - x_exact):.6e}")
    print(f"  Preconditioned GD final error: {np.linalg.norm(x_prec - x_exact):.6e}")
    
    # Practical preconditioning strategies
    print(f"\nPractical Preconditioning Strategies:")
    print(f"  1. Diagonal preconditioning: P = diag(A)")
    print(f"  2. Incomplete factorization: P ≈ LL^T")
    print(f"  3. Multigrid methods for PDEs")
    print(f"  4. Adaptive preconditioning")
    
    return x_std, x_prec, obj_std, obj_prec

x_std, x_prec, obj_std, obj_prec = preconditioned_gradient_descent()
```

**4. Linear Algebra in Modern Optimization:**

**4.1 Stochastic Gradient Descent:**
```python
def stochastic_gradient_descent():
    """Demonstrate SGD using linear algebra for mini-batch processing"""
    
    print("\nStochastic Gradient Descent:")
    print("-" * 27)
    
    # Simulate large-scale linear regression
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = X @ w_true + 0.1 * np.random.randn(n_samples)
    
    def batch_gradient(X_batch, y_batch, w):
        """Compute gradient for a batch using vectorized operations"""
        predictions = X_batch @ w
        residuals = predictions - y_batch
        # Gradient: X^T @ residuals / batch_size
        return X_batch.T @ residuals / len(X_batch)
    
    def sgd_linear_regression(X, y, batch_size=32, lr=0.01, epochs=100):
        """SGD for linear regression using efficient linear algebra"""
        
        n_samples, n_features = X.shape
        w = np.random.randn(n_features) * 0.01
        
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Compute gradient using linear algebra
                grad = batch_gradient(X_batch, y_batch, w)
                
                # Update parameters
                w = w - lr * grad
                
                # Compute batch loss
                batch_loss = np.mean((X_batch @ w - y_batch)**2)
                epoch_loss += batch_loss
                n_batches += 1
            
            losses.append(epoch_loss / n_batches)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {losses[-1]:.6f}")
        
        return w, losses
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Batch size: 32")
    
    # Train using SGD
    w_sgd, losses = sgd_linear_regression(X, y)
    
    # Compare with analytical solution
    w_analytical = np.linalg.solve(X.T @ X, X.T @ y)
    
    print(f"\nFinal Results:")
    print(f"  SGD parameter error: {np.linalg.norm(w_sgd - w_true):.6f}")
    print(f"  Analytical parameter error: {np.linalg.norm(w_analytical - w_true):.6f}")
    print(f"  SGD vs Analytical: {np.linalg.norm(w_sgd - w_analytical):.6f}")
    
    return w_sgd, w_analytical, losses

w_sgd, w_analytical, sgd_losses = stochastic_gradient_descent()
```

**4.2 Second-Order Methods:**
```python
def second_order_optimization():
    """Demonstrate second-order optimization methods"""
    
    print("\nSecond-Order Optimization Methods:")
    print("-" * 35)
    
    # Problem: Logistic regression
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    logits = X @ w_true
    probabilities = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probabilities).astype(float)
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def logistic_loss(w, X, y):
        """Logistic loss function"""
        z = X @ w
        p = sigmoid(z)
        return -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
    
    def logistic_gradient(w, X, y):
        """Gradient of logistic loss"""
        z = X @ w
        p = sigmoid(z)
        return X.T @ (p - y) / len(y)
    
    def logistic_hessian(w, X, y):
        """Hessian of logistic loss"""
        z = X @ w
        p = sigmoid(z)
        D = np.diag(p * (1 - p))
        return X.T @ D @ X / len(y)
    
    # Newton's method for logistic regression
    def newton_logistic(X, y, max_iter=20, tol=1e-8):
        w = np.zeros(X.shape[1])
        
        for iteration in range(max_iter):
            grad = logistic_gradient(w, X, y)
            
            if np.linalg.norm(grad) < tol:
                print(f"Newton converged at iteration {iteration}")
                break
            
            hess = logistic_hessian(w, X, y)
            
            # Solve Newton system: H @ d = -grad
            try:
                newton_step = np.linalg.solve(hess, -grad)
                w = w + newton_step
            except np.linalg.LinAlgError:
                print("Hessian is singular, falling back to gradient descent")
                w = w - 0.1 * grad
            
            if iteration % 5 == 0:
                loss = logistic_loss(w, X, y)
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        return w
    
    # Quasi-Newton method (BFGS approximation)
    def quasi_newton_logistic(X, y, max_iter=50, tol=1e-8):
        w = np.zeros(X.shape[1])
        B = np.eye(X.shape[1])  # Initial Hessian approximation
        
        for iteration in range(max_iter):
            grad = logistic_gradient(w, X, y)
            
            if np.linalg.norm(grad) < tol:
                print(f"Quasi-Newton converged at iteration {iteration}")
                break
            
            # Solve B @ p = -grad
            p = np.linalg.solve(B, -grad)
            
            # Line search (simplified)
            alpha = 1.0
            w_new = w + alpha * p
            
            # BFGS update
            if iteration > 0:
                s = w_new - w  # Step
                grad_new = logistic_gradient(w_new, X, y)
                y_bfgs = grad_new - grad  # Gradient difference
                
                # BFGS formula
                if np.dot(s, y_bfgs) > 1e-10:
                    rho = 1.0 / np.dot(s, y_bfgs)
                    B = B - rho * (np.outer(B @ s, s) + np.outer(s, s @ B)) + rho * np.outer(y_bfgs, y_bfgs)
            
            w = w_new
            
            if iteration % 10 == 0:
                loss = logistic_loss(w, X, y)
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        return w
    
    print("Logistic Regression Optimization:")
    print(f"  Dataset: {n_samples} samples, {n_features} features")
    
    # Compare methods
    print("\nNewton's Method:")
    w_newton = newton_logistic(X, y)
    
    print("\nQuasi-Newton (BFGS):")
    w_quasi = quasi_newton_logistic(X, y)
    
    print(f"\nParameter Comparison:")
    print(f"  True parameters: {w_true}")
    print(f"  Newton solution: {w_newton}")
    print(f"  Quasi-Newton solution: {w_quasi}")
    print(f"  Newton error: {np.linalg.norm(w_newton - w_true):.6f}")
    print(f"  Quasi-Newton error: {np.linalg.norm(w_quasi - w_true):.6f}")
    
    return w_newton, w_quasi

w_newton, w_quasi = second_order_optimization()
```

**Key Takeaways:**

**Linear Algebra Fundamentals in Optimization:**
1. **Gradient Computation**: Matrix-vector operations for efficient gradient calculation
2. **Direction Finding**: Linear systems solving for Newton directions
3. **Eigenanalysis**: Understanding convergence through eigenvalues of Hessian
4. **Matrix Conditioning**: Impact on convergence rates and numerical stability
5. **Vectorization**: Efficient batch processing using matrix operations

**Optimization Algorithm Categories:**
1. **First-Order**: Gradient descent, SGD, momentum methods
2. **Second-Order**: Newton's method, quasi-Newton (BFGS, L-BFGS)
3. **Hybrid Methods**: Conjugate gradient, preconditioning
4. **Constrained**: Lagrangian methods, KKT conditions

**Practical Considerations:**
1. **Computational Complexity**: O(n²) for matrix operations, O(n³) for matrix inversion
2. **Memory Requirements**: Dense vs sparse matrix storage
3. **Numerical Stability**: Condition numbers, regularization
4. **Scalability**: Mini-batch processing, distributed computation

Linear algebra provides the mathematical framework that makes modern optimization algorithms both theoretically sound and computationally efficient, enabling the training of complex machine learning models at scale.

---

## Question 2

**How would you handle large-scale matrix operations efficiently in terms of memory and computation?**

**Answer:** Handling large-scale matrix operations efficiently requires a comprehensive strategy combining algorithmic optimization, memory management, numerical techniques, and parallel computing. The key is to leverage matrix structure, use appropriate data representations, and employ scalable algorithms that minimize both computational complexity and memory footprint.

**1. Memory-Efficient Matrix Representations:**

**1.1 Sparse Matrix Storage:**
```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import time
import psutil
import os

def demonstrate_sparse_matrices():
    """Demonstrate memory efficiency of sparse matrices"""
    
    print("Large-Scale Matrix Operations - Memory Efficiency")
    print("=" * 55)
    
    # Create large sparse matrix
    n = 10000
    density = 0.001  # 0.1% non-zero elements
    
    print(f"Matrix size: {n} × {n}")
    print(f"Density: {density*100:.1f}%")
    
    # Generate sparse matrix
    np.random.seed(42)
    nnz = int(n * n * density)
    rows = np.random.randint(0, n, nnz)
    cols = np.random.randint(0, n, nnz)
    data = np.random.randn(nnz)
    
    # Different sparse formats
    A_coo = coo_matrix((data, (rows, cols)), shape=(n, n))
    A_csr = A_coo.tocsr()
    A_csc = A_coo.tocsc()
    
    # Dense matrix (for comparison, smaller size)
    n_dense = 1000
    A_dense = np.random.randn(n_dense, n_dense)
    
    def get_memory_usage(matrix):
        """Get memory usage of matrix"""
        if hasattr(matrix, 'data'):
            # Sparse matrix
            return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        else:
            # Dense matrix
            return matrix.nbytes
    
    print("\nMemory Usage Comparison:")
    print(f"  Dense matrix ({n_dense}×{n_dense}): {get_memory_usage(A_dense) / 1024**2:.1f} MB")
    print(f"  Sparse COO ({n}×{n}): {get_memory_usage(A_coo) / 1024**2:.1f} MB")
    print(f"  Sparse CSR ({n}×{n}): {get_memory_usage(A_csr) / 1024**2:.1f} MB")
    print(f"  Sparse CSC ({n}×{n}): {get_memory_usage(A_csc) / 1024**2:.1f} MB")
    
    theoretical_dense = n * n * 8  # 8 bytes per float64
    print(f"  Theoretical dense ({n}×{n}): {theoretical_dense / 1024**2:.1f} MB")
    
    savings = theoretical_dense / get_memory_usage(A_csr)
    print(f"  Memory savings: {savings:.0f}x")
    
    return A_csr, A_csc, A_coo

A_csr, A_csc, A_coo = demonstrate_sparse_matrices()
```

**1.2 Block Matrix Operations:**
```python
def block_matrix_operations():
    """Demonstrate block-wise matrix operations for memory efficiency"""
    
    print("\nBlock Matrix Operations:")
    print("-" * 25)
    
    # Large matrix broken into blocks
    n = 8000
    block_size = 1000
    
    print(f"Matrix size: {n} × {n}")
    print(f"Block size: {block_size} × {block_size}")
    print(f"Number of blocks: {(n // block_size)**2}")
    
    def create_block_matrix(n, block_size, sparse_prob=0.7):
        """Create block matrix with some sparse blocks"""
        
        n_blocks = n // block_size
        blocks = {}
        
        for i in range(n_blocks):
            for j in range(n_blocks):
                if np.random.rand() > sparse_prob:
                    # Create dense block
                    block = np.random.randn(block_size, block_size)
                    blocks[(i, j)] = block
                # Sparse blocks are implicitly zero (not stored)
        
        return blocks, n_blocks
    
    def block_matrix_multiply(blocks_A, blocks_B, n_blocks, block_size):
        """Multiply two block matrices"""
        
        blocks_C = {}
        
        for i in range(n_blocks):
            for j in range(n_blocks):
                C_ij = np.zeros((block_size, block_size))
                
                for k in range(n_blocks):
                    # Only compute if both blocks exist
                    if (i, k) in blocks_A and (k, j) in blocks_B:
                        C_ij += blocks_A[(i, k)] @ blocks_B[(k, j)]
                
                # Only store non-zero blocks
                if np.any(C_ij != 0):
                    blocks_C[(i, j)] = C_ij
        
        return blocks_C
    
    # Create block matrices
    blocks_A, n_blocks = create_block_matrix(n, block_size, sparse_prob=0.8)
    blocks_B, _ = create_block_matrix(n, block_size, sparse_prob=0.8)
    
    print(f"Matrix A: {len(blocks_A)} non-zero blocks out of {n_blocks**2}")
    print(f"Matrix B: {len(blocks_B)} non-zero blocks out of {n_blocks**2}")
    
    # Measure memory usage
    memory_per_block = block_size * block_size * 8  # 8 bytes per float64
    total_memory_A = len(blocks_A) * memory_per_block
    full_matrix_memory = n * n * 8
    
    print(f"Block matrix A memory: {total_memory_A / 1024**2:.1f} MB")
    print(f"Full matrix memory would be: {full_matrix_memory / 1024**2:.1f} MB")
    print(f"Memory savings: {full_matrix_memory / total_memory_A:.1f}x")
    
    # Time block multiplication
    start_time = time.time()
    blocks_C = block_matrix_multiply(blocks_A, blocks_B, n_blocks, block_size)
    end_time = time.time()
    
    print(f"Block multiplication time: {end_time - start_time:.2f} seconds")
    print(f"Result matrix C: {len(blocks_C)} non-zero blocks")
    
    return blocks_A, blocks_B, blocks_C

blocks_A, blocks_B, blocks_C = block_matrix_operations()
```

**2. Computational Optimization Strategies:**

**2.1 Algorithm Selection and Complexity:**
```python
def algorithm_complexity_analysis():
    """Analyze computational complexity of different algorithms"""
    
    print("\nAlgorithm Complexity Analysis:")
    print("-" * 30)
    
    def time_matrix_operations(n_values):
        """Time different matrix operations for various sizes"""
        
        operations = {
            'Matrix Multiplication': lambda A, B: A @ B,
            'Matrix Inversion': lambda A, B: np.linalg.inv(A),
            'Eigendecomposition': lambda A, B: np.linalg.eig(A),
            'SVD': lambda A, B: np.linalg.svd(A),
            'QR Decomposition': lambda A, B: np.linalg.qr(A),
            'Cholesky': lambda A, B: np.linalg.cholesky(A.T @ A + np.eye(A.shape[1]))
        }
        
        results = {op: [] for op in operations}
        
        for n in n_values:
            print(f"\nMatrix size: {n} × {n}")
            
            # Create test matrices
            np.random.seed(42)
            A = np.random.randn(n, n)
            B = np.random.randn(n, n)
            
            for op_name, op_func in operations.items():
                try:
                    start_time = time.time()
                    result = op_func(A, B)
                    end_time = time.time()
                    
                    elapsed = end_time - start_time
                    results[op_name].append(elapsed)
                    print(f"  {op_name}: {elapsed:.4f}s")
                    
                except Exception as e:
                    results[op_name].append(float('inf'))
                    print(f"  {op_name}: Failed ({str(e)[:50]})")
        
        return results
    
    # Test with increasing matrix sizes
    n_values = [100, 200, 500, 1000]
    
    complexity_results = time_matrix_operations(n_values)
    
    # Analyze complexity trends
    print("\nComplexity Analysis:")
    print("Operation\t\tTheoretical\tObserved Scaling")
    print("-" * 50)
    
    theoretical_complexity = {
        'Matrix Multiplication': 'O(n³)',
        'Matrix Inversion': 'O(n³)',
        'Eigendecomposition': 'O(n³)',
        'SVD': 'O(n³)',
        'QR Decomposition': 'O(n³)',
        'Cholesky': 'O(n³)'
    }
    
    for op_name in complexity_results:
        times = complexity_results[op_name]
        if len(times) >= 2 and all(t != float('inf') for t in times):
            # Estimate scaling from last two points
            ratio = times[-1] / times[-2]
            size_ratio = n_values[-1] / n_values[-2]
            observed_power = np.log(ratio) / np.log(size_ratio)
            
            print(f"{op_name:20s}\t{theoretical_complexity[op_name]}\t~O(n^{observed_power:.1f})")
        else:
            print(f"{op_name:20s}\t{theoretical_complexity[op_name]}\tInsufficient data")
    
    return complexity_results

complexity_results = algorithm_complexity_analysis()
```

**2.2 Optimized BLAS/LAPACK Usage:**
```python
def optimized_linear_algebra():
    """Demonstrate optimized linear algebra operations"""
    
    print("\nOptimized Linear Algebra Operations:")
    print("-" * 35)
    
    # Compare different BLAS levels
    n = 2000
    
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    x = np.random.randn(n)
    
    print(f"Matrix size: {n} × {n}")
    
    # Level 1 BLAS: Vector operations O(n)
    print("\nLevel 1 BLAS (Vector operations):")
    
    start_time = time.time()
    dot_product = np.dot(x, x)
    end_time = time.time()
    print(f"  Vector dot product: {end_time - start_time:.6f}s")
    
    start_time = time.time()
    vector_norm = np.linalg.norm(x)
    end_time = time.time()
    print(f"  Vector norm: {end_time - start_time:.6f}s")
    
    # Level 2 BLAS: Matrix-vector operations O(n²)
    print("\nLevel 2 BLAS (Matrix-vector operations):")
    
    start_time = time.time()
    mv_product = A @ x
    end_time = time.time()
    print(f"  Matrix-vector multiply: {end_time - start_time:.6f}s")
    
    # Level 3 BLAS: Matrix-matrix operations O(n³)
    print("\nLevel 3 BLAS (Matrix-matrix operations):")
    
    start_time = time.time()
    mm_product = A @ B
    end_time = time.time()
    print(f"  Matrix-matrix multiply: {end_time - start_time:.6f}s")
    
    # Demonstrate memory-efficient operations
    print("\nMemory-Efficient Strategies:")
    
    # In-place operations
    C = A.copy()
    start_time = time.time()
    C += B  # In-place addition
    end_time = time.time()
    print(f"  In-place addition: {end_time - start_time:.6f}s")
    
    # Views vs copies
    start_time = time.time()
    A_view = A[:n//2, :n//2]  # View (no copy)
    end_time = time.time()
    print(f"  Creating view: {end_time - start_time:.6f}s")
    
    start_time = time.time()
    A_copy = A[:n//2, :n//2].copy()  # Explicit copy
    end_time = time.time()
    print(f"  Creating copy: {end_time - start_time:.6f}s")
    
    # Memory layout optimization
    print("\nMemory Layout Impact:")
    
    # Row-major (C-style) vs column-major (Fortran-style)
    A_c = np.random.randn(n, n)  # Default: C-style
    A_f = np.asfortranarray(A_c)  # Fortran-style
    
    # Row access
    start_time = time.time()
    row_sum_c = np.sum(A_c[100, :])
    end_time = time.time()
    time_row_c = end_time - start_time
    
    start_time = time.time()
    row_sum_f = np.sum(A_f[100, :])
    end_time = time.time()
    time_row_f = end_time - start_time
    
    # Column access
    start_time = time.time()
    col_sum_c = np.sum(A_c[:, 100])
    end_time = time.time()
    time_col_c = end_time - start_time
    
    start_time = time.time()
    col_sum_f = np.sum(A_f[:, 100])
    end_time = time.time()
    time_col_f = end_time - start_time
    
    print(f"  Row access - C-style: {time_row_c:.6f}s, F-style: {time_row_f:.6f}s")
    print(f"  Col access - C-style: {time_col_c:.6f}s, F-style: {time_col_f:.6f}s")

optimized_linear_algebra()
```

**3. Parallel and Distributed Computing:**

**3.1 Multi-threading and Vectorization:**
```python
def parallel_matrix_operations():
    """Demonstrate parallel matrix operations"""
    
    print("\nParallel Matrix Operations:")
    print("-" * 25)
    
    n = 2000
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    # NumPy automatically uses multi-threading for large operations
    # through optimized BLAS libraries (OpenBLAS, MKL, etc.)
    
    print(f"Matrix size: {n} × {n}")
    print("NumPy automatically uses multithreading via BLAS libraries")
    
    # Manual parallel processing for custom operations
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    
    def parallel_element_wise_operation(A, B, n_workers=None):
        """Demonstrate manual parallelization"""
        
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        
        def process_chunk(args):
            start_row, end_row, A_chunk, B_chunk = args
            return A_chunk * B_chunk + np.sin(A_chunk)
        
        # Split matrix into chunks
        chunk_size = n // n_workers
        chunks = []
        
        for i in range(n_workers):
            start_row = i * chunk_size
            end_row = min((i + 1) * chunk_size, n)
            A_chunk = A[start_row:end_row, :]
            B_chunk = B[start_row:end_row, :]
            chunks.append((start_row, end_row, A_chunk, B_chunk))
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            start_time = time.time()
            results = list(executor.map(process_chunk, chunks))
            end_time = time.time()
        
        # Combine results
        result = np.vstack(results)
        
        return result, end_time - start_time
    
    # Compare serial vs parallel
    start_time = time.time()
    result_serial = A * B + np.sin(A)
    time_serial = time.time() - start_time
    
    result_parallel, time_parallel = parallel_element_wise_operation(A, B)
    
    print(f"\nCustom Operation Timing:")
    print(f"  Serial execution: {time_serial:.4f}s")
    print(f"  Parallel execution: {time_parallel:.4f}s")
    print(f"  Speedup: {time_serial/time_parallel:.2f}x")
    print(f"  Results match: {np.allclose(result_serial, result_parallel)}")
    
    # Built-in parallel matrix multiplication
    start_time = time.time()
    C_builtin = A @ B
    time_builtin = time.time() - start_time
    
    print(f"\nBuilt-in Matrix Multiplication:")
    print(f"  Time: {time_builtin:.4f}s (automatically parallelized)")
    
    return result_serial, result_parallel

result_serial, result_parallel = parallel_matrix_operations()
```

**3.2 Out-of-Core Processing:**
```python
def out_of_core_processing():
    """Demonstrate out-of-core matrix operations for very large matrices"""
    
    print("\nOut-of-Core Matrix Processing:")
    print("-" * 30)
    
    # Simulate very large matrix using memory mapping
    import tempfile
    
    n = 5000
    chunk_size = 1000
    
    print(f"Matrix size: {n} × {n}")
    print(f"Chunk size: {chunk_size} × {chunk_size}")
    
    # Create temporary files for matrices
    with tempfile.NamedTemporaryFile(delete=False) as f_A:
        A_filename = f_A.name
    with tempfile.NamedTemporaryFile(delete=False) as f_B:
        B_filename = f_B.name
    with tempfile.NamedTemporaryFile(delete=False) as f_C:
        C_filename = f_C.name
    
    try:
        # Create memory-mapped matrices
        A_mmap = np.memmap(A_filename, dtype='float64', mode='w+', shape=(n, n))
        B_mmap = np.memmap(B_filename, dtype='float64', mode='w+', shape=(n, n))
        C_mmap = np.memmap(C_filename, dtype='float64', mode='w+', shape=(n, n))
        
        # Initialize matrices in chunks to avoid loading everything into memory
        print("Initializing matrices...")
        for i in range(0, n, chunk_size):
            for j in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                end_j = min(j + chunk_size, n)
                
                A_mmap[i:end_i, j:end_j] = np.random.randn(end_i - i, end_j - j)
                B_mmap[i:end_i, j:end_j] = np.random.randn(end_i - i, end_j - j)
        
        # Block matrix multiplication: C = A @ B
        print("Performing block matrix multiplication...")
        start_time = time.time()
        
        for i in range(0, n, chunk_size):
            for j in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                end_j = min(j + chunk_size, n)
                
                C_block = np.zeros((end_i - i, end_j - j))
                
                for k in range(0, n, chunk_size):
                    end_k = min(k + chunk_size, n)
                    
                    # Load chunks into memory for computation
                    A_chunk = A_mmap[i:end_i, k:end_k]
                    B_chunk = B_mmap[k:end_k, j:end_j]
                    
                    # Accumulate result
                    C_block += A_chunk @ B_chunk
                
                # Store result
                C_mmap[i:end_i, j:end_j] = C_block
        
        end_time = time.time()
        
        print(f"Block multiplication completed in {end_time - start_time:.2f}s")
        
        # Verify a small portion
        test_size = 100
        A_test = A_mmap[:test_size, :test_size]
        B_test = B_mmap[:test_size, :test_size]
        C_test = C_mmap[:test_size, :test_size]
        C_verify = A_test @ B_test
        
        error = np.max(np.abs(C_test - C_verify))
        print(f"Verification error (max absolute): {error:.2e}")
        
        # Memory usage analysis
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024**2
        theoretical_memory = 3 * n * n * 8 / 1024**2  # 3 matrices, 8 bytes each
        
        print(f"\nMemory Usage:")
        print(f"  Current process memory: {memory_usage:.1f} MB")
        print(f"  Theoretical memory for 3 dense matrices: {theoretical_memory:.1f} MB")
        print(f"  Memory savings: {theoretical_memory / memory_usage:.1f}x")
        
    finally:
        # Clean up temporary files
        import os
        for filename in [A_filename, B_filename, C_filename]:
            try:
                os.unlink(filename)
            except:
                pass

out_of_core_processing()
```

**4. Specialized Algorithms and Data Structures:**

**4.1 Iterative Solvers:**
```python
def iterative_solvers():
    """Demonstrate iterative solvers for large linear systems"""
    
    print("\nIterative Solvers for Large Systems:")
    print("-" * 35)
    
    from scipy.sparse.linalg import cg, gmres, bicgstab
    from scipy.sparse import diags
    
    # Create large sparse system Ax = b
    n = 10000
    
    # Create sparse matrix (tridiagonal for demonstration)
    diagonals = [
        -np.ones(n-1),    # Lower diagonal
        2*np.ones(n),     # Main diagonal  
        -np.ones(n-1)     # Upper diagonal
    ]
    offsets = [-1, 0, 1]
    A_sparse = diags(diagonals, offsets, shape=(n, n), format='csr')
    
    # Right-hand side
    np.random.seed(42)
    b = np.random.randn(n)
    
    print(f"System size: {n} × {n}")
    print(f"Matrix sparsity: {1 - A_sparse.nnz / (n*n):.4f}")
    print(f"Condition number estimate: {np.linalg.cond(A_sparse[:100, :100].toarray()):.2e}")
    
    # Compare different iterative solvers
    solvers = {
        'Conjugate Gradient': lambda A, b: cg(A, b, tol=1e-8),
        'GMRES': lambda A, b: gmres(A, b, tol=1e-8),
        'BiCGSTAB': lambda A, b: bicgstab(A, b, tol=1e-8)
    }
    
    results = {}
    
    for solver_name, solver_func in solvers.items():
        print(f"\n{solver_name}:")
        
        start_time = time.time()
        try:
            x, info = solver_func(A_sparse, b)
            end_time = time.time()
            
            if info == 0:
                residual = np.linalg.norm(A_sparse @ x - b)
                print(f"  Converged in {end_time - start_time:.4f}s")
                print(f"  Residual norm: {residual:.2e}")
                results[solver_name] = {'time': end_time - start_time, 'residual': residual}
            else:
                print(f"  Failed to converge (info={info})")
                results[solver_name] = {'time': float('inf'), 'residual': float('inf')}
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[solver_name] = {'time': float('inf'), 'residual': float('inf')}
    
    # Compare with direct solver (for smaller system)
    n_direct = 1000
    A_direct = A_sparse[:n_direct, :n_direct]
    b_direct = b[:n_direct]
    
    start_time = time.time()
    x_direct = sp.linalg.spsolve(A_direct, b_direct)
    time_direct = time.time() - start_time
    
    print(f"\nDirect solver (n={n_direct}):")
    print(f"  Time: {time_direct:.4f}s")
    print(f"  Residual: {np.linalg.norm(A_direct @ x_direct - b_direct):.2e}")
    
    return results

iterative_results = iterative_solvers()
```

**4.2 Approximation Methods:**
```python
def matrix_approximation_methods():
    """Demonstrate matrix approximation techniques for large-scale problems"""
    
    print("\nMatrix Approximation Methods:")
    print("-" * 30)
    
    from scipy.sparse.linalg import svds
    from sklearn.decomposition import IncrementalPCA
    from sklearn.random_projection import GaussianRandomProjection
    
    # Create large low-rank matrix
    n, m = 5000, 3000
    rank = 50
    
    np.random.seed(42)
    U = np.random.randn(n, rank)
    V = np.random.randn(rank, m)
    A = U @ V + 0.1 * np.random.randn(n, m)  # Low-rank + noise
    
    print(f"Matrix size: {n} × {m}")
    print(f"True rank (approximately): {rank}")
    
    # Method 1: Truncated SVD
    print("\n1. Truncated SVD:")
    start_time = time.time()
    U_svd, s_svd, Vt_svd = svds(A, k=rank)
    time_svd = time.time() - start_time
    
    A_svd = U_svd @ np.diag(s_svd) @ Vt_svd
    error_svd = np.linalg.norm(A - A_svd, 'fro')
    
    print(f"  Time: {time_svd:.4f}s")
    print(f"  Reconstruction error: {error_svd:.4f}")
    
    # Method 2: Random Projection
    print("\n2. Random Projection:")
    target_dim = rank * 2
    
    start_time = time.time()
    rp = GaussianRandomProjection(n_components=target_dim)
    A_projected = rp.fit_transform(A)
    time_rp = time.time() - start_time
    
    print(f"  Time: {time_rp:.4f}s")
    print(f"  Reduced dimensions: {n} → {target_dim}")
    print(f"  Compression ratio: {n / target_dim:.1f}x")
    
    # Method 3: Incremental PCA (for streaming data)
    print("\n3. Incremental PCA:")
    batch_size = 500
    
    start_time = time.time()
    ipca = IncrementalPCA(n_components=rank, batch_size=batch_size)
    
    # Process in batches
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        ipca.partial_fit(A[i:end_i, :])
    
    A_ipca = ipca.transform(A)
    A_reconstructed = ipca.inverse_transform(A_ipca)
    time_ipca = time.time() - start_time
    
    error_ipca = np.linalg.norm(A - A_reconstructed, 'fro')
    
    print(f"  Time: {time_ipca:.4f}s")
    print(f"  Reconstruction error: {error_ipca:.4f}")
    print(f"  Memory efficient: processes in batches of {batch_size}")
    
    # Method 4: Randomized SVD
    print("\n4. Randomized SVD:")
    
    def randomized_svd(A, k, n_iter=2):
        """Randomized SVD algorithm"""
        n, m = A.shape
        
        # Random matrix
        Omega = np.random.randn(m, k + 10)  # Oversampling
        
        # Power iteration for better approximation
        Y = A @ Omega
        for _ in range(n_iter):
            Y = A @ (A.T @ Y)
        
        # QR decomposition
        Q, _ = np.linalg.qr(Y)
        
        # Project A onto Q
        B = Q.T @ A
        
        # SVD of smaller matrix B
        U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)
        
        # Reconstruct U
        U = Q @ U_tilde
        
        return U[:, :k], s[:k], Vt[:k, :]
    
    start_time = time.time()
    U_rand, s_rand, Vt_rand = randomized_svd(A, rank)
    time_rand = time.time() - start_time
    
    A_rand = U_rand @ np.diag(s_rand) @ Vt_rand
    error_rand = np.linalg.norm(A - A_rand, 'fro')
    
    print(f"  Time: {time_rand:.4f}s")
    print(f"  Reconstruction error: {error_rand:.4f}")
    
    # Comparison
    print(f"\nMethod Comparison:")
    print(f"  Truncated SVD: {time_svd:.4f}s, error: {error_svd:.4f}")
    print(f"  Incremental PCA: {time_ipca:.4f}s, error: {error_ipca:.4f}")
    print(f"  Randomized SVD: {time_rand:.4f}s, error: {error_rand:.4f}")
    print(f"  Random Projection: {time_rp:.4f}s (dimensionality reduction)")
    
    return {
        'svd': (time_svd, error_svd),
        'ipca': (time_ipca, error_ipca),
        'randomized': (time_rand, error_rand),
        'random_proj': (time_rp, None)
    }

approximation_results = matrix_approximation_methods()
```

**5. Best Practices Summary:**

```python
def best_practices_summary():
    """Summarize best practices for large-scale matrix operations"""
    
    print("\nBest Practices for Large-Scale Matrix Operations")
    print("=" * 55)
    
    practices = {
        "Memory Management": [
            "Use sparse matrices when density < 10%",
            "Employ memory mapping for very large matrices",
            "Process data in blocks/chunks",
            "Use views instead of copies when possible",
            "Consider different storage formats (CSR, CSC, COO)"
        ],
        
        "Computational Efficiency": [
            "Leverage optimized BLAS/LAPACK libraries",
            "Use vectorized operations over loops",
            "Choose algorithms based on matrix properties",
            "Employ iterative solvers for large sparse systems",
            "Use approximation methods when exact solutions not needed"
        ],
        
        "Parallel Computing": [
            "Utilize multi-threaded BLAS operations",
            "Implement block-parallel algorithms",
            "Consider GPU acceleration for suitable problems",
            "Use distributed computing for extremely large problems",
            "Balance computation and communication costs"
        ],
        
        "Numerical Stability": [
            "Monitor condition numbers",
            "Use regularization for ill-conditioned problems",
            "Employ stable algorithms (QR, SVD over normal equations)",
            "Consider iterative refinement",
            "Use appropriate precision (single vs double)"
        ],
        
        "Algorithm Selection": [
            "Direct methods: small to medium problems",
            "Iterative methods: large sparse problems",
            "Randomized algorithms: very large low-rank problems",
            "Streaming algorithms: data that doesn't fit in memory",
            "Specialized algorithms: structured matrices"
        ]
    }
    
    for category, items in practices.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print(f"\nPerformance Guidelines:")
    print(f"  • Matrix size < 1K: Any method works")
    print(f"  • Matrix size 1K-10K: Consider algorithm choice")
    print(f"  • Matrix size 10K-100K: Use sparse/block methods")
    print(f"  • Matrix size > 100K: Out-of-core/distributed methods")
    print(f"  • Sparsity > 90%: Always use sparse methods")
    print(f"  • Low-rank structure: Use approximation methods")

best_practices_summary()
```

**Key Strategies for Large-Scale Matrix Operations:**

1. **Data Structure Optimization**: Sparse matrices, block formats, memory mapping
2. **Algorithm Selection**: Iterative solvers, randomized methods, approximation techniques
3. **Computational Optimization**: Vectorization, BLAS utilization, parallel processing
4. **Memory Management**: Out-of-core processing, streaming algorithms, chunking
5. **Problem-Specific Approaches**: Leverage matrix structure, use domain knowledge

The choice of strategy depends on matrix characteristics (size, sparsity, structure), computational resources, and accuracy requirements. Successful large-scale matrix computation requires balancing these factors to achieve optimal performance.

---

## Question 3

**Propose a method for dimensionality reduction using linear algebra techniques.**

**Answer:** Dimensionality reduction through linear algebra techniques is essential for handling high-dimensional data in machine learning and data analysis. I'll present a comprehensive approach using multiple linear algebraic methods, focusing on Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and advanced techniques like Random Projections and Matrix Factorization.

**1. Principal Component Analysis (PCA) - Foundation:**

**1.1 Mathematical Framework:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.manifold import Isomap
import seaborn as sns
import time

def mathematical_pca_foundation():
    """Demonstrate the mathematical foundation of PCA"""
    
    print("Mathematical Foundation of PCA")
    print("=" * 30)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Create correlated data
    mean = np.zeros(n_features)
    # Create covariance matrix with known structure
    cov = np.array([
        [4.0, 2.0, 1.0, 0.5, 0.2],
        [2.0, 3.0, 1.5, 0.3, 0.1],
        [1.0, 1.5, 2.0, 0.8, 0.4],
        [0.5, 0.3, 0.8, 1.5, 0.6],
        [0.2, 0.1, 0.4, 0.6, 1.0]
    ])
    
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    print(f"Original data shape: {X.shape}")
    print(f"Original covariance matrix:\n{np.cov(X.T)}")
    
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    print(f"\nCovariance matrix shape: {cov_matrix.shape}")
    
    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors shape: {eigenvectors.shape}")
    
    # Step 4: Compute explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"\nExplained variance ratio: {explained_variance_ratio}")
    print(f"Cumulative variance: {cumulative_variance}")
    
    # Step 5: Select number of components (95% variance)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Components needed for 95% variance: {n_components}")
    
    # Step 6: Transform data
    principal_components = eigenvectors[:, :n_components]
    X_transformed = X_centered @ principal_components
    
    print(f"Transformed data shape: {X_transformed.shape}")
    
    # Step 7: Reconstruction
    X_reconstructed = X_transformed @ principal_components.T + np.mean(X, axis=0)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    
    print(f"Reconstruction error (MSE): {reconstruction_error:.6f}")
    
    return X, X_transformed, eigenvalues, eigenvectors, explained_variance_ratio

X_orig, X_pca, eigenvals, eigenvecs, var_ratio = mathematical_pca_foundation()
```

**1.2 Advanced PCA Implementation:**
```python
def advanced_pca_implementation():
    """Advanced PCA implementation with multiple variants"""
    
    print("\nAdvanced PCA Implementation:")
    print("-" * 28)
    
    # Generate high-dimensional dataset
    n_samples = 2000
    n_features = 50
    n_informative = 10
    n_target_components = 5
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=20,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    class AdvancedPCA:
        """Advanced PCA with multiple algorithms"""
        
        def __init__(self, n_components, method='eigen'):
            self.n_components = n_components
            self.method = method
            self.components_ = None
            self.explained_variance_ = None
            self.explained_variance_ratio_ = None
            self.mean_ = None
            
        def fit_eigen(self, X):
            """Eigendecomposition-based PCA"""
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            
            # Covariance matrix
            cov_matrix = np.cov(X_centered.T)
            
            # Eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Sort descending
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Select components
            self.components_ = eigenvecs[:, :self.n_components].T
            self.explained_variance_ = eigenvals[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvals)
            
        def fit_svd(self, X):
            """SVD-based PCA"""
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            
            # SVD: X = U @ S @ Vt
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # Components are rows of Vt
            self.components_ = Vt[:self.n_components]
            
            # Explained variance from singular values
            self.explained_variance_ = (s[:self.n_components] ** 2) / (X.shape[0] - 1)
            total_variance = np.sum(s ** 2) / (X.shape[0] - 1)
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        def fit_kernel(self, X, kernel='rbf', gamma=1.0):
            """Kernel PCA"""
            n_samples = X.shape[0]
            
            if kernel == 'rbf':
                # RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
                pairwise_sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + \
                                   np.sum(X**2, axis=1).reshape(1, -1) - \
                                   2 * np.dot(X, X.T)
                K = np.exp(-gamma * pairwise_sq_dists)
            elif kernel == 'poly':
                # Polynomial kernel: K(x,y) = (x^T y + 1)^d
                K = (np.dot(X, X.T) + 1) ** 2
            else:
                # Linear kernel
                K = np.dot(X, X.T)
            
            # Center kernel matrix
            one_n = np.ones((n_samples, n_samples)) / n_samples
            K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
            
            # Eigendecomposition of centered kernel matrix
            eigenvals, eigenvecs = np.linalg.eigh(K_centered)
            
            # Sort descending
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Normalize eigenvectors
            for i in range(self.n_components):
                if eigenvals[i] > 0:
                    eigenvecs[:, i] /= np.sqrt(eigenvals[i])
            
            self.components_ = eigenvecs[:, :self.n_components].T
            self.explained_variance_ = eigenvals[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvals)
            self.kernel_matrix_ = K
            self.alphas_ = eigenvecs[:, :self.n_components]
        
        def fit(self, X):
            """Fit PCA using specified method"""
            if self.method == 'eigen':
                self.fit_eigen(X)
            elif self.method == 'svd':
                self.fit_svd(X)
            elif self.method in ['kernel_rbf', 'kernel_poly', 'kernel_linear']:
                kernel_type = self.method.split('_')[1]
                self.fit_kernel(X, kernel=kernel_type)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            return self
        
        def transform(self, X):
            """Transform data"""
            if self.method.startswith('kernel'):
                # For kernel PCA, need to compute kernel with training data
                # This is simplified - full implementation would store training data
                return self.alphas_.T  # Simplified
            else:
                return (X - self.mean_) @ self.components_.T
    
    # Compare different PCA methods
    methods = ['eigen', 'svd', 'kernel_rbf']
    results = {}
    
    for method in methods:
        print(f"\n{method.upper()} PCA:")
        
        start_time = time.time()
        pca = AdvancedPCA(n_components=n_target_components, method=method)
        pca.fit(X)
        
        if not method.startswith('kernel'):
            X_transformed = pca.transform(X)
        else:
            X_transformed = pca.alphas_  # Kernel PCA components
        
        fit_time = time.time() - start_time
        
        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        print(f"  Component shape: {pca.components_.shape}")
        print(f"  Transformed shape: {X_transformed.shape}")
        
        results[method] = {
            'pca': pca,
            'X_transformed': X_transformed,
            'time': fit_time
        }
    
    return results, X, y

pca_results, X_data, y_data = advanced_pca_implementation()
```

**2. Linear Discriminant Analysis (LDA):**

```python
def implement_lda_dimensionality_reduction():
    """Implement Linear Discriminant Analysis for supervised dimensionality reduction"""
    
    print("\nLinear Discriminant Analysis (LDA):")
    print("-" * 32)
    
    # Generate multi-class dataset
    n_samples = 1500
    n_features = 20
    n_classes = 3
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    
    class CustomLDA:
        """Custom LDA implementation"""
        
        def __init__(self, n_components=None):
            self.n_components = n_components
            self.components_ = None
            self.means_ = None
            self.overall_mean_ = None
            
        def fit(self, X, y):
            """Fit LDA"""
            n_samples, n_features = X.shape
            classes = np.unique(y)
            n_classes = len(classes)
            
            # Maximum components = min(n_features, n_classes - 1)
            max_components = min(n_features, n_classes - 1)
            if self.n_components is None:
                self.n_components = max_components
            else:
                self.n_components = min(self.n_components, max_components)
            
            # Compute class means
            self.means_ = {}
            for class_label in classes:
                self.means_[class_label] = np.mean(X[y == class_label], axis=0)
            
            # Overall mean
            self.overall_mean_ = np.mean(X, axis=0)
            
            # Within-class scatter matrix Sw
            Sw = np.zeros((n_features, n_features))
            for class_label in classes:
                X_class = X[y == class_label]
                class_mean = self.means_[class_label]
                
                # Center class data
                X_centered = X_class - class_mean
                Sw += X_centered.T @ X_centered
            
            # Between-class scatter matrix Sb
            Sb = np.zeros((n_features, n_features))
            for class_label in classes:
                n_class = np.sum(y == class_label)
                mean_diff = (self.means_[class_label] - self.overall_mean_).reshape(-1, 1)
                Sb += n_class * (mean_diff @ mean_diff.T)
            
            # Solve generalized eigenvalue problem: Sb @ v = λ @ Sw @ v
            # Equivalent to: inv(Sw) @ Sb @ v = λ @ v
            try:
                Sw_inv = np.linalg.inv(Sw)
                eigenvals, eigenvecs = np.linalg.eig(Sw_inv @ Sb)
                
                # Sort by eigenvalues (descending)
                idx = np.argsort(eigenvals.real)[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                
                # Select top components
                self.components_ = eigenvecs[:, :self.n_components].T.real
                self.eigenvalues_ = eigenvals[:self.n_components].real
                
            except np.linalg.LinAlgError:
                # If Sw is singular, use pseudoinverse
                Sw_pinv = np.linalg.pinv(Sw)
                eigenvals, eigenvecs = np.linalg.eig(Sw_pinv @ Sb)
                
                idx = np.argsort(eigenvals.real)[::-1]
                self.components_ = eigenvecs[:, idx][:, :self.n_components].T.real
                self.eigenvalues_ = eigenvals[idx][:self.n_components].real
            
            return self
        
        def transform(self, X):
            """Transform data using LDA components"""
            return X @ self.components_.T
    
    # Compare custom LDA with sklearn
    print("\nComparing LDA implementations:")
    
    # Custom LDA
    start_time = time.time()
    custom_lda = CustomLDA(n_components=2)
    custom_lda.fit(X, y)
    X_custom = custom_lda.transform(X)
    custom_time = time.time() - start_time
    
    print(f"Custom LDA:")
    print(f"  Fit time: {custom_time:.4f}s")
    print(f"  Components shape: {custom_lda.components_.shape}")
    print(f"  Eigenvalues: {custom_lda.eigenvalues_}")
    
    # Sklearn LDA
    start_time = time.time()
    sklearn_lda = LinearDiscriminantAnalysis(n_components=2)
    X_sklearn = sklearn_lda.fit_transform(X, y)
    sklearn_time = time.time() - start_time
    
    print(f"\nSklearn LDA:")
    print(f"  Fit time: {sklearn_time:.4f}s")
    print(f"  Components shape: {sklearn_lda.components_.shape}")
    print(f"  Explained variance ratio: {sklearn_lda.explained_variance_ratio_}")
    
    # Compare results
    component_similarity = np.abs(np.corrcoef(
        custom_lda.components_.flatten(),
        sklearn_lda.components_.flatten()
    )[0, 1])
    
    print(f"\nComponent correlation: {component_similarity:.4f}")
    
    return custom_lda, sklearn_lda, X, y, X_custom, X_sklearn

custom_lda, sklearn_lda, X_lda, y_lda, X_custom_lda, X_sklearn_lda = implement_lda_dimensionality_reduction()
```

**3. Random Projections and Johnson-Lindenstrauss:**

```python
def implement_random_projections():
    """Implement random projection methods for dimensionality reduction"""
    
    print("\nRandom Projection Methods:")
    print("-" * 25)
    
    # High-dimensional dataset
    n_samples = 5000
    n_features = 1000
    target_dim = 100
    
    # Generate sparse high-dimensional data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure
    X[:, :50] *= 3  # First 50 features have higher variance
    X[:, 50:100] += np.random.randn(n_samples, 50) * 2  # Correlated features
    
    print(f"Original data: {X.shape}")
    print(f"Target dimensions: {target_dim}")
    
    # Johnson-Lindenstrauss bound
    def johnson_lindenstrauss_bound(n_samples, eps=0.1):
        """Compute JL bound for minimum dimensions"""
        return int(4 * np.log(n_samples) / (eps**2 / 2 - eps**3 / 3))
    
    eps_values = [0.1, 0.2, 0.3]
    print(f"\nJohnson-Lindenstrauss bounds:")
    for eps in eps_values:
        min_dim = johnson_lindenstrauss_bound(n_samples, eps)
        print(f"  ε = {eps}: minimum {min_dim} dimensions")
    
    class RandomProjection:
        """Custom random projection implementation"""
        
        def __init__(self, n_components, projection_type='gaussian'):
            self.n_components = n_components
            self.projection_type = projection_type
            self.projection_matrix_ = None
            
        def fit(self, X):
            """Generate random projection matrix"""
            n_features = X.shape[1]
            
            if self.projection_type == 'gaussian':
                # Gaussian random projection
                self.projection_matrix_ = np.random.normal(
                    0, 1/np.sqrt(self.n_components), 
                    size=(self.n_components, n_features)
                )
            
            elif self.projection_type == 'sparse':
                # Sparse random projection (Achlioptas)
                # Elements are -1, 0, +1 with probabilities 1/6, 2/3, 1/6
                self.projection_matrix_ = np.zeros((self.n_components, n_features))
                
                for i in range(self.n_components):
                    for j in range(n_features):
                        rand_val = np.random.random()
                        if rand_val < 1/6:
                            self.projection_matrix_[i, j] = -np.sqrt(3)
                        elif rand_val > 5/6:
                            self.projection_matrix_[i, j] = np.sqrt(3)
                        # else remains 0
            
            elif self.projection_type == 'very_sparse':
                # Very sparse projection (Li et al.)
                # Density s = 1/sqrt(n_features)
                density = 1 / np.sqrt(n_features)
                self.projection_matrix_ = np.zeros((self.n_components, n_features))
                
                for i in range(self.n_components):
                    # Select random subset of features
                    n_nonzero = int(density * n_features)
                    indices = np.random.choice(n_features, n_nonzero, replace=False)
                    values = np.random.choice([-1, 1], n_nonzero)
                    self.projection_matrix_[i, indices] = values / np.sqrt(density)
            
            return self
        
        def transform(self, X):
            """Apply random projection"""
            return X @ self.projection_matrix_.T
    
    # Test different projection methods
    projection_methods = ['gaussian', 'sparse', 'very_sparse']
    results = {}
    
    # Original pairwise distances (sample)
    sample_indices = np.random.choice(n_samples, 200, replace=False)
    X_sample = X[sample_indices]
    original_distances = np.linalg.norm(
        X_sample[:, None] - X_sample[None, :], axis=2
    )
    
    for method in projection_methods:
        print(f"\n{method.upper()} Random Projection:")
        
        start_time = time.time()
        
        rp = RandomProjection(n_components=target_dim, projection_type=method)
        rp.fit(X)
        X_projected = rp.transform(X)
        
        projection_time = time.time() - start_time
        
        # Compute projected distances
        X_proj_sample = X_projected[sample_indices]
        projected_distances = np.linalg.norm(
            X_proj_sample[:, None] - X_proj_sample[None, :], axis=2
        )
        
        # Distance preservation analysis
        distance_ratios = projected_distances / (original_distances + 1e-10)
        mean_ratio = np.mean(distance_ratios[original_distances > 0])
        std_ratio = np.std(distance_ratios[original_distances > 0])
        
        # Sparsity analysis
        sparsity = np.mean(rp.projection_matrix_ == 0)
        
        print(f"  Projection time: {projection_time:.4f}s")
        print(f"  Projected shape: {X_projected.shape}")
        print(f"  Distance ratio: {mean_ratio:.3f} ± {std_ratio:.3f}")
        print(f"  Matrix sparsity: {sparsity:.3f}")
        
        results[method] = {
            'rp': rp,
            'X_projected': X_projected,
            'time': projection_time,
            'distance_ratio': mean_ratio,
            'sparsity': sparsity
        }
    
    # Compare with sklearn
    print(f"\nSklearn Random Projections:")
    
    # Gaussian
    start_time = time.time()
    grp = GaussianRandomProjection(n_components=target_dim, random_state=42)
    X_grp = grp.fit_transform(X)
    grp_time = time.time() - start_time
    
    # Sparse
    start_time = time.time()
    srp = SparseRandomProjection(n_components=target_dim, random_state=42)
    X_srp = srp.fit_transform(X)
    srp_time = time.time() - start_time
    
    print(f"  Gaussian RP: {grp_time:.4f}s, shape: {X_grp.shape}")
    print(f"  Sparse RP: {srp_time:.4f}s, shape: {X_srp.shape}")
    
    return results, X, original_distances, projected_distances

rp_results, X_rp, orig_dist, proj_dist = implement_random_projections()
```

**4. Matrix Factorization Techniques:**

```python
def implement_matrix_factorization():
    """Implement various matrix factorization techniques for dimensionality reduction"""
    
    print("\nMatrix Factorization Techniques:")
    print("-" * 30)
    
    # Generate data with latent structure
    n_samples = 2000
    n_features = 100
    latent_dim = 10
    noise_level = 0.1
    
    # Generate ground truth latent factors
    np.random.seed(42)
    W_true = np.random.randn(n_samples, latent_dim)
    H_true = np.random.randn(latent_dim, n_features)
    X = W_true @ H_true + noise_level * np.random.randn(n_samples, n_features)
    
    print(f"Data shape: {X.shape}")
    print(f"True latent dimension: {latent_dim}")
    
    class NonNegativeMatrixFactorization:
        """Custom NMF implementation"""
        
        def __init__(self, n_components, max_iter=200, tol=1e-4):
            self.n_components = n_components
            self.max_iter = max_iter
            self.tol = tol
            self.W_ = None
            self.H_ = None
            
        def fit_transform(self, X):
            """Fit NMF and return transformed data"""
            # Ensure non-negative data
            X = np.maximum(X, 0)
            
            n_samples, n_features = X.shape
            
            # Initialize factors
            self.W_ = np.random.rand(n_samples, self.n_components)
            self.H_ = np.random.rand(self.n_components, n_features)
            
            prev_error = float('inf')
            
            for iteration in range(self.max_iter):
                # Update H
                self.H_ *= (self.W_.T @ X) / (self.W_.T @ self.W_ @ self.H_ + 1e-10)
                
                # Update W
                self.W_ *= (X @ self.H_.T) / (self.W_ @ self.H_ @ self.H_.T + 1e-10)
                
                # Check convergence
                if iteration % 10 == 0:
                    error = np.linalg.norm(X - self.W_ @ self.H_, 'fro')
                    if abs(prev_error - error) < self.tol:
                        print(f"    NMF converged at iteration {iteration}")
                        break
                    prev_error = error
            
            return self.W_
    
    class FactorAnalysis:
        """Custom Factor Analysis implementation"""
        
        def __init__(self, n_components, max_iter=100):
            self.n_components = n_components
            self.max_iter = max_iter
            self.components_ = None
            self.noise_variance_ = None
            self.mean_ = None
            
        def fit(self, X):
            """Fit Factor Analysis model"""
            n_samples, n_features = X.shape
            
            # Center data
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            
            # Initialize parameters
            self.components_ = np.random.randn(self.n_components, n_features) * 0.1
            self.noise_variance_ = np.ones(n_features)
            
            for iteration in range(self.max_iter):
                # E-step: compute posterior
                precision = np.diag(1 / self.noise_variance_)
                cov_z = np.linalg.inv(
                    np.eye(self.n_components) + 
                    self.components_ @ precision @ self.components_.T
                )
                
                mean_z = cov_z @ self.components_ @ precision @ X_centered.T
                
                # M-step: update parameters
                # Update loadings
                sum_z_outer = n_samples * cov_z + mean_z @ mean_z.T
                self.components_ = (X_centered.T @ mean_z.T) @ np.linalg.inv(sum_z_outer)
                
                # Update noise variance
                reconstruction = self.components_.T @ mean_z
                residual = X_centered.T - reconstruction
                self.noise_variance_ = np.mean(residual**2, axis=1) + \
                                     np.diag(self.components_.T @ cov_z @ self.components_)
            
            return self
        
        def transform(self, X):
            """Transform data to latent space"""
            X_centered = X - self.mean_
            precision = np.diag(1 / self.noise_variance_)
            cov_z = np.linalg.inv(
                np.eye(self.n_components) + 
                self.components_ @ precision @ self.components_.T
            )
            return (cov_z @ self.components_ @ precision @ X_centered.T).T
    
    # Test different factorization methods
    methods = {}
    
    # 1. PCA (baseline)
    print(f"\n1. PCA:")
    start_time = time.time()
    pca = PCA(n_components=latent_dim)
    X_pca = pca.fit_transform(X)
    pca_time = time.time() - start_time
    
    reconstruction_pca = pca.inverse_transform(X_pca)
    error_pca = np.linalg.norm(X - reconstruction_pca, 'fro')
    
    print(f"  Time: {pca_time:.4f}s")
    print(f"  Reconstruction error: {error_pca:.4f}")
    print(f"  Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    methods['pca'] = {'time': pca_time, 'error': error_pca}
    
    # 2. SVD
    print(f"\n2. Truncated SVD:")
    start_time = time.time()
    svd = TruncatedSVD(n_components=latent_dim, random_state=42)
    X_svd = svd.fit_transform(X)
    svd_time = time.time() - start_time
    
    reconstruction_svd = svd.inverse_transform(X_svd)
    error_svd = np.linalg.norm(X - reconstruction_svd, 'fro')
    
    print(f"  Time: {svd_time:.4f}s")
    print(f"  Reconstruction error: {error_svd:.4f}")
    print(f"  Explained variance: {np.sum(svd.explained_variance_ratio_):.4f}")
    
    methods['svd'] = {'time': svd_time, 'error': error_svd}
    
    # 3. NMF (for non-negative data)
    print(f"\n3. Non-negative Matrix Factorization:")
    X_nonneg = np.maximum(X, 0)  # Ensure non-negative
    
    start_time = time.time()
    nmf = NonNegativeMatrixFactorization(n_components=latent_dim)
    X_nmf = nmf.fit_transform(X_nonneg)
    nmf_time = time.time() - start_time
    
    reconstruction_nmf = X_nmf @ nmf.H_
    error_nmf = np.linalg.norm(X_nonneg - reconstruction_nmf, 'fro')
    
    print(f"  Time: {nmf_time:.4f}s")
    print(f"  Reconstruction error: {error_nmf:.4f}")
    
    methods['nmf'] = {'time': nmf_time, 'error': error_nmf}
    
    # 4. Factor Analysis
    print(f"\n4. Factor Analysis:")
    start_time = time.time()
    fa = FactorAnalysis(n_components=latent_dim)
    fa.fit(X)
    X_fa = fa.transform(X)
    fa_time = time.time() - start_time
    
    reconstruction_fa = X_fa @ fa.components_ + fa.mean_
    error_fa = np.linalg.norm(X - reconstruction_fa, 'fro')
    
    print(f"  Time: {fa_time:.4f}s")
    print(f"  Reconstruction error: {error_fa:.4f}")
    
    methods['factor_analysis'] = {'time': fa_time, 'error': error_fa}
    
    # 5. Sklearn Factor Analysis (comparison)
    print(f"\n5. Sklearn Factor Analysis:")
    start_time = time.time()
    sklearn_fa = FactorAnalysis(n_components=latent_dim, random_state=42)
    X_sklearn_fa = sklearn_fa.fit_transform(X)
    sklearn_fa_time = time.time() - start_time
    
    reconstruction_sklearn_fa = sklearn_fa.inverse_transform(X_sklearn_fa)
    error_sklearn_fa = np.linalg.norm(X - reconstruction_sklearn_fa, 'fro')
    
    print(f"  Time: {sklearn_fa_time:.4f}s")
    print(f"  Reconstruction error: {error_sklearn_fa:.4f}")
    
    methods['sklearn_fa'] = {'time': sklearn_fa_time, 'error': error_sklearn_fa}
    
    return methods, X, X_pca, X_svd, X_nmf, X_fa

factorization_methods, X_fact, X_pca_fact, X_svd_fact, X_nmf_fact, X_fa_fact = implement_matrix_factorization()
```

**5. Comprehensive Dimensionality Reduction Pipeline:**

```python
def comprehensive_dimensionality_reduction_pipeline():
    """Complete pipeline for dimensionality reduction with multiple techniques"""
    
    print("\nComprehensive Dimensionality Reduction Pipeline:")
    print("=" * 50)
    
    # Generate complex high-dimensional dataset
    n_samples = 3000
    n_features = 200
    n_informative = 50
    n_redundant = 30
    n_clusters = 4
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_clusters,
        n_clusters_per_class=1,
        flip_y=0.01,
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_clusters} classes")
    
    class DimensionalityReductionPipeline:
        """Complete pipeline for dimensionality reduction"""
        
        def __init__(self):
            self.methods = {}
            self.results = {}
            self.recommendations = {}
            
        def add_method(self, name, method, params=None):
            """Add a dimensionality reduction method"""
            self.methods[name] = {
                'method': method,
                'params': params or {}
            }
        
        def evaluate_method(self, name, X, y=None, target_dim=2):
            """Evaluate a single method"""
            method_info = self.methods[name]
            method = method_info['method']
            params = method_info['params']
            
            # Create instance
            if params:
                instance = method(**params)
            else:
                instance = method(n_components=target_dim)
            
            # Fit and transform
            start_time = time.time()
            if hasattr(instance, 'fit_transform'):
                if name in ['lda'] and y is not None:
                    X_transformed = instance.fit_transform(X, y)
                else:
                    X_transformed = instance.fit_transform(X)
            else:
                if name in ['lda'] and y is not None:
                    instance.fit(X, y)
                else:
                    instance.fit(X)
                X_transformed = instance.transform(X)
            
            fit_time = time.time() - start_time
            
            # Compute metrics
            metrics = {
                'fit_time': fit_time,
                'output_shape': X_transformed.shape,
                'memory_reduction': X.nbytes / X_transformed.nbytes
            }
            
            # Reconstruction error (if possible)
            if hasattr(instance, 'inverse_transform'):
                X_reconstructed = instance.inverse_transform(X_transformed)
                metrics['reconstruction_error'] = np.linalg.norm(X - X_reconstructed, 'fro')
            
            # Explained variance (if available)
            if hasattr(instance, 'explained_variance_ratio_'):
                metrics['explained_variance'] = np.sum(instance.explained_variance_ratio_)
            
            # Class separability (if supervised)
            if y is not None:
                from sklearn.metrics import silhouette_score
                try:
                    metrics['silhouette_score'] = silhouette_score(X_transformed, y)
                except:
                    metrics['silhouette_score'] = None
            
            self.results[name] = {
                'instance': instance,
                'X_transformed': X_transformed,
                'metrics': metrics
            }
            
            return X_transformed, metrics
        
        def run_pipeline(self, X, y=None, target_dims=[2, 5, 10]):
            """Run complete pipeline"""
            
            # Define methods
            self.add_method('pca', PCA)
            self.add_method('truncated_svd', TruncatedSVD)
            self.add_method('factor_analysis', FactorAnalysis)
            self.add_method('gaussian_rp', GaussianRandomProjection)
            self.add_method('sparse_rp', SparseRandomProjection)
            
            if y is not None:
                self.add_method('lda', LinearDiscriminantAnalysis)
            
            # For each target dimension
            for target_dim in target_dims:
                print(f"\nTarget Dimensions: {target_dim}")
                print("-" * 20)
                
                dimension_results = {}
                
                for method_name in self.methods:
                    try:
                        print(f"  {method_name.upper()}:")
                        
                        X_transformed, metrics = self.evaluate_method(
                            method_name, X, y, target_dim
                        )
                        
                        dimension_results[method_name] = metrics
                        
                        print(f"    Time: {metrics['fit_time']:.4f}s")
                        print(f"    Shape: {metrics['output_shape']}")
                        print(f"    Memory reduction: {metrics['memory_reduction']:.1f}x")
                        
                        if 'reconstruction_error' in metrics:
                            print(f"    Reconstruction error: {metrics['reconstruction_error']:.4f}")
                        
                        if 'explained_variance' in metrics:
                            print(f"    Explained variance: {metrics['explained_variance']:.4f}")
                        
                        if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
                            print(f"    Silhouette score: {metrics['silhouette_score']:.4f}")
                        
                    except Exception as e:
                        print(f"    Error: {str(e)}")
                        dimension_results[method_name] = {'error': str(e)}
                
                # Recommendations
                self.generate_recommendations(dimension_results, target_dim)
        
        def generate_recommendations(self, results, target_dim):
            """Generate recommendations based on results"""
            
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if not valid_results:
                return
            
            # Best by different criteria
            best_speed = min(valid_results.items(), key=lambda x: x[1]['fit_time'])
            
            if any('explained_variance' in v for v in valid_results.values()):
                best_variance = max(
                    [(k, v) for k, v in valid_results.items() if 'explained_variance' in v],
                    key=lambda x: x[1]['explained_variance']
                )
                print(f"    Best explained variance: {best_variance[0]} ({best_variance[1]['explained_variance']:.4f})")
            
            if any('silhouette_score' in v and v['silhouette_score'] is not None for v in valid_results.values()):
                best_separation = max(
                    [(k, v) for k, v in valid_results.items() 
                     if 'silhouette_score' in v and v['silhouette_score'] is not None],
                    key=lambda x: x[1]['silhouette_score']
                )
                print(f"    Best class separation: {best_separation[0]} ({best_separation[1]['silhouette_score']:.4f})")
            
            print(f"    Fastest method: {best_speed[0]} ({best_speed[1]['fit_time']:.4f}s)")
            
            # General recommendations
            if target_dim <= 3:
                print(f"    Recommendation: PCA or LDA for visualization")
            elif target_dim <= 20:
                print(f"    Recommendation: PCA for interpretability, Random Projection for speed")
            else:
                print(f"    Recommendation: Random Projection for efficiency")
    
    # Run pipeline
    pipeline = DimensionalityReductionPipeline()
    pipeline.run_pipeline(X, y, target_dims=[2, 10, 50])
    
    # Final recommendations
    print(f"\nFinal Recommendations:")
    print("=" * 22)
    print(f"• Visualization (2-3D): PCA or LDA")
    print(f"• Feature extraction: PCA with 95% variance")
    print(f"• Speed critical: Random Projections")
    print(f"• Interpretability: PCA or Factor Analysis")
    print(f"• Supervised: LDA when applicable")
    print(f"• Non-negative data: NMF")
    print(f"• Very high dimensions: Random Projections")
    
    return pipeline, X, y

pipeline, X_pipeline, y_pipeline = comprehensive_dimensionality_reduction_pipeline()
```

**Summary and Best Practices:**

This comprehensive approach to dimensionality reduction using linear algebra provides:

1. **Multiple Techniques**: PCA, LDA, Random Projections, Matrix Factorization
2. **Scalability**: Handles various data sizes and computational constraints
3. **Flexibility**: Supervised and unsupervised approaches
4. **Performance Optimization**: Efficient implementations with complexity analysis
5. **Practical Guidelines**: Method selection based on data characteristics and requirements

**Key Selection Criteria:**
- **Data Size**: Random projections for very large datasets
- **Supervision**: LDA when labels available, PCA otherwise
- **Interpretability**: PCA for transparent feature combinations
- **Speed**: Random projections for real-time applications
- **Structure**: Matrix factorization for latent factors

---

## Question 4

**How would you use matrices to model relational data in databases?**

**Answer:** Matrices provide a powerful mathematical framework for modeling and analyzing relational data in databases, enabling efficient representation of relationships, operations, and analytics. This approach transforms relational concepts into linear algebraic operations, facilitating advanced analytics, graph analysis, and machine learning applications on database content.

**1. Fundamental Matrix Representations of Database Relations:**

**1.1 Entity-Relationship Matrix Modeling:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict

def demonstrate_entity_relationship_matrices():
    """Demonstrate how to model database relations using matrices"""
    
    print("Entity-Relationship Matrix Modeling")
    print("=" * 35)
    
    # Simulate database tables
    # Users table
    users_data = {
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'username': ['alice', 'bob', 'carol', 'david', 'eve', 'frank', 'grace', 'henry', 'iris', 'jack'],
        'age': [25, 30, 22, 35, 28, 45, 33, 29, 31, 27],
        'location': ['NY', 'CA', 'TX', 'NY', 'CA', 'TX', 'NY', 'CA', 'TX', 'NY']
    }
    
    # Products table
    products_data = {
        'product_id': [101, 102, 103, 104, 105, 106, 107, 108],
        'product_name': ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones', 'Camera', 'Speaker', 'Mouse'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Audio', 'Photo', 'Audio', 'Computer'],
        'price': [1200, 800, 500, 300, 150, 900, 200, 50]
    }
    
    # Purchases table (many-to-many relationship)
    purchases_data = {
        'user_id': [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10],
        'product_id': [101, 102, 102, 103, 101, 104, 105, 102, 106, 107, 108, 101, 103, 104, 105],
        'quantity': [1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 1],
        'purchase_date': ['2023-01-15', '2023-01-20', '2023-02-01', '2023-02-05', '2023-02-10', 
                         '2023-02-15', '2023-03-01', '2023-03-05', '2023-03-10', '2023-03-15',
                         '2023-04-01', '2023-04-05', '2023-04-10', '2023-04-15', '2023-04-20']
    }
    
    # Create DataFrames
    users_df = pd.DataFrame(users_data)
    products_df = pd.DataFrame(products_data)
    purchases_df = pd.DataFrame(purchases_data)
    
    print(f"Users table: {users_df.shape}")
    print(f"Products table: {products_df.shape}")
    print(f"Purchases table: {purchases_df.shape}")
    
    # 1. User-Product Interaction Matrix
    print(f"\n1. User-Product Interaction Matrix:")
    
    # Create user-product matrix
    user_ids = sorted(users_df['user_id'].unique())
    product_ids = sorted(products_df['product_id'].unique())
    
    user_product_matrix = np.zeros((len(user_ids), len(product_ids)))
    
    # Map IDs to indices
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    product_id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    
    # Fill matrix with purchase quantities
    for _, row in purchases_df.iterrows():
        user_idx = user_id_to_idx[row['user_id']]
        product_idx = product_id_to_idx[row['product_id']]
        user_product_matrix[user_idx, product_idx] += row['quantity']
    
    print(f"Matrix shape: {user_product_matrix.shape}")
    print(f"Matrix sparsity: {np.sum(user_product_matrix == 0) / user_product_matrix.size:.2f}")
    print(f"User-Product Matrix:\n{user_product_matrix}")
    
    # 2. Binary Relationship Matrix
    print(f"\n2. Binary Relationship Matrix:")
    binary_matrix = (user_product_matrix > 0).astype(int)
    print(f"Binary matrix (purchased=1, not purchased=0):\n{binary_matrix}")
    
    # 3. Weighted Relationship Matrix
    print(f"\n3. Weighted Relationship Matrix:")
    
    # Weight by product price
    product_prices = products_df.set_index('product_id')['price'].to_dict()
    weighted_matrix = np.zeros_like(user_product_matrix)
    
    for i, user_id in enumerate(user_ids):
        for j, product_id in enumerate(product_ids):
            if user_product_matrix[i, j] > 0:
                weighted_matrix[i, j] = user_product_matrix[i, j] * product_prices[product_id]
    
    print(f"Weighted matrix (quantity × price):\n{weighted_matrix}")
    
    return user_product_matrix, binary_matrix, weighted_matrix, users_df, products_df, purchases_df

user_product_matrix, binary_matrix, weighted_matrix, users_df, products_df, purchases_df = demonstrate_entity_relationship_matrices()
```

**1.2 Advanced Relationship Modeling:**
```python
def advanced_relationship_modeling():
    """Advanced techniques for modeling complex database relationships"""
    
    print("\nAdvanced Relationship Modeling:")
    print("-" * 30)
    
    # Multi-dimensional relationship tensor
    # Users × Products × Time
    
    # Extract temporal information
    purchases_df['purchase_date'] = pd.to_datetime(purchases_df['purchase_date'])
    purchases_df['month'] = purchases_df['purchase_date'].dt.month
    
    months = sorted(purchases_df['month'].unique())
    user_ids = sorted(users_df['user_id'].unique())
    product_ids = sorted(products_df['product_id'].unique())
    
    print(f"Tensor dimensions: {len(user_ids)} users × {len(product_ids)} products × {len(months)} months")
    
    # Create 3D tensor
    relationship_tensor = np.zeros((len(user_ids), len(product_ids), len(months)))
    
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    product_id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    month_to_idx = {m: idx for idx, m in enumerate(months)}
    
    for _, row in purchases_df.iterrows():
        user_idx = user_id_to_idx[row['user_id']]
        product_idx = product_id_to_idx[row['product_id']]
        month_idx = month_to_idx[row['month']]
        relationship_tensor[user_idx, product_idx, month_idx] += row['quantity']
    
    print(f"Tensor shape: {relationship_tensor.shape}")
    print(f"Non-zero entries: {np.count_nonzero(relationship_tensor)}")
    
    # Tensor decomposition for pattern discovery
    print(f"\nTensor Analysis:")
    
    # Flatten tensor for analysis
    # Mode-1 unfolding (users)
    mode_1_matrix = relationship_tensor.reshape(len(user_ids), -1)
    print(f"Mode-1 (users) matrix shape: {mode_1_matrix.shape}")
    
    # Mode-2 unfolding (products)
    mode_2_matrix = relationship_tensor.transpose(1, 0, 2).reshape(len(product_ids), -1)
    print(f"Mode-2 (products) matrix shape: {mode_2_matrix.shape}")
    
    # Mode-3 unfolding (time)
    mode_3_matrix = relationship_tensor.transpose(2, 0, 1).reshape(len(months), -1)
    print(f"Mode-3 (time) matrix shape: {mode_3_matrix.shape}")
    
    return relationship_tensor, mode_1_matrix, mode_2_matrix, mode_3_matrix

relationship_tensor, mode_1_matrix, mode_2_matrix, mode_3_matrix = advanced_relationship_modeling()
```

**2. Graph-Based Database Modeling:**

```python
def graph_database_modeling():
    """Model database relationships as graphs using adjacency matrices"""
    
    print("\nGraph Database Modeling:")
    print("-" * 23)
    
    # Create different types of graphs from database
    
    # 1. User Similarity Graph
    print(f"1. User Similarity Graph:")
    
    # Compute user similarity based on purchase behavior
    user_similarity = cosine_similarity(user_product_matrix)
    
    print(f"User similarity matrix shape: {user_similarity.shape}")
    print(f"User similarity matrix:\n{user_similarity}")
    
    # Create adjacency matrix (threshold similarity)
    similarity_threshold = 0.3
    user_graph_adj = (user_similarity > similarity_threshold).astype(int)
    np.fill_diagonal(user_graph_adj, 0)  # Remove self-connections
    
    print(f"User graph edges: {np.sum(user_graph_adj) // 2}")
    
    # 2. Product Co-purchase Graph
    print(f"\n2. Product Co-purchase Graph:")
    
    # Products bought together
    product_similarity = cosine_similarity(user_product_matrix.T)
    
    print(f"Product similarity matrix shape: {product_similarity.shape}")
    
    copurchase_threshold = 0.2
    product_graph_adj = (product_similarity > copurchase_threshold).astype(int)
    np.fill_diagonal(product_graph_adj, 0)
    
    print(f"Product co-purchase edges: {np.sum(product_graph_adj) // 2}")
    
    # 3. Bipartite User-Product Graph
    print(f"\n3. Bipartite User-Product Graph:")
    
    n_users = len(user_ids)
    n_products = len(product_ids)
    
    # Create bipartite adjacency matrix
    bipartite_adj = np.zeros((n_users + n_products, n_users + n_products))
    
    # Fill user-product connections
    bipartite_adj[:n_users, n_users:] = binary_matrix
    bipartite_adj[n_users:, :n_users] = binary_matrix.T
    
    print(f"Bipartite adjacency matrix shape: {bipartite_adj.shape}")
    print(f"Total edges: {np.sum(bipartite_adj) // 2}")
    
    # Graph metrics using linear algebra
    print(f"\nGraph Metrics:")
    
    # Degree centrality
    user_degrees = np.sum(user_graph_adj, axis=1)
    product_degrees = np.sum(product_graph_adj, axis=1)
    
    print(f"User degrees: {user_degrees}")
    print(f"Product degrees: {product_degrees}")
    
    # Clustering coefficient (using matrix operations)
    def clustering_coefficient_matrix(adj_matrix):
        """Compute clustering coefficient using matrix operations"""
        n = adj_matrix.shape[0]
        adj_cubed = np.linalg.matrix_power(adj_matrix, 3)
        triangles = np.diag(adj_cubed) / 2
        
        degrees = np.sum(adj_matrix, axis=1)
        possible_triangles = degrees * (degrees - 1) / 2
        
        clustering = np.divide(triangles, possible_triangles, 
                             out=np.zeros_like(triangles), 
                             where=possible_triangles!=0)
        return clustering
    
    user_clustering = clustering_coefficient_matrix(user_graph_adj)
    print(f"User clustering coefficients: {user_clustering}")
    
    return user_similarity, product_similarity, bipartite_adj, user_graph_adj, product_graph_adj

user_sim, product_sim, bipartite_adj, user_graph_adj, product_graph_adj = graph_database_modeling()
```

**3. Matrix-Based Database Operations:**

```python
def matrix_database_operations():
    """Implement database operations using matrix algebra"""
    
    print("\nMatrix-Based Database Operations:")
    print("-" * 32)
    
    # 1. JOIN Operations using Matrix Multiplication
    print(f"1. JOIN Operations:")
    
    # Create relationship matrices for JOIN
    # User-Location matrix
    locations = users_df['location'].unique()
    user_location_matrix = np.zeros((len(user_ids), len(locations)))
    
    location_to_idx = {loc: idx for idx, loc in enumerate(locations)}
    
    for i, user_id in enumerate(user_ids):
        user_location = users_df[users_df['user_id'] == user_id]['location'].iloc[0]
        loc_idx = location_to_idx[user_location]
        user_location_matrix[i, loc_idx] = 1
    
    print(f"User-Location matrix shape: {user_location_matrix.shape}")
    print(f"User-Location matrix:\n{user_location_matrix}")
    
    # Product-Category matrix
    categories = products_df['category'].unique()
    product_category_matrix = np.zeros((len(product_ids), len(categories)))
    
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    for i, product_id in enumerate(product_ids):
        product_category = products_df[products_df['product_id'] == product_id]['category'].iloc[0]
        cat_idx = category_to_idx[product_category]
        product_category_matrix[i, cat_idx] = 1
    
    print(f"Product-Category matrix shape: {product_category_matrix.shape}")
    
    # JOIN: Users → Products → Categories (via matrix multiplication)
    user_category_purchases = user_product_matrix @ product_category_matrix
    
    print(f"User-Category purchases (via JOIN):\n{user_category_purchases}")
    print(f"Categories: {list(categories)}")
    
    # 2. Aggregation Operations
    print(f"\n2. Aggregation Operations:")
    
    # SUM by user (total purchases per user)
    user_totals = np.sum(user_product_matrix, axis=1)
    print(f"Total purchases per user: {user_totals}")
    
    # SUM by product (total sales per product)
    product_totals = np.sum(user_product_matrix, axis=0)
    print(f"Total sales per product: {product_totals}")
    
    # GROUP BY using matrix operations
    # Group users by location and sum purchases
    location_purchases = user_location_matrix.T @ user_product_matrix
    print(f"Purchases by location:\n{location_purchases}")
    print(f"Locations: {list(locations)}")
    
    # 3. Filtering Operations
    print(f"\n3. Filtering Operations:")
    
    # Filter: Users who bought more than 2 items
    active_users_mask = user_totals > 2
    active_user_matrix = user_product_matrix[active_users_mask, :]
    
    print(f"Active users (>2 purchases): {np.sum(active_users_mask)}")
    print(f"Active user matrix shape: {active_user_matrix.shape}")
    
    # Filter: High-value products (price > 500)
    high_value_products = products_df[products_df['price'] > 500]['product_id'].values
    high_value_indices = [product_id_to_idx[pid] for pid in high_value_products]
    high_value_matrix = user_product_matrix[:, high_value_indices]
    
    print(f"High-value products matrix shape: {high_value_matrix.shape}")
    
    # 4. Ranking and Sorting
    print(f"\n4. Ranking and Sorting:")
    
    # Rank users by total spending
    user_spending = weighted_matrix.sum(axis=1)
    user_spending_rank = np.argsort(user_spending)[::-1]
    
    print(f"User spending ranking (highest to lowest):")
    for i, user_idx in enumerate(user_spending_rank[:5]):
        user_id = user_ids[user_idx]
        spending = user_spending[user_idx]
        print(f"  {i+1}. User {user_id}: ${spending:.2f}")
    
    # Rank products by popularity
    product_popularity = binary_matrix.sum(axis=0)
    product_popularity_rank = np.argsort(product_popularity)[::-1]
    
    print(f"\nProduct popularity ranking:")
    for i, prod_idx in enumerate(product_popularity_rank[:5]):
        product_id = product_ids[prod_idx]
        popularity = product_popularity[prod_idx]
        product_name = products_df[products_df['product_id'] == product_id]['product_name'].iloc[0]
        print(f"  {i+1}. {product_name}: {popularity} users")
    
    return user_location_matrix, product_category_matrix, user_category_purchases

user_location_matrix, product_category_matrix, user_category_purchases = matrix_database_operations()
```

**4. Recommendation Systems using Matrix Methods:**

```python
def matrix_based_recommendation_system():
    """Implement recommendation systems using matrix factorization"""
    
    print("\nMatrix-Based Recommendation System:")
    print("-" * 34)
    
    # 1. Collaborative Filtering using SVD
    print(f"1. Collaborative Filtering:")
    
    # Use sparse matrix for efficiency
    sparse_user_product = csr_matrix(user_product_matrix)
    
    # Apply SVD for matrix factorization
    n_components = min(5, min(user_product_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    # Fit SVD on user-product matrix
    user_factors = svd.fit_transform(sparse_user_product)
    product_factors = svd.components_.T
    
    print(f"User factors shape: {user_factors.shape}")
    print(f"Product factors shape: {product_factors.shape}")
    print(f"Explained variance ratio: {svd.explained_variance_ratio_}")
    
    # Reconstruct matrix for recommendations
    reconstructed_matrix = user_factors @ svd.components_
    
    print(f"Reconstruction error: {np.linalg.norm(user_product_matrix - reconstructed_matrix, 'fro'):.4f}")
    
    # Generate recommendations
    def get_recommendations(user_idx, n_recommendations=3):
        """Get product recommendations for a user"""
        user_scores = reconstructed_matrix[user_idx]
        
        # Exclude already purchased products
        purchased_mask = user_product_matrix[user_idx] > 0
        user_scores[purchased_mask] = -np.inf
        
        # Get top recommendations
        top_indices = np.argsort(user_scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            product_id = product_ids[idx]
            product_name = products_df[products_df['product_id'] == product_id]['product_name'].iloc[0]
            score = user_scores[idx]
            recommendations.append((product_id, product_name, score))
        
        return recommendations
    
    # Test recommendations
    print(f"\nRecommendations for User 1:")
    recommendations = get_recommendations(0)
    for pid, name, score in recommendations:
        print(f"  {name} (ID: {pid}): {score:.3f}")
    
    # 2. Matrix Factorization with Non-negative constraints
    print(f"\n2. Non-negative Matrix Factorization:")
    
    from sklearn.decomposition import NMF
    
    # Ensure non-negative matrix
    nn_matrix = np.maximum(user_product_matrix, 0)
    
    nmf = NMF(n_components=n_components, random_state=42, max_iter=200)
    user_factors_nmf = nmf.fit_transform(nn_matrix)
    product_factors_nmf = nmf.components_.T
    
    print(f"NMF User factors shape: {user_factors_nmf.shape}")
    print(f"NMF Product factors shape: {product_factors_nmf.shape}")
    print(f"NMF Reconstruction error: {nmf.reconstruction_err_:.4f}")
    
    # 3. Similarity-based Recommendations
    print(f"\n3. Similarity-based Recommendations:")
    
    def similarity_recommendations(user_idx, n_recommendations=3):
        """Recommend based on user similarity"""
        
        # Find similar users
        user_similarities = user_sim[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:6]  # Top 5 similar users (excluding self)
        
        # Aggregate recommendations from similar users
        recommendation_scores = np.zeros(len(product_ids))
        
        for similar_user in similar_users:
            similarity_weight = user_similarities[similar_user]
            user_purchases = user_product_matrix[similar_user]
            
            # Weight by similarity
            recommendation_scores += similarity_weight * user_purchases
        
        # Exclude already purchased
        purchased_mask = user_product_matrix[user_idx] > 0
        recommendation_scores[purchased_mask] = -np.inf
        
        # Get top recommendations
        top_indices = np.argsort(recommendation_scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            product_id = product_ids[idx]
            product_name = products_df[products_df['product_id'] == product_id]['product_name'].iloc[0]
            score = recommendation_scores[idx]
            recommendations.append((product_id, product_name, score))
        
        return recommendations
    
    print(f"Similarity-based recommendations for User 1:")
    sim_recommendations = similarity_recommendations(0)
    for pid, name, score in sim_recommendations:
        print(f"  {name} (ID: {pid}): {score:.3f}")
    
    return user_factors, product_factors, reconstructed_matrix, user_factors_nmf, product_factors_nmf

user_factors, product_factors, reconstructed_matrix, user_factors_nmf, product_factors_nmf = matrix_based_recommendation_system()
```

**5. Database Analytics using Linear Algebra:**

```python
def database_analytics_linear_algebra():
    """Advanced database analytics using linear algebra techniques"""
    
    print("\nDatabase Analytics using Linear Algebra:")
    print("-" * 40)
    
    # 1. Principal Component Analysis for Customer Segmentation
    print(f"1. Customer Segmentation (PCA):")
    
    # Create feature matrix for users
    user_features = []
    
    for user_id in user_ids:
        user_row = users_df[users_df['user_id'] == user_id].iloc[0]
        user_idx = user_id_to_idx[user_id]
        
        features = [
            user_row['age'],  # Age
            np.sum(user_product_matrix[user_idx]),  # Total purchases
            np.sum(weighted_matrix[user_idx]),  # Total spending
            np.count_nonzero(user_product_matrix[user_idx]),  # Unique products
        ]
        
        # Add category preferences
        user_category_prefs = user_category_purchases[user_idx]
        features.extend(user_category_prefs)
        
        user_features.append(features)
    
    user_feature_matrix = np.array(user_features)
    
    print(f"User feature matrix shape: {user_feature_matrix.shape}")
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_feature_matrix)
    
    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    user_components = pca.fit_transform(user_features_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Clustering in reduced space
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_clusters = kmeans.fit_predict(user_components)
    
    print(f"User clusters: {user_clusters}")
    
    # 2. Market Basket Analysis using Association Rules
    print(f"\n2. Market Basket Analysis:")
    
    # Convert to binary transaction matrix
    transactions = binary_matrix
    
    # Compute support matrix (frequency of itemsets)
    support_matrix = transactions.T @ transactions / len(user_ids)
    
    print(f"Support matrix shape: {support_matrix.shape}")
    print(f"Product support (individual):")
    for i, product_id in enumerate(product_ids[:5]):
        product_name = products_df[products_df['product_id'] == product_id]['product_name'].iloc[0]
        support = support_matrix[i, i]
        print(f"  {product_name}: {support:.3f}")
    
    # Find frequent itemsets (pairs)
    min_support = 0.2
    frequent_pairs = []
    
    for i in range(len(product_ids)):
        for j in range(i+1, len(product_ids)):
            support = support_matrix[i, j]
            if support >= min_support:
                product_i = products_df[products_df['product_id'] == product_ids[i]]['product_name'].iloc[0]
                product_j = products_df[products_df['product_id'] == product_ids[j]]['product_name'].iloc[0]
                frequent_pairs.append((product_i, product_j, support))
    
    print(f"\nFrequent product pairs (support >= {min_support}):")
    for prod_i, prod_j, support in frequent_pairs:
        print(f"  {prod_i} + {prod_j}: {support:.3f}")
    
    # 3. Time Series Analysis using Matrix Methods
    print(f"\n3. Time Series Analysis:")
    
    # Analyze temporal patterns in the relationship tensor
    monthly_totals = np.sum(relationship_tensor, axis=(0, 1))
    monthly_user_activity = np.sum(relationship_tensor, axis=(1, 2))
    monthly_product_sales = np.sum(relationship_tensor, axis=(0, 2))
    
    print(f"Monthly total sales: {monthly_totals}")
    print(f"Monthly active users: {np.count_nonzero(monthly_user_activity)}")
    
    # Trend analysis using linear regression (matrix form)
    time_points = np.arange(len(months)).reshape(-1, 1)
    design_matrix = np.column_stack([np.ones(len(months)), time_points.flatten()])
    
    # Solve for trend: y = a + b*t
    trend_coeffs = np.linalg.lstsq(design_matrix, monthly_totals, rcond=None)[0]
    
    print(f"Sales trend: intercept={trend_coeffs[0]:.2f}, slope={trend_coeffs[1]:.2f}")
    
    # 4. Anomaly Detection using Matrix Decomposition
    print(f"\n4. Anomaly Detection:")
    
    # Use reconstruction error from SVD for anomaly detection
    reconstruction_errors = []
    
    for i in range(len(user_ids)):
        original_profile = user_product_matrix[i]
        reconstructed_profile = reconstructed_matrix[i]
        
        error = np.linalg.norm(original_profile - reconstructed_profile)
        reconstruction_errors.append(error)
    
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Identify anomalies (high reconstruction error)
    threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
    anomalous_users = np.where(reconstruction_errors > threshold)[0]
    
    print(f"Anomaly threshold: {threshold:.3f}")
    print(f"Anomalous users: {[user_ids[i] for i in anomalous_users]}")
    
    return user_feature_matrix, user_components, user_clusters, support_matrix, reconstruction_errors

user_feature_matrix, user_components, user_clusters, support_matrix, reconstruction_errors = database_analytics_linear_algebra()
```

**6. Scalable Matrix Operations for Large Databases:**

```python
def scalable_matrix_operations():
    """Demonstrate scalable matrix operations for large database systems"""
    
    print("\nScalable Matrix Operations:")
    print("-" * 26)
    
    # 1. Sparse Matrix Operations
    print(f"1. Sparse Matrix Efficiency:")
    
    # Convert to sparse format
    sparse_user_product = csr_matrix(user_product_matrix)
    sparse_binary = csr_matrix(binary_matrix)
    
    print(f"Dense matrix memory: {user_product_matrix.nbytes} bytes")
    print(f"Sparse matrix memory: {sparse_user_product.data.nbytes + sparse_user_product.indices.nbytes + sparse_user_product.indptr.nbytes} bytes")
    
    # Sparse matrix operations
    sparse_similarity = sparse_user_product @ sparse_user_product.T
    print(f"Sparse similarity matrix shape: {sparse_similarity.shape}")
    print(f"Sparse similarity non-zeros: {sparse_similarity.nnz}")
    
    # 2. Incremental Matrix Updates
    print(f"\n2. Incremental Updates:")
    
    def incremental_similarity_update(similarity_matrix, new_user_vector, user_vectors):
        """Update similarity matrix when new user is added"""
        
        # Compute similarities with existing users
        new_similarities = new_user_vector @ user_vectors.T
        
        # Expand similarity matrix
        n_users = similarity_matrix.shape[0]
        expanded_similarity = np.zeros((n_users + 1, n_users + 1))
        
        # Copy existing similarities
        expanded_similarity[:n_users, :n_users] = similarity_matrix
        
        # Add new similarities
        expanded_similarity[n_users, :n_users] = new_similarities
        expanded_similarity[:n_users, n_users] = new_similarities
        expanded_similarity[n_users, n_users] = np.dot(new_user_vector, new_user_vector)
        
        return expanded_similarity
    
    # Simulate new user
    new_user_purchases = np.array([0, 1, 0, 1, 1, 0, 0, 0])  # New user's purchases
    updated_similarity = incremental_similarity_update(
        user_sim, new_user_purchases, user_product_matrix
    )
    
    print(f"Original similarity matrix: {user_sim.shape}")
    print(f"Updated similarity matrix: {updated_similarity.shape}")
    
    # 3. Distributed Matrix Operations
    print(f"\n3. Distributed Operations Simulation:")
    
    def distributed_matrix_multiply_simulation(A, B, n_partitions=2):
        """Simulate distributed matrix multiplication"""
        
        # Partition matrices
        partition_size = A.shape[0] // n_partitions
        
        results = []
        
        for i in range(n_partitions):
            start_row = i * partition_size
            end_row = min((i + 1) * partition_size, A.shape[0])
            
            # Simulate computation on partition
            A_partition = A[start_row:end_row]
            C_partition = A_partition @ B
            
            results.append(C_partition)
            
            print(f"  Partition {i}: rows {start_row}-{end_row-1}, result shape {C_partition.shape}")
        
        # Combine results
        C = np.vstack(results)
        return C
    
    # Test distributed multiplication
    distributed_result = distributed_matrix_multiply_simulation(
        user_product_matrix, product_category_matrix
    )
    
    print(f"Distributed result shape: {distributed_result.shape}")
    print(f"Results match: {np.allclose(distributed_result, user_category_purchases)}")
    
    # 4. Memory-Efficient Operations
    print(f"\n4. Memory-Efficient Operations:")
    
    def chunked_similarity_computation(matrix, chunk_size=3):
        """Compute similarity matrix in chunks to save memory"""
        
        n_samples = matrix.shape[0]
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(0, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)
            
            for j in range(0, n_samples, chunk_size):
                end_j = min(j + chunk_size, n_samples)
                
                # Compute chunk similarity
                chunk_i = matrix[i:end_i]
                chunk_j = matrix[j:end_j]
                
                similarity_chunk = cosine_similarity(chunk_i, chunk_j)
                similarity_matrix[i:end_i, j:end_j] = similarity_chunk
                
                print(f"  Processed chunk ({i}-{end_i-1}, {j}-{end_j-1})")
        
        return similarity_matrix
    
    chunked_similarity = chunked_similarity_computation(user_product_matrix)
    
    print(f"Chunked similarity matches original: {np.allclose(chunked_similarity, user_sim, atol=1e-10)}")
    
    return sparse_user_product, updated_similarity, distributed_result, chunked_similarity

sparse_user_product, updated_similarity, distributed_result, chunked_similarity = scalable_matrix_operations()
```

**7. Complete Database Matrix Framework:**

```python
def complete_database_matrix_framework():
    """Comprehensive framework for matrix-based database modeling"""
    
    print("\nComplete Database Matrix Framework:")
    print("=" * 38)
    
    class MatrixDatabaseFramework:
        """Comprehensive framework for matrix-based database operations"""
        
        def __init__(self):
            self.entities = {}
            self.relationships = {}
            self.matrices = {}
            self.analytics = {}
            
        def add_entity(self, name, data):
            """Add entity data"""
            self.entities[name] = data
            print(f"Added entity '{name}' with {len(data)} records")
            
        def create_relationship_matrix(self, entity1, entity2, relationship_data, 
                                     value_column=None, binary=False):
            """Create relationship matrix between entities"""
            
            entity1_ids = sorted(self.entities[entity1].index)
            entity2_ids = sorted(self.entities[entity2].index)
            
            matrix = np.zeros((len(entity1_ids), len(entity2_ids)))
            
            entity1_to_idx = {eid: idx for idx, eid in enumerate(entity1_ids)}
            entity2_to_idx = {eid: idx for idx, eid in enumerate(entity2_ids)}
            
            for _, row in relationship_data.iterrows():
                e1_idx = entity1_to_idx.get(row[f'{entity1}_id'])
                e2_idx = entity2_to_idx.get(row[f'{entity2}_id'])
                
                if e1_idx is not None and e2_idx is not None:
                    if binary:
                        matrix[e1_idx, e2_idx] = 1
                    elif value_column:
                        matrix[e1_idx, e2_idx] += row[value_column]
                    else:
                        matrix[e1_idx, e2_idx] += 1
            
            relationship_name = f"{entity1}_{entity2}"
            self.matrices[relationship_name] = matrix
            self.relationships[relationship_name] = {
                'entity1': entity1,
                'entity2': entity2,
                'entity1_ids': entity1_ids,
                'entity2_ids': entity2_ids
            }
            
            print(f"Created {relationship_name} matrix: {matrix.shape}")
            return matrix
        
        def compute_similarity(self, matrix_name, method='cosine'):
            """Compute similarity matrix"""
            matrix = self.matrices[matrix_name]
            
            if method == 'cosine':
                similarity = cosine_similarity(matrix)
            elif method == 'correlation':
                similarity = np.corrcoef(matrix)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            similarity_name = f"{matrix_name}_similarity"
            self.matrices[similarity_name] = similarity
            
            return similarity
        
        def matrix_factorization(self, matrix_name, n_components, method='svd'):
            """Perform matrix factorization"""
            matrix = self.matrices[matrix_name]
            
            if method == 'svd':
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                factors1 = svd.fit_transform(matrix)
                factors2 = svd.components_.T
                
            elif method == 'nmf':
                nmf = NMF(n_components=n_components, random_state=42)
                factors1 = nmf.fit_transform(np.maximum(matrix, 0))
                factors2 = nmf.components_.T
            
            factorization_name = f"{matrix_name}_{method}"
            self.analytics[factorization_name] = {
                'factors1': factors1,
                'factors2': factors2,
                'method': method
            }
            
            return factors1, factors2
        
        def recommendation_system(self, matrix_name, entity_id, n_recommendations=5):
            """Generate recommendations"""
            if f"{matrix_name}_svd" not in self.analytics:
                self.matrix_factorization(matrix_name, n_components=5)
            
            factors = self.analytics[f"{matrix_name}_svd"]
            reconstructed = factors['factors1'] @ factors['factors2'].T
            
            relationship = self.relationships[matrix_name]
            entity1_ids = relationship['entity1_ids']
            
            if entity_id not in entity1_ids:
                return []
            
            entity_idx = entity1_ids.index(entity_id)
            scores = reconstructed[entity_idx]
            
            # Exclude already connected entities
            original_matrix = self.matrices[matrix_name]
            connected_mask = original_matrix[entity_idx] > 0
            scores[connected_mask] = -np.inf
            
            # Get top recommendations
            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            entity2_ids = relationship['entity2_ids']
            
            recommendations = [(entity2_ids[idx], scores[idx]) for idx in top_indices]
            return recommendations
        
        def analytics_summary(self):
            """Print analytics summary"""
            print(f"\nFramework Summary:")
            print(f"Entities: {list(self.entities.keys())}")
            print(f"Matrices: {list(self.matrices.keys())}")
            print(f"Analytics: {list(self.analytics.keys())}")
            
            for name, matrix in self.matrices.items():
                sparsity = np.sum(matrix == 0) / matrix.size
                print(f"  {name}: {matrix.shape}, sparsity: {sparsity:.3f}")
    
    # Demonstrate framework
    framework = MatrixDatabaseFramework()
    
    # Add entities
    users_indexed = users_df.set_index('user_id')
    products_indexed = products_df.set_index('product_id')
    
    framework.add_entity('user', users_indexed)
    framework.add_entity('product', products_indexed)
    
    # Create relationships
    purchase_matrix = framework.create_relationship_matrix(
        'user', 'product', purchases_df, value_column='quantity'
    )
    
    # Compute similarities
    user_similarity = framework.compute_similarity('user_product', method='cosine')
    
    # Matrix factorization
    user_factors, product_factors = framework.matrix_factorization('user_product', n_components=3)
    
    # Generate recommendations
    recommendations = framework.recommendation_system('user_product', 1, n_recommendations=3)
    
    print(f"\nRecommendations for user 1:")
    for product_id, score in recommendations:
        product_name = products_df[products_df['product_id'] == product_id]['product_name'].iloc[0]
        print(f"  {product_name}: {score:.3f}")
    
    # Framework summary
    framework.analytics_summary()
    
    return framework

framework = complete_database_matrix_framework()
```

**Key Benefits of Matrix-Based Database Modeling:**

1. **Efficient Operations**: Linear algebra operations for joins, aggregations, and analytics
2. **Scalability**: Sparse matrices and distributed computations for large datasets
3. **Advanced Analytics**: PCA, clustering, recommendation systems, anomaly detection
4. **Graph Analysis**: Network analysis using adjacency matrices
5. **Real-time Processing**: Incremental updates and streaming algorithms
6. **Machine Learning Integration**: Direct interface with ML algorithms

**Applications:**
- **E-commerce**: Product recommendations, customer segmentation
- **Social Networks**: Friend recommendations, community detection
- **Financial Systems**: Fraud detection, risk analysis
- **Content Platforms**: Content recommendation, user behavior analysis
- **Supply Chain**: Relationship modeling, optimization

This matrix-based approach transforms traditional database operations into efficient linear algebraic computations, enabling sophisticated analytics and machine learning directly on relational data.

---

## Question 5

**Discuss how to applylinear algebratoimage processingtasks.**

**Answer:** _[To be filled]_

---

## Question 6

**Discuss the role oflinear algebraindeep learning, specifically in trainingconvolutional neural networks.**

**Answer:** _[To be filled]_

---

## Question 7

**Propose strategies to visualizehigh-dimensional datausinglinear algebratechniques.**

**Answer:** _[To be filled]_

---

## Question 8

**Discuss an approach for optimizingmemory usageinmatrix computationsfor a large-scalemachine learning application.**

**Answer:** _[To be filled]_

---

