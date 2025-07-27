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

**Discuss how to apply linear algebra to image processing tasks.**

**Answer:** Linear algebra forms the mathematical foundation of modern image processing, providing efficient tools for image representation, transformation, enhancement, and analysis. Images are naturally represented as matrices, making linear algebraic operations directly applicable for various image processing tasks including filtering, compression, feature extraction, and computer vision applications.

**1. Image Representation and Basic Operations:**

**1.1 Image as Matrix Representation:**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from scipy.linalg import svd
from sklearn.decomposition import PCA, TruncatedSVD
from skimage import data, color, filters, feature, transform
from skimage.restoration import denoise_tv_chambolle
import cv2
from PIL import Image
import time

def image_matrix_fundamentals():
    """Demonstrate fundamental image-matrix operations"""
    
    print("Image-Matrix Fundamentals")
    print("=" * 25)
    
    # Load sample images
    # Grayscale image
    gray_image = data.camera()  # 512x512 grayscale
    
    # Color image  
    color_image = data.astronaut()  # RGB image
    
    print(f"Grayscale image shape: {gray_image.shape}")
    print(f"Color image shape: {color_image.shape}")
    print(f"Grayscale data type: {gray_image.dtype}")
    print(f"Grayscale value range: [{gray_image.min()}, {gray_image.max()}]")
    
    # Basic matrix operations on images
    print(f"\n1. Basic Matrix Operations:")
    
    # Image arithmetic
    brightened = gray_image + 50
    darkened = gray_image - 50
    contrast_enhanced = gray_image * 1.5
    gamma_corrected = np.power(gray_image / 255.0, 0.5) * 255
    
    # Clamp values to valid range
    brightened = np.clip(brightened, 0, 255)
    darkened = np.clip(darkened, 0, 255)
    contrast_enhanced = np.clip(contrast_enhanced, 0, 255)
    gamma_corrected = np.clip(gamma_corrected, 0, 255)
    
    print(f"  Brightened range: [{brightened.min()}, {brightened.max()}]")
    print(f"  Darkened range: [{darkened.min()}, {darkened.max()}]")
    print(f"  Contrast enhanced range: [{contrast_enhanced.min()}, {contrast_enhanced.max()}]")
    
    # Image statistics using linear algebra
    print(f"\n2. Image Statistics:")
    
    # Mean, variance using matrix operations
    mean_intensity = np.mean(gray_image)
    variance = np.var(gray_image)
    std_dev = np.std(gray_image)
    
    # Histogram as matrix operation
    histogram = np.bincount(gray_image.flatten(), minlength=256)
    
    print(f"  Mean intensity: {mean_intensity:.2f}")
    print(f"  Variance: {variance:.2f}")
    print(f"  Standard deviation: {std_dev:.2f}")
    print(f"  Histogram shape: {histogram.shape}")
    
    # Matrix norms for image analysis
    l1_norm = np.linalg.norm(gray_image, ord=1)
    l2_norm = np.linalg.norm(gray_image, ord='fro')  # Frobenius norm
    max_norm = np.linalg.norm(gray_image, ord=np.inf)
    
    print(f"  L1 norm: {l1_norm:.0f}")
    print(f"  L2 (Frobenius) norm: {l2_norm:.0f}")
    print(f"  Max norm: {max_norm:.0f}")
    
    # 3. Channel operations for color images
    print(f"\n3. Color Channel Operations:")
    
    # Split RGB channels
    red_channel = color_image[:, :, 0]
    green_channel = color_image[:, :, 1]
    blue_channel = color_image[:, :, 2]
    
    print(f"  Red channel shape: {red_channel.shape}")
    print(f"  Channel statistics:")
    print(f"    Red - mean: {np.mean(red_channel):.1f}, std: {np.std(red_channel):.1f}")
    print(f"    Green - mean: {np.mean(green_channel):.1f}, std: {np.std(green_channel):.1f}")
    print(f"    Blue - mean: {np.mean(blue_channel):.1f}, std: {np.std(blue_channel):.1f}")
    
    # Color space conversion using matrix multiplication
    # RGB to Grayscale using weighted sum
    rgb_to_gray_weights = np.array([0.2989, 0.5870, 0.1140])
    grayscale_converted = color_image @ rgb_to_gray_weights
    
    print(f"  RGB to Grayscale conversion shape: {grayscale_converted.shape}")
    print(f"  Converted grayscale range: [{grayscale_converted.min():.1f}, {grayscale_converted.max():.1f}]")
    
    return gray_image, color_image, red_channel, green_channel, blue_channel, grayscale_converted

gray_img, color_img, red_ch, green_ch, blue_ch, gray_converted = image_matrix_fundamentals()
```

**1.2 Geometric Transformations:**
```python
def geometric_transformations():
    """Demonstrate geometric transformations using linear algebra"""
    
    print("\nGeometric Transformations:")
    print("-" * 25)
    
    image = gray_img
    h, w = image.shape
    
    # 1. Translation
    print(f"1. Translation:")
    
    def translate_image(image, tx, ty):
        """Translate image using affine transformation matrix"""
        h, w = image.shape
        
        # Translation matrix
        T = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
        
        # Create coordinate meshgrid
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.vstack([x_coords.ravel(), y_coords.ravel(), np.ones(h*w)])
        
        # Apply transformation
        transformed_coords = T @ coords
        x_new = transformed_coords[0].reshape(h, w)
        y_new = transformed_coords[1].reshape(h, w)
        
        # Interpolate new image
        translated = np.zeros_like(image)
        
        # Simple nearest neighbor interpolation
        valid_mask = (x_new >= 0) & (x_new < w) & (y_new >= 0) & (y_new < h)
        x_valid = x_new[valid_mask].astype(int)
        y_valid = y_new[valid_mask].astype(int)
        
        translated[valid_mask] = image[y_valid, x_valid]
        
        return translated
    
    translated = translate_image(image, 50, 30)
    print(f"  Translated image shape: {translated.shape}")
    print(f"  Non-zero pixels: {np.count_nonzero(translated)}")
    
    # 2. Rotation
    print(f"\n2. Rotation:")
    
    def rotate_image(image, angle_degrees):
        """Rotate image using rotation matrix"""
        angle_rad = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotation matrix (around center)
        h, w = image.shape
        cx, cy = w // 2, h // 2
        
        R = np.array([
            [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
            [sin_a, cos_a, cy - sin_a * cx - cos_a * cy],
            [0, 0, 1]
        ])
        
        # Apply transformation
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.vstack([x_coords.ravel(), y_coords.ravel(), np.ones(h*w)])
        
        transformed_coords = R @ coords
        x_new = transformed_coords[0].reshape(h, w)
        y_new = transformed_coords[1].reshape(h, w)
        
        # Bilinear interpolation
        rotated = np.zeros_like(image, dtype=float)
        
        valid_mask = (x_new >= 0) & (x_new < w-1) & (y_new >= 0) & (y_new < h-1)
        
        x_floor = np.floor(x_new[valid_mask]).astype(int)
        y_floor = np.floor(y_new[valid_mask]).astype(int)
        x_ceil = x_floor + 1
        y_ceil = y_floor + 1
        
        # Bilinear weights
        wx = x_new[valid_mask] - x_floor
        wy = y_new[valid_mask] - y_floor
        
        # Bilinear interpolation
        rotated[valid_mask] = (
            (1 - wx) * (1 - wy) * image[y_floor, x_floor] +
            wx * (1 - wy) * image[y_floor, x_ceil] +
            (1 - wx) * wy * image[y_ceil, x_floor] +
            wx * wy * image[y_ceil, x_ceil]
        )
        
        return rotated.astype(image.dtype)
    
    rotated = rotate_image(image, 45)
    print(f"  Rotated image shape: {rotated.shape}")
    print(f"  Rotation preserves area: {np.sum(rotated > 0)} pixels")
    
    # 3. Scaling
    print(f"\n3. Scaling:")
    
    def scale_image(image, sx, sy):
        """Scale image using scaling matrix"""
        h, w = image.shape
        
        # Scaling matrix
        S = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])
        
        # Create new image dimensions
        new_h, new_w = int(h * sy), int(w * sx)
        scaled = np.zeros((new_h, new_w), dtype=image.dtype)
        
        # Inverse transformation for sampling
        S_inv = np.linalg.inv(S)
        
        y_coords, x_coords = np.mgrid[0:new_h, 0:new_w]
        coords = np.vstack([x_coords.ravel(), y_coords.ravel(), np.ones(new_h * new_w)])
        
        orig_coords = S_inv @ coords
        x_orig = orig_coords[0].reshape(new_h, new_w)
        y_orig = orig_coords[1].reshape(new_h, new_w)
        
        # Interpolate
        valid_mask = (x_orig >= 0) & (x_orig < w-1) & (y_orig >= 0) & (y_orig < h-1)
        
        x_floor = np.floor(x_orig[valid_mask]).astype(int)
        y_floor = np.floor(y_orig[valid_mask]).astype(int)
        x_ceil = x_floor + 1
        y_ceil = y_floor + 1
        
        wx = x_orig[valid_mask] - x_floor
        wy = y_orig[valid_mask] - y_floor
        
        scaled[valid_mask] = (
            (1 - wx) * (1 - wy) * image[y_floor, x_floor] +
            wx * (1 - wy) * image[y_floor, x_ceil] +
            (1 - wx) * wy * image[y_ceil, x_floor] +
            wx * wy * image[y_ceil, x_ceil]
        )
        
        return scaled
    
    scaled_up = scale_image(image, 1.5, 1.5)
    scaled_down = scale_image(image, 0.5, 0.5)
    
    print(f"  Original: {image.shape}")
    print(f"  Scaled up (1.5x): {scaled_up.shape}")
    print(f"  Scaled down (0.5x): {scaled_down.shape}")
    
    # 4. General Affine Transformation
    print(f"\n4. General Affine Transformation:")
    
    def affine_transform(image, matrix):
        """Apply general affine transformation"""
        h, w = image.shape
        
        # Apply transformation
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.vstack([x_coords.ravel(), y_coords.ravel(), np.ones(h*w)])
        
        transformed_coords = matrix @ coords
        x_new = transformed_coords[0].reshape(h, w)
        y_new = transformed_coords[1].reshape(h, w)
        
        # Create output image
        transformed = np.zeros_like(image, dtype=float)
        
        valid_mask = (x_new >= 0) & (x_new < w-1) & (y_new >= 0) & (y_new < h-1)
        
        # Simple nearest neighbor for demonstration
        x_round = np.round(x_new[valid_mask]).astype(int)
        y_round = np.round(y_new[valid_mask]).astype(int)
        
        # Ensure indices are valid
        valid_indices = (x_round >= 0) & (x_round < w) & (y_round >= 0) & (y_round < h)
        final_mask = np.zeros_like(valid_mask)
        final_mask[valid_mask] = valid_indices
        
        x_final = x_round[valid_indices]
        y_final = y_round[valid_indices]
        
        transformed[final_mask] = image[y_final, x_final]
        
        return transformed.astype(image.dtype)
    
    # Shear transformation
    shear_matrix = np.array([
        [1, 0.3, 0],
        [0.2, 1, 0],
        [0, 0, 1]
    ])
    
    sheared = affine_transform(image, shear_matrix)
    print(f"  Sheared image non-zero pixels: {np.count_nonzero(sheared)}")
    
    return translated, rotated, scaled_up, scaled_down, sheared

translated, rotated, scaled_up, scaled_down, sheared = geometric_transformations()
```

**2. Image Filtering and Convolution:**

```python
def image_filtering_convolution():
    """Demonstrate image filtering using linear algebra and convolution"""
    
    print("\nImage Filtering and Convolution:")
    print("-" * 31)
    
    image = gray_img.astype(float)
    
    # 1. Basic Convolution Kernels
    print(f"1. Basic Convolution Kernels:")
    
    # Define common kernels
    kernels = {
        'identity': np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]]),
        
        'box_blur': np.ones((5, 5)) / 25,
        
        'gaussian': np.array([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]]) / 16,
        
        'edge_horizontal': np.array([[-1, -1, -1],
                                   [0, 0, 0],
                                   [1, 1, 1]]),
        
        'edge_vertical': np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]]),
        
        'sobel_x': np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]]),
        
        'sobel_y': np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]]),
        
        'laplacian': np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]]),
        
        'sharpen': np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
    }
    
    # Apply kernels
    filtered_images = {}
    
    for kernel_name, kernel in kernels.items():
        filtered = convolve(image, kernel, mode='constant')
        filtered_images[kernel_name] = filtered
        
        print(f"  {kernel_name}: kernel shape {kernel.shape}, "
              f"output range [{filtered.min():.1f}, {filtered.max():.1f}]")
    
    # 2. Custom Convolution Implementation
    print(f"\n2. Custom Convolution Implementation:")
    
    def custom_convolve_2d(image, kernel, padding='valid'):
        """Custom 2D convolution implementation"""
        
        if len(image.shape) != 2 or len(kernel.shape) != 2:
            raise ValueError("Both image and kernel must be 2D")
        
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        
        if padding == 'valid':
            out_h = img_h - ker_h + 1
            out_w = img_w - ker_w + 1
            padded_image = image
        elif padding == 'same':
            out_h, out_w = img_h, img_w
            pad_h = ker_h // 2
            pad_w = ker_w // 2
            padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        else:
            raise ValueError("Padding must be 'valid' or 'same'")
        
        # Initialize output
        output = np.zeros((out_h, out_w))
        
        # Flip kernel (for convolution)
        kernel_flipped = np.flip(np.flip(kernel, 0), 1)
        
        # Perform convolution
        for i in range(out_h):
            for j in range(out_w):
                # Extract patch
                patch = padded_image[i:i+ker_h, j:j+ker_w]
                
                # Element-wise multiply and sum
                output[i, j] = np.sum(patch * kernel_flipped)
        
        return output
    
    # Test custom convolution
    custom_blur = custom_convolve_2d(image, kernels['gaussian'], padding='same')
    scipy_blur = convolve(image, kernels['gaussian'], mode='constant')
    
    print(f"  Custom convolution shape: {custom_blur.shape}")
    print(f"  SciPy convolution shape: {scipy_blur.shape}")
    print(f"  Results similar: {np.allclose(custom_blur, scipy_blur[:custom_blur.shape[0], :custom_blur.shape[1]], atol=1e-10)}")
    
    # 3. Separable Filters
    print(f"\n3. Separable Filters:")
    
    def create_separable_gaussian(size, sigma):
        """Create separable Gaussian filter"""
        
        # 1D Gaussian kernel
        x = np.arange(size) - size // 2
        gaussian_1d = np.exp(-x**2 / (2 * sigma**2))
        gaussian_1d /= np.sum(gaussian_1d)
        
        # 2D kernel as outer product
        gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
        
        return gaussian_1d, gaussian_2d
    
    # Compare separable vs non-separable filtering
    gaussian_1d, gaussian_2d = create_separable_gaussian(7, 1.5)
    
    # Method 1: Direct 2D convolution
    start_time = time.time()
    filtered_2d = convolve(image, gaussian_2d, mode='constant')
    time_2d = time.time() - start_time
    
    # Method 2: Separable (two 1D convolutions)
    start_time = time.time()
    temp = convolve(image, gaussian_1d.reshape(-1, 1), mode='constant')
    filtered_separable = convolve(temp, gaussian_1d.reshape(1, -1), mode='constant')
    time_separable = time.time() - start_time
    
    print(f"  2D convolution time: {time_2d:.4f}s")
    print(f"  Separable convolution time: {time_separable:.4f}s")
    print(f"  Speedup: {time_2d / time_separable:.1f}x")
    print(f"  Results identical: {np.allclose(filtered_2d, filtered_separable)}")
    
    # 4. Frequency Domain Filtering
    print(f"\n4. Frequency Domain Filtering:")
    
    def frequency_domain_filter(image, filter_func):
        """Apply filter in frequency domain"""
        
        # Forward FFT
        f_image = np.fft.fft2(image)
        f_image_shifted = np.fft.fftshift(f_image)
        
        # Create frequency filter
        h, w = image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u - w // 2
        v = v - h // 2
        
        # Apply filter
        filter_mask = filter_func(u, v, h, w)
        f_filtered = f_image_shifted * filter_mask
        
        # Inverse FFT
        f_filtered_shifted = np.fft.ifftshift(f_filtered)
        filtered_image = np.real(np.fft.ifft2(f_filtered_shifted))
        
        return filtered_image, filter_mask
    
    # Low-pass filter (Gaussian)
    def gaussian_lowpass(u, v, h, w, cutoff=50):
        d_squared = u**2 + v**2
        return np.exp(-d_squared / (2 * cutoff**2))
    
    # High-pass filter
    def gaussian_highpass(u, v, h, w, cutoff=30):
        return 1 - gaussian_lowpass(u, v, h, w, cutoff)
    
    # Apply frequency domain filters
    lowpass_filtered, lowpass_mask = frequency_domain_filter(image, gaussian_lowpass)
    highpass_filtered, highpass_mask = frequency_domain_filter(image, gaussian_highpass)
    
    print(f"  Frequency domain filtering completed")
    print(f"  Low-pass result range: [{lowpass_filtered.min():.1f}, {lowpass_filtered.max():.1f}]")
    print(f"  High-pass result range: [{highpass_filtered.min():.1f}, {highpass_filtered.max():.1f}]")
    
    return filtered_images, custom_blur, gaussian_1d, gaussian_2d, lowpass_filtered, highpass_filtered

filtered_imgs, custom_blur, gauss_1d, gauss_2d, lowpass_filt, highpass_filt = image_filtering_convolution()
```

**3. Image Compression using Matrix Decomposition:**

```python
def image_compression_svd():
    """Demonstrate image compression using SVD and other matrix decomposition techniques"""
    
    print("\nImage Compression using Matrix Decomposition:")
    print("-" * 43)
    
    image = gray_img.astype(float)
    
    # 1. SVD-based Image Compression
    print(f"1. SVD-based Compression:")
    
    # Perform SVD
    U, s, Vt = svd(image, full_matrices=False)
    
    print(f"  Original image shape: {image.shape}")
    print(f"  SVD components: U{U.shape}, s{s.shape}, Vt{Vt.shape}")
    print(f"  Total singular values: {len(s)}")
    
    # Analyze singular value distribution
    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
    
    # Find number of components for different compression ratios
    compression_ratios = [0.90, 0.95, 0.99]
    components_needed = []
    
    for ratio in compression_ratios:
        k = np.argmax(cumulative_energy >= ratio) + 1
        components_needed.append(k)
        print(f"  {ratio*100}% energy retained with {k} components ({k/len(s)*100:.1f}% of total)")
    
    # Reconstruct images with different numbers of components
    compressed_images = {}
    compression_stats = {}
    
    test_components = [1, 5, 10, 20, 50, 100, min(200, len(s))]
    
    for k in test_components:
        if k <= len(s):
            # Reconstruct using first k components
            reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            
            # Compute compression metrics
            original_size = image.size * 8  # Assume 8 bytes per pixel
            compressed_size = (U[:, :k].size + s[:k].size + Vt[:k, :].size) * 8
            compression_ratio = original_size / compressed_size
            
            # Compute quality metrics
            mse = np.mean((image - reconstructed)**2)
            psnr = 20 * np.log10(255) - 10 * np.log10(mse)
            
            compressed_images[k] = reconstructed
            compression_stats[k] = {
                'compression_ratio': compression_ratio,
                'mse': mse,
                'psnr': psnr,
                'energy_retained': cumulative_energy[k-1] if k <= len(cumulative_energy) else 1.0
            }
            
            print(f"  k={k:3d}: compression={compression_ratio:.1f}x, "
                  f"PSNR={psnr:.1f}dB, energy={cumulative_energy[k-1]*100:.1f}%")
    
    # 2. Block-based DCT Compression
    print(f"\n2. Block-based DCT Compression:")
    
    def dct_2d(block):
        """2D Discrete Cosine Transform"""
        return cv2.dct(block.astype(np.float32))
    
    def idct_2d(block):
        """2D Inverse Discrete Cosine Transform"""
        return cv2.idct(block.astype(np.float32))
    
    def block_dct_compress(image, block_size=8, quality_factor=50):
        """Compress image using block-based DCT"""
        
        h, w = image.shape
        
        # Pad image to make it divisible by block_size
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
        
        # JPEG-like quantization matrix (simplified)
        quantization_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ]) * (100 - quality_factor) / 50
        
        compressed_image = np.zeros_like(padded_image)
        
        # Process each block
        for i in range(0, padded_image.shape[0], block_size):
            for j in range(0, padded_image.shape[1], block_size):
                # Extract block
                block = padded_image[i:i+block_size, j:j+block_size]
                
                # DCT
                dct_block = dct_2d(block - 128)  # Center around 0
                
                # Quantization
                quantized_block = np.round(dct_block / quantization_matrix)
                
                # Dequantization
                dequantized_block = quantized_block * quantization_matrix
                
                # Inverse DCT
                reconstructed_block = idct_2d(dequantized_block) + 128
                
                compressed_image[i:i+block_size, j:j+block_size] = reconstructed_block
        
        # Remove padding
        compressed_image = compressed_image[:h, :w]
        
        return compressed_image
    
    # Apply DCT compression with different quality factors
    quality_factors = [10, 30, 50, 70, 90]
    dct_compressed = {}
    
    for quality in quality_factors:
        compressed = block_dct_compress(image, quality_factor=quality)
        mse = np.mean((image - compressed)**2)
        psnr = 20 * np.log10(255) - 10 * np.log10(mse)
        
        dct_compressed[quality] = compressed
        
        print(f"  Quality {quality}: PSNR = {psnr:.1f} dB")
    
    # 3. PCA-based Compression
    print(f"\n3. PCA-based Compression:")
    
    def pca_image_compression(image, n_components):
        """Compress image using PCA"""
        
        # Reshape image to 2D array (pixels as samples)
        h, w = image.shape
        image_flat = image.reshape(h, -1)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        compressed_data = pca.fit_transform(image_flat)
        
        # Reconstruct
        reconstructed_flat = pca.inverse_transform(compressed_data)
        reconstructed = reconstructed_flat.reshape(h, w)
        
        # Compute compression ratio
        original_size = h * w
        compressed_size = compressed_data.size + pca.components_.size + pca.mean_.size
        compression_ratio = original_size / compressed_size
        
        return reconstructed, compression_ratio, pca.explained_variance_ratio_
    
    # Test PCA compression
    pca_components = [10, 20, 50, 100]
    pca_compressed = {}
    
    for n_comp in pca_components:
        if n_comp < min(image.shape):
            compressed, comp_ratio, var_ratio = pca_image_compression(image, n_comp)
            
            mse = np.mean((image - compressed)**2)
            psnr = 20 * np.log10(255) - 10 * np.log10(mse)
            
            pca_compressed[n_comp] = compressed
            
            print(f"  {n_comp} components: compression={comp_ratio:.1f}x, "
                  f"PSNR={psnr:.1f}dB, variance={np.sum(var_ratio)*100:.1f}%")
    
    # 4. Comparison of Compression Methods
    print(f"\n4. Compression Method Comparison:")
    
    # Compare different methods at similar compression ratios
    target_psnr = 25  # dB
    
    print(f"  Targeting PSNR ≈ {target_psnr} dB:")
    
    # Find best SVD rank
    best_svd_k = None
    best_svd_psnr = 0
    for k, stats in compression_stats.items():
        if abs(stats['psnr'] - target_psnr) < abs(best_svd_psnr - target_psnr):
            best_svd_k = k
            best_svd_psnr = stats['psnr']
    
    if best_svd_k:
        print(f"    SVD (k={best_svd_k}): PSNR={best_svd_psnr:.1f}dB, "
              f"compression={compression_stats[best_svd_k]['compression_ratio']:.1f}x")
    
    return compressed_images, compression_stats, dct_compressed, pca_compressed

svd_compressed, comp_stats, dct_compressed, pca_compressed = image_compression_svd()
```

**4. Feature Extraction and Computer Vision:**

```python
def feature_extraction_computer_vision():
    """Demonstrate feature extraction using linear algebra techniques"""
    
    print("\nFeature Extraction and Computer Vision:")
    print("-" * 38)
    
    image = gray_img.astype(float)
    
    # 1. Edge Detection using Gradient Operators
    print(f"1. Edge Detection:")
    
    # Sobel operators
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    
    # Compute gradients
    grad_x = convolve(image, sobel_x, mode='constant')
    grad_y = convolve(image, sobel_y, mode='constant')
    
    # Gradient magnitude and direction
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_direction = np.arctan2(grad_y, grad_x)
    
    print(f"  Gradient magnitude range: [{gradient_magnitude.min():.1f}, {gradient_magnitude.max():.1f}]")
    print(f"  Gradient direction range: [{gradient_direction.min():.2f}, {gradient_direction.max():.2f}] radians")
    
    # Prewitt operators
    prewitt_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    
    prewitt_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
    
    prewitt_grad_x = convolve(image, prewitt_x, mode='constant')
    prewitt_grad_y = convolve(image, prewitt_y, mode='constant')
    prewitt_magnitude = np.sqrt(prewitt_grad_x**2 + prewitt_grad_y**2)
    
    print(f"  Prewitt magnitude range: [{prewitt_magnitude.min():.1f}, {prewitt_magnitude.max():.1f}]")
    
    # 2. Corner Detection using Structure Tensor
    print(f"\n2. Corner Detection (Harris):")
    
    def harris_corner_detection(image, k=0.04, threshold=0.01):
        """Harris corner detection using structure tensor"""
        
        # Compute gradients
        grad_x = convolve(image, sobel_x, mode='constant')
        grad_y = convolve(image, sobel_y, mode='constant')
        
        # Compute structure tensor components
        Ixx = grad_x * grad_x
        Iyy = grad_y * grad_y
        Ixy = grad_x * grad_y
        
        # Apply Gaussian smoothing to structure tensor
        sigma = 1.5
        gaussian_kernel = np.outer(
            np.exp(-np.arange(-3, 4)**2 / (2 * sigma**2)),
            np.exp(-np.arange(-3, 4)**2 / (2 * sigma**2))
        )
        gaussian_kernel /= np.sum(gaussian_kernel)
        
        Sxx = convolve(Ixx, gaussian_kernel, mode='constant')
        Syy = convolve(Iyy, gaussian_kernel, mode='constant')
        Sxy = convolve(Ixy, gaussian_kernel, mode='constant')
        
        # Compute Harris response
        det_S = Sxx * Syy - Sxy * Sxy
        trace_S = Sxx + Syy
        
        harris_response = det_S - k * trace_S**2
        
        # Find corners above threshold
        corners = harris_response > threshold * harris_response.max()
        
        return harris_response, corners
    
    harris_response, corners = harris_corner_detection(image)
    
    print(f"  Harris response range: [{harris_response.min():.2e}, {harris_response.max():.2e}]")
    print(f"  Detected corners: {np.sum(corners)}")
    
    # 3. Texture Analysis using Gray-Level Co-occurrence Matrix
    print(f"\n3. Texture Analysis (GLCM):")
    
    def compute_glcm(image, distance=1, angle=0):
        """Compute Gray-Level Co-occurrence Matrix"""
        
        # Quantize image to reduce GLCM size
        quantized = np.round(image / 4).astype(int)  # Reduce to 64 levels
        max_val = quantized.max()
        
        # Initialize GLCM
        glcm = np.zeros((max_val + 1, max_val + 1))
        
        # Compute offset based on distance and angle
        dy = int(distance * np.sin(angle))
        dx = int(distance * np.cos(angle))
        
        h, w = quantized.shape
        
        # Fill GLCM
        for i in range(h):
            for j in range(w):
                # Check if neighbor is within bounds
                ni, nj = i + dy, j + dx
                if 0 <= ni < h and 0 <= nj < w:
                    glcm[quantized[i, j], quantized[ni, nj]] += 1
        
        # Normalize GLCM
        glcm = glcm / np.sum(glcm)
        
        return glcm
    
    def glcm_features(glcm):
        """Compute texture features from GLCM"""
        
        # Contrast
        i, j = np.meshgrid(range(glcm.shape[0]), range(glcm.shape[1]), indexing='ij')
        contrast = np.sum(glcm * (i - j)**2)
        
        # Energy (Angular Second Moment)
        energy = np.sum(glcm**2)
        
        # Homogeneity
        homogeneity = np.sum(glcm / (1 + (i - j)**2))
        
        # Correlation
        mu_i = np.sum(i * glcm)
        mu_j = np.sum(j * glcm)
        sigma_i = np.sqrt(np.sum((i - mu_i)**2 * glcm))
        sigma_j = np.sqrt(np.sum((j - mu_j)**2 * glcm))
        
        if sigma_i > 0 and sigma_j > 0:
            correlation = np.sum((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j)
        else:
            correlation = 0
        
        return {
            'contrast': contrast,
            'energy': energy,
            'homogeneity': homogeneity,
            'correlation': correlation
        }
    
    # Compute GLCM for different directions
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    texture_features = {}
    
    for i, angle in enumerate(angles):
        glcm = compute_glcm(image, distance=1, angle=angle)
        features = glcm_features(glcm)
        texture_features[f'angle_{int(np.degrees(angle))}'] = features
        
        print(f"  Angle {int(np.degrees(angle))}°: contrast={features['contrast']:.3f}, "
              f"energy={features['energy']:.3f}, homogeneity={features['homogeneity']:.3f}")
    
    # 4. Histogram of Oriented Gradients (HOG)
    print(f"\n4. Histogram of Oriented Gradients:")
    
    def compute_hog_features(image, cell_size=8, block_size=2, n_bins=9):
        """Compute HOG features"""
        
        # Compute gradients
        grad_x = convolve(image, sobel_x, mode='constant')
        grad_y = convolve(image, sobel_y, mode='constant')
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi
        orientation[orientation < 0] += 180  # Convert to 0-180 range
        
        h, w = image.shape
        
        # Compute HOG for each cell
        n_cells_y = h // cell_size
        n_cells_x = w // cell_size
        
        cell_histograms = np.zeros((n_cells_y, n_cells_x, n_bins))
        
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Extract cell
                y_start, y_end = i * cell_size, (i + 1) * cell_size
                x_start, x_end = j * cell_size, (j + 1) * cell_size
                
                cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
                cell_orientation = orientation[y_start:y_end, x_start:x_end]
                
                # Compute histogram
                hist, _ = np.histogram(
                    cell_orientation.ravel(),
                    bins=n_bins,
                    range=(0, 180),
                    weights=cell_magnitude.ravel()
                )
                
                cell_histograms[i, j] = hist
        
        # Block normalization
        n_blocks_y = n_cells_y - block_size + 1
        n_blocks_x = n_cells_x - block_size + 1
        
        hog_features = []
        
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                # Extract block
                block = cell_histograms[i:i+block_size, j:j+block_size].ravel()
                
                # L2 normalization
                norm = np.linalg.norm(block)
                if norm > 0:
                    block = block / norm
                
                hog_features.extend(block)
        
        return np.array(hog_features)
    
    hog_features = compute_hog_features(image)
    
    print(f"  HOG feature vector length: {len(hog_features)}")
    print(f"  HOG feature range: [{hog_features.min():.3f}, {hog_features.max():.3f}]")
    print(f"  HOG feature mean: {hog_features.mean():.3f}")
    
    return gradient_magnitude, gradient_direction, harris_response, corners, texture_features, hog_features

grad_mag, grad_dir, harris_resp, corners, texture_feat, hog_feat = feature_extraction_computer_vision()
```

**5. Advanced Image Processing Applications:**

```python
def advanced_image_processing():
    """Demonstrate advanced image processing applications using linear algebra"""
    
    print("\nAdvanced Image Processing Applications:")
    print("-" * 40)
    
    image = gray_img.astype(float)
    
    # 1. Image Denoising using Total Variation
    print(f"1. Image Denoising:")
    
    # Add noise to image
    noise_level = 25
    noisy_image = image + np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # Total Variation denoising
    denoised_tv = denoise_tv_chambolle(noisy_image, weight=0.1)
    
    # Gaussian denoising
    denoised_gaussian = gaussian_filter(noisy_image, sigma=1.0)
    
    # Compute quality metrics
    def compute_psnr(original, processed):
        mse = np.mean((original - processed)**2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255) - 10 * np.log10(mse)
    
    psnr_noisy = compute_psnr(image, noisy_image)
    psnr_tv = compute_psnr(image, denoised_tv * 255)
    psnr_gaussian = compute_psnr(image, denoised_gaussian)
    
    print(f"  Original vs Noisy: PSNR = {psnr_noisy:.1f} dB")
    print(f"  TV Denoising: PSNR = {psnr_tv:.1f} dB")
    print(f"  Gaussian Denoising: PSNR = {psnr_gaussian:.1f} dB")
    
    # 2. Image Inpainting using Matrix Completion
    print(f"\n2. Image Inpainting:")
    
    def create_mask(shape, missing_ratio=0.3):
        """Create random mask for inpainting"""
        mask = np.random.rand(*shape) > missing_ratio
        return mask
    
    def matrix_completion_inpainting(image, mask, max_iter=100):
        """Simple matrix completion for inpainting"""
        
        # Initialize with mean of known pixels
        inpainted = image.copy()
        mean_value = np.mean(image[mask])
        inpainted[~mask] = mean_value
        
        for iteration in range(max_iter):
            # Apply low-rank approximation
            U, s, Vt = svd(inpainted, full_matrices=False)
            
            # Keep top components (adjust rank as needed)
            rank = min(50, len(s))
            inpainted_lowrank = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
            
            # Restore known pixels
            inpainted_lowrank[mask] = image[mask]
            
            # Check convergence
            if iteration > 0:
                change = np.linalg.norm(inpainted - inpainted_lowrank)
                if change < 1e-3:
                    break
            
            inpainted = inpainted_lowrank
        
        return inpainted
    
    # Create damaged image
    mask = create_mask(image.shape, missing_ratio=0.2)
    damaged_image = image.copy()
    damaged_image[~mask] = 0
    
    # Inpaint
    inpainted = matrix_completion_inpainting(damaged_image, mask)
    
    psnr_damaged = compute_psnr(image, damaged_image)
    psnr_inpainted = compute_psnr(image, inpainted)
    
    print(f"  Damaged image: PSNR = {psnr_damaged:.1f} dB")
    print(f"  Inpainted image: PSNR = {psnr_inpainted:.1f} dB")
    print(f"  Missing pixels: {np.sum(~mask)} ({np.sum(~mask)/mask.size*100:.1f}%)")
    
    # 3. Image Registration using Cross-Correlation
    print(f"\n3. Image Registration:")
    
    def template_matching(image, template):
        """Template matching using normalized cross-correlation"""
        
        # Normalize template and image
        template_norm = (template - np.mean(template)) / np.std(template)
        
        h_img, w_img = image.shape
        h_temp, w_temp = template.shape
        
        result = np.zeros((h_img - h_temp + 1, w_img - w_temp + 1))
        
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                # Extract patch
                patch = image[i:i+h_temp, j:j+w_temp]
                patch_norm = (patch - np.mean(patch)) / np.std(patch)
                
                # Normalized cross-correlation
                correlation = np.sum(patch_norm * template_norm) / (h_temp * w_temp)
                result[i, j] = correlation
        
        return result
    
    # Create template from center of image
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    template_size = 50
    template = image[
        center_y - template_size//2:center_y + template_size//2,
        center_x - template_size//2:center_x + template_size//2
    ]
    
    # Find template in image
    correlation_map = template_matching(image, template)
    
    # Find best match
    best_match = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
    max_correlation = correlation_map[best_match]
    
    print(f"  Template size: {template.shape}")
    print(f"  Best match at: {best_match}")
    print(f"  Max correlation: {max_correlation:.3f}")
    
    # 4. Image Morphology using Structuring Elements
    print(f"\n4. Mathematical Morphology:")
    
    def morphological_operation(image, structuring_element, operation='erosion'):
        """Basic morphological operations"""
        
        # Convert to binary if needed
        binary_image = (image > 128).astype(int)
        
        h_img, w_img = binary_image.shape
        h_se, w_se = structuring_element.shape
        
        result = np.zeros_like(binary_image)
        
        # Pad image
        pad_h, pad_w = h_se // 2, w_se // 2
        padded = np.pad(binary_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        for i in range(h_img):
            for j in range(w_img):
                # Extract neighborhood
                neighborhood = padded[i:i+h_se, j:j+w_se]
                
                if operation == 'erosion':
                    # Erosion: all SE pixels must match
                    result[i, j] = int(np.all(neighborhood >= structuring_element))
                elif operation == 'dilation':
                    # Dilation: any SE pixel matches
                    result[i, j] = int(np.any(neighborhood * structuring_element))
        
        return result
    
    # Create structuring elements
    se_cross = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    
    se_square = np.ones((5, 5))
    
    # Apply morphological operations
    binary_image = (image > 128).astype(int)
    
    eroded = morphological_operation(binary_image, se_cross, 'erosion')
    dilated = morphological_operation(binary_image, se_cross, 'dilation')
    
    # Opening and closing
    opened = morphological_operation(eroded, se_cross, 'dilation')
    closed = morphological_operation(dilated, se_cross, 'erosion')
    
    print(f"  Binary image pixels: {np.sum(binary_image)}")
    print(f"  After erosion: {np.sum(eroded)}")
    print(f"  After dilation: {np.sum(dilated)}")
    print(f"  After opening: {np.sum(opened)}")
    print(f"  After closing: {np.sum(closed)}")
    
    return noisy_image, denoised_tv, inpainted, correlation_map, binary_image, eroded, dilated

noisy_img, denoised_img, inpainted_img, corr_map, binary_img, eroded_img, dilated_img = advanced_image_processing()
```

**6. Complete Image Processing Framework:**

```python
def complete_image_processing_framework():
    """Comprehensive framework demonstrating linear algebra in image processing"""
    
    print("\nComplete Image Processing Framework:")
    print("=" * 38)
    
    class LinearAlgebraImageProcessor:
        """Image processing framework using linear algebra"""
        
        def __init__(self):
            self.filters = {}
            self.transforms = {}
            self.features = {}
            
        def add_filter(self, name, kernel):
            """Add convolution filter"""
            self.filters[name] = kernel
            
        def apply_filter(self, image, filter_name):
            """Apply named filter to image"""
            if filter_name not in self.filters:
                raise ValueError(f"Filter '{filter_name}' not found")
            return convolve(image, self.filters[filter_name], mode='constant')
        
        def geometric_transform(self, image, transform_matrix):
            """Apply geometric transformation"""
            # This would use scipy.ndimage.affine_transform in practice
            return transform.warp(image, transform_matrix.T)
        
        def compress_svd(self, image, k):
            """Compress image using SVD"""
            U, s, Vt = svd(image, full_matrices=False)
            return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        def extract_hog_features(self, image):
            """Extract HOG features"""
            return feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys')
        
        def process_pipeline(self, image, operations):
            """Process image through pipeline of operations"""
            result = image.copy()
            
            for operation, params in operations:
                if operation == 'filter':
                    result = self.apply_filter(result, params['name'])
                elif operation == 'transform':
                    result = self.geometric_transform(result, params['matrix'])
                elif operation == 'compress':
                    result = self.compress_svd(result, params['k'])
                elif operation == 'normalize':
                    result = (result - result.min()) / (result.max() - result.min())
                
            return result
    
    # Demonstrate framework
    processor = LinearAlgebraImageProcessor()
    
    # Add standard filters
    processor.add_filter('gaussian', np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16)
    processor.add_filter('sobel_x', np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    processor.add_filter('laplacian', np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
    
    # Define processing pipeline
    pipeline = [
        ('filter', {'name': 'gaussian'}),
        ('compress', {'k': 50}),
        ('normalize', {}),
        ('filter', {'name': 'sobel_x'})
    ]
    
    # Process image
    processed = processor.process_pipeline(gray_img.astype(float), pipeline)
    
    print(f"  Framework initialized with {len(processor.filters)} filters")
    print(f"  Pipeline applied: {len(pipeline)} operations")
    print(f"  Final image range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Performance analysis
    print(f"\n  Performance Summary:")
    print(f"    Original image: {gray_img.shape} ({gray_img.size} pixels)")
    print(f"    Processing pipeline: {len(pipeline)} stages")
    print(f"    Output characteristics: shape={processed.shape}, dtype={processed.dtype}")
    
    # Applications summary
    applications = {
        'Computer Vision': ['Edge detection', 'Corner detection', 'Feature extraction'],
        'Image Enhancement': ['Denoising', 'Sharpening', 'Contrast adjustment'],
        'Compression': ['SVD-based', 'DCT-based', 'PCA-based'],
        'Medical Imaging': ['Image registration', 'Segmentation', 'Enhancement'],
        'Robotics': ['Object recognition', 'SLAM', 'Navigation'],
        'Graphics': ['Filtering', 'Transformations', 'Morphology']
    }
    
    print(f"\n  Linear Algebra Applications in Image Processing:")
    for domain, techniques in applications.items():
        print(f"    {domain}: {', '.join(techniques)}")
    
    return processor, processed

processor, processed_img = complete_image_processing_framework()
```

**Summary and Key Benefits:**

**Linear Algebra in Image Processing provides:**

1. **Efficient Representation**: Images as matrices enable vectorized operations
2. **Geometric Transformations**: Affine transformations via matrix multiplication
3. **Filtering**: Convolution as matrix operations for noise reduction and enhancement
4. **Compression**: SVD, DCT, and PCA for lossy/lossless compression
5. **Feature Extraction**: Gradients, corners, textures using matrix computations
6. **Advanced Processing**: Denoising, inpainting, morphology through linear methods

**Core Applications:**
- **Computer Vision**: Object detection, recognition, tracking
- **Medical Imaging**: Enhancement, segmentation, registration
- **Remote Sensing**: Satellite image analysis, change detection
- **Graphics**: Rendering, filtering, special effects
- **Robotics**: Visual navigation, SLAM, perception

The mathematical rigor of linear algebra provides both theoretical understanding and computational efficiency for modern image processing systems.

---

## Question 6

**Discuss the role of linear algebra in deep learning, specifically in training convolutional neural networks.**

**Answer:** Linear algebra serves as the mathematical foundation of deep learning and convolutional neural networks (CNNs), providing the computational framework for forward propagation, backpropagation, weight updates, and optimization. Understanding these linear algebraic operations is crucial for designing, implementing, and optimizing deep learning systems effectively.

**1. Forward Propagation in CNNs:**

**1.1 Convolution as Matrix Multiplication:**
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

def convolution_as_matrix_multiplication():
    """Demonstrate convolution as matrix multiplication"""
    
    print("Convolution as Matrix Multiplication")
    print("=" * 35)
    
    # Create sample input
    input_size = 5
    kernel_size = 3
    
    # Input image (5x5)
    X = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ], dtype=float)
    
    # Kernel (3x3)
    K = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=float)
    
    print(f"Input shape: {X.shape}")
    print(f"Kernel shape: {K.shape}")
    
    # Method 1: Direct convolution
    output_direct = convolve2d(X, K, mode='valid')
    print(f"Direct convolution output shape: {output_direct.shape}")
    
    # Method 2: Convert to matrix multiplication
    def convolution_to_toeplitz(input_matrix, kernel, output_shape):
        """Convert convolution to Toeplitz matrix multiplication"""
        
        input_h, input_w = input_matrix.shape
        kernel_h, kernel_w = kernel.shape
        output_h, output_w = output_shape
        
        # Create Toeplitz matrix
        toeplitz_matrix = np.zeros((output_h * output_w, input_h * input_w))
        
        # Fill Toeplitz matrix
        for i in range(output_h):
            for j in range(output_w):
                output_idx = i * output_w + j
                
                for ki in range(kernel_h):
                    for kj in range(kernel_w):
                        input_i = i + ki
                        input_j = j + kj
                        input_idx = input_i * input_w + input_j
                        
                        if 0 <= input_i < input_h and 0 <= input_j < input_w:
                            toeplitz_matrix[output_idx, input_idx] = kernel[ki, kj]
        
        return toeplitz_matrix
    
    # Create Toeplitz matrix
    output_shape = (input_size - kernel_size + 1, input_size - kernel_size + 1)
    toeplitz_matrix = convolution_to_toeplitz(X, K, output_shape)
    
    # Perform matrix multiplication
    X_flattened = X.flatten()
    output_matrix = toeplitz_matrix @ X_flattened
    output_matrix = output_matrix.reshape(output_shape)
    
    print(f"Matrix multiplication output shape: {output_matrix.shape}")
    print(f"Results match: {np.allclose(output_direct, output_matrix)}")
    
    print(f"\nToeplitz matrix shape: {toeplitz_matrix.shape}")
    print(f"Sparsity: {np.sum(toeplitz_matrix == 0) / toeplitz_matrix.size * 100:.1f}% zeros")
    
    return X, K, output_direct, toeplitz_matrix

X, K, output_conv, toeplitz = convolution_as_matrix_multiplication()
```

**1.2 Multi-Channel Convolution:**
```python
def multi_channel_convolution():
    """Demonstrate multi-channel convolution operations"""
    
    print("\nMulti-Channel Convolution:")
    print("-" * 26)
    
    # Multi-channel input (batch_size=1, channels=3, height=5, width=5)
    batch_size, in_channels, height, width = 1, 3, 5, 5
    out_channels = 2
    kernel_size = 3
    
    # Create random input
    np.random.seed(42)
    X = np.random.randn(batch_size, in_channels, height, width)
    
    # Create random kernels (out_channels, in_channels, kernel_height, kernel_width)
    W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
    
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}")
    
    # Manual convolution computation
    def manual_conv2d(input_tensor, weights):
        """Manual 2D convolution computation"""
        
        batch_size, in_channels, in_h, in_w = input_tensor.shape
        out_channels, _, kernel_h, kernel_w = weights.shape
        
        out_h = in_h - kernel_h + 1
        out_w = in_w - kernel_w + 1
        
        output = np.zeros((batch_size, out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Extract patch
                        patch = input_tensor[b, :, i:i+kernel_h, j:j+kernel_w]
                        
                        # Compute dot product with kernel
                        output[b, oc, i, j] = np.sum(patch * weights[oc])
        
        return output
    
    # Compute using manual implementation
    output_manual = manual_conv2d(X, W)
    
    # Compare with PyTorch
    X_torch = torch.tensor(X, dtype=torch.float32)
    W_torch = torch.tensor(W, dtype=torch.float32)
    
    output_torch = F.conv2d(X_torch, W_torch)
    
    print(f"Manual output shape: {output_manual.shape}")
    print(f"PyTorch output shape: {output_torch.shape}")
    print(f"Results match: {np.allclose(output_manual, output_torch.numpy(), atol=1e-6)}")
    
    # Analyze computational complexity
    input_elements = np.prod(X.shape)
    weight_elements = np.prod(W.shape)
    output_elements = np.prod(output_manual.shape)
    
    operations = output_elements * in_channels * kernel_size * kernel_size
    
    print(f"\nComputational Analysis:")
    print(f"  Input elements: {input_elements:,}")
    print(f"  Weight elements: {weight_elements:,}")
    print(f"  Output elements: {output_elements:,}")
    print(f"  Multiply-add operations: {operations:,}")
    
    return X, W, output_manual

X_multi, W_multi, output_multi = multi_channel_convolution()
```

**1.3 Batch Processing and Vectorization:**
```python
def batch_processing_vectorization():
    """Demonstrate batch processing and vectorization in CNNs"""
    
    print("\nBatch Processing and Vectorization:")
    print("-" * 34)
    
    # Batch processing parameters
    batch_sizes = [1, 8, 32, 128]
    in_channels, out_channels = 64, 128
    input_size = 32
    kernel_size = 3
    
    # Create sample data
    def create_batch_data(batch_size):
        X = torch.randn(batch_size, in_channels, input_size, input_size)
        W = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        return X, W
    
    print("Batch Size | Time (ms) | Throughput (samples/s) | Memory (MB)")
    print("-" * 65)
    
    timing_results = {}
    
    for batch_size in batch_sizes:
        X, W = create_batch_data(batch_size)
        
        # Time the convolution
        start_time = time.time()
        
        # Multiple runs for accurate timing
        num_runs = 10
        for _ in range(num_runs):
            output = F.conv2d(X, W, padding=1)
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        throughput = batch_size / (avg_time / 1000)  # samples per second
        
        # Estimate memory usage
        input_memory = X.element_size() * X.nelement() / (1024**2)
        weight_memory = W.element_size() * W.nelement() / (1024**2)
        output_memory = output.element_size() * output.nelement() / (1024**2)
        total_memory = input_memory + weight_memory + output_memory
        
        print(f"{batch_size:10d} | {avg_time:8.2f} | {throughput:17.1f} | {total_memory:10.1f}")
        
        timing_results[batch_size] = {
            'time': avg_time,
            'throughput': throughput,
            'memory': total_memory
        }
    
    # Analyze scaling
    print(f"\nScaling Analysis:")
    base_batch = batch_sizes[0]
    for batch_size in batch_sizes[1:]:
        theoretical_speedup = batch_size / base_batch
        actual_speedup = timing_results[base_batch]['time'] / timing_results[batch_size]['time'] * batch_size
        efficiency = actual_speedup / theoretical_speedup * 100
        
        print(f"  Batch {batch_size}: {efficiency:.1f}% efficiency vs theoretical")
    
    return timing_results

timing_results = batch_processing_vectorization()
```

**2. Backpropagation and Gradient Computation:**

**2.1 Gradient Flow through Convolution:**
```python
def convolution_gradients():
    """Demonstrate gradient computation in convolutional layers"""
    
    print("\nGradient Computation in Convolution:")
    print("-" * 35)
    
    # Simple example with small dimensions
    batch_size, in_channels, height, width = 1, 1, 4, 4
    out_channels = 1
    kernel_size = 3
    
    # Create input and weight tensors with gradient tracking
    X = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    W = torch.randn(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)
    b = torch.randn(out_channels, requires_grad=True)
    
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}")
    print(f"Bias shape: {b.shape}")
    
    # Forward pass
    output = F.conv2d(X, W, b)
    print(f"Output shape: {output.shape}")
    
    # Create a simple loss (sum of all outputs)
    loss = output.sum()
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nGradient shapes:")
    print(f"  dL/dX: {X.grad.shape}")
    print(f"  dL/dW: {W.grad.shape}")
    print(f"  dL/db: {b.grad.shape}")
    
    # Manual gradient computation for verification
    def manual_conv_gradient(input_tensor, weight, grad_output):
        """Manual computation of convolution gradients"""
        
        # Gradient w.r.t. input (convolution with flipped kernel)
        grad_input = F.conv_transpose2d(grad_output, weight)
        
        # Gradient w.r.t. weight (convolution of input with grad_output)
        grad_weight = F.conv2d(
            input_tensor.transpose(0, 1), 
            grad_output.transpose(0, 1)
        ).transpose(0, 1)
        
        # Gradient w.r.t. bias (sum over spatial dimensions)
        grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias
    
    # Verify gradients manually
    grad_output = torch.ones_like(output)
    grad_input_manual, grad_weight_manual, grad_bias_manual = manual_conv_gradient(X, W, grad_output)
    
    print(f"\nGradient verification:")
    print(f"  Input gradients match: {torch.allclose(X.grad, grad_input_manual[:, :, :X.shape[2], :X.shape[3]], atol=1e-6)}")
    print(f"  Weight gradients match: {torch.allclose(W.grad, grad_weight_manual, atol=1e-6)}")
    print(f"  Bias gradients match: {torch.allclose(b.grad, grad_bias_manual, atol=1e-6)}")
    
    return X, W, b, output

X_grad, W_grad, b_grad, output_grad = convolution_gradients()
```

**2.2 Chain Rule in Deep Networks:**
```python
def chain_rule_deep_networks():
    """Demonstrate chain rule application in deep networks"""
    
    print("\nChain Rule in Deep Networks:")
    print("-" * 28)
    
    # Create a simple CNN architecture
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # Store intermediate activations for analysis
            self.activations = {}
            
            x = self.conv1(x)
            self.activations['conv1'] = x.clone()
            x = self.relu(x)
            x = self.pool(x)
            
            x = self.conv2(x)
            self.activations['conv2'] = x.clone()
            x = self.relu(x)
            x = self.pool(x)
            
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            self.activations['fc1'] = x.clone()
            x = self.relu(x)
            
            x = self.fc2(x)
            self.activations['fc2'] = x.clone()
            
            return x
    
    # Create model and sample data
    model = SimpleCNN()
    X = torch.randn(4, 1, 32, 32, requires_grad=True)
    target = torch.randint(0, 10, (4,))
    
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {target.shape}")
    
    # Forward pass
    output = model(X)
    print(f"Output shape: {output.shape}")
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Analyze gradients at each layer
    print(f"\nGradient Analysis:")
    
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
    
    for name in layer_names:
        layer = getattr(model, name)
        
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            weight_grad_norm = torch.norm(layer.weight.grad).item()
            weight_norm = torch.norm(layer.weight).item()
            relative_grad = weight_grad_norm / weight_norm
            
            print(f"  {name}:")
            print(f"    Weight gradient norm: {weight_grad_norm:.6f}")
            print(f"    Weight norm: {weight_norm:.6f}")
            print(f"    Relative gradient: {relative_grad:.6f}")
    
    # Gradient flow analysis
    def analyze_gradient_flow():
        """Analyze gradient magnitudes through the network"""
        
        gradients = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.norm().item())
                layer_names.append(name)
        
        return gradients, layer_names
    
    gradients, param_names = analyze_gradient_flow()
    
    print(f"\nGradient Flow Summary:")
    for name, grad in zip(param_names, gradients):
        print(f"  {name}: {grad:.6f}")
    
    # Check for vanishing/exploding gradients
    grad_ratios = []
    for i in range(1, len(gradients)):
        ratio = gradients[i] / gradients[i-1]
        grad_ratios.append(ratio)
    
    print(f"\nGradient Ratios (layer i+1 / layer i):")
    for i, ratio in enumerate(grad_ratios):
        print(f"  Layer {i+1}/{i}: {ratio:.4f}")
    
    return model, X, output, gradients

model, X_cnn, output_cnn, gradients = chain_rule_deep_networks()
```

**3. Weight Updates and Optimization:**

**3.1 Matrix-Based Optimization Algorithms:**
```python
def matrix_optimization_algorithms():
    """Demonstrate matrix-based optimization algorithms"""
    
    print("\nMatrix-Based Optimization Algorithms:")
    print("-" * 38)
    
    # Create a simple linear layer for demonstration
    input_size, output_size = 100, 50
    batch_size = 32
    
    # Initialize weights and biases
    W = torch.randn(output_size, input_size, requires_grad=True) * 0.01
    b = torch.zeros(output_size, requires_grad=True)
    
    # Generate sample data
    X = torch.randn(batch_size, input_size)
    y = torch.randn(batch_size, output_size)
    
    print(f"Weight matrix shape: {W.shape}")
    print(f"Input batch shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Different optimization algorithms
    optimizers = {
        'SGD': torch.optim.SGD([W, b], lr=0.01),
        'Adam': torch.optim.Adam([W, b], lr=0.001),
        'RMSprop': torch.optim.RMSprop([W, b], lr=0.001),
        'AdaGrad': torch.optim.Adagrad([W, b], lr=0.01)
    }
    
    # Training loop for each optimizer
    def train_with_optimizer(optimizer_name, optimizer, num_epochs=100):
        """Train with specific optimizer"""
        
        losses = []
        weight_norms = []
        
        # Reset weights
        W.data = torch.randn(output_size, input_size) * 0.01
        b.data = torch.zeros(output_size)
        
        for epoch in range(num_epochs):
            # Forward pass
            output = X @ W.T + b
            loss = F.mse_loss(output, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Record metrics
            losses.append(loss.item())
            weight_norms.append(torch.norm(W).item())
        
        return losses, weight_norms
    
    # Train with different optimizers
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name}:")
        losses, weight_norms = train_with_optimizer(name, optimizer)
        
        final_loss = losses[-1]
        final_weight_norm = weight_norms[-1]
        
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Final weight norm: {final_weight_norm:.6f}")
        print(f"  Convergence rate: {(losses[0] - final_loss) / losses[0] * 100:.2f}%")
        
        results[name] = {
            'losses': losses,
            'weight_norms': weight_norms,
            'final_loss': final_loss
        }
    
    # Compare convergence
    print(f"\nConvergence Comparison:")
    sorted_optimizers = sorted(results.items(), key=lambda x: x[1]['final_loss'])
    
    for i, (name, metrics) in enumerate(sorted_optimizers):
        print(f"  {i+1}. {name}: {metrics['final_loss']:.6f}")
    
    return results

opt_results = matrix_optimization_algorithms()
```

**3.2 Second-Order Optimization Methods:**
```python
def second_order_optimization():
    """Demonstrate second-order optimization methods"""
    
    print("\nSecond-Order Optimization Methods:")
    print("-" * 33)
    
    # Simple quadratic function for demonstration
    def quadratic_function(x):
        """Quadratic function: f(x) = x^T A x + b^T x + c"""
        A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        b = torch.tensor([1.0, -1.0])
        c = 0.5
        
        return x.T @ A @ x + b.T @ x + c
    
    def quadratic_gradient(x):
        """Gradient of quadratic function"""
        A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        b = torch.tensor([1.0, -1.0])
        
        return 2 * A @ x + b
    
    def quadratic_hessian():
        """Hessian of quadratic function"""
        A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        return 2 * A
    
    # Starting point
    x0 = torch.tensor([3.0, 2.0])
    
    print(f"Starting point: {x0}")
    print(f"Initial function value: {quadratic_function(x0):.6f}")
    
    # 1. Gradient Descent
    def gradient_descent(x_start, lr=0.1, num_steps=20):
        """Standard gradient descent"""
        x = x_start.clone()
        trajectory = [x.clone()]
        
        for _ in range(num_steps):
            grad = quadratic_gradient(x)
            x = x - lr * grad
            trajectory.append(x.clone())
        
        return x, trajectory
    
    x_gd, traj_gd = gradient_descent(x0)
    
    # 2. Newton's Method
    def newton_method(x_start, num_steps=10):
        """Newton's method using Hessian"""
        x = x_start.clone()
        trajectory = [x.clone()]
        
        H = quadratic_hessian()
        H_inv = torch.inverse(H)
        
        for _ in range(num_steps):
            grad = quadratic_gradient(x)
            x = x - H_inv @ grad
            trajectory.append(x.clone())
        
        return x, trajectory
    
    x_newton, traj_newton = newton_method(x0)
    
    # 3. Quasi-Newton (BFGS approximation)
    def quasi_newton_bfgs(x_start, num_steps=15):
        """Quasi-Newton method with BFGS approximation"""
        x = x_start.clone()
        trajectory = [x.clone()]
        
        # Initialize Hessian approximation as identity
        H_inv = torch.eye(2)
        
        for i in range(num_steps):
            grad = quadratic_gradient(x)
            
            # Update step
            dx = -H_inv @ grad
            x_new = x + 0.1 * dx  # Line search step size
            
            if i > 0:
                # BFGS update
                grad_new = quadratic_gradient(x_new)
                s = x_new - x
                y = grad_new - grad
                
                if torch.dot(s, y) > 1e-8:  # Check curvature condition
                    rho = 1.0 / torch.dot(s, y)
                    I = torch.eye(2)
                    
                    H_inv = (I - rho * torch.outer(s, y)) @ H_inv @ (I - rho * torch.outer(y, s)) + rho * torch.outer(s, s)
            
            x = x_new
            trajectory.append(x.clone())
        
        return x, trajectory
    
    x_bfgs, traj_bfgs = quasi_newton_bfgs(x0)
    
    # Compare methods
    methods = {
        'Gradient Descent': (x_gd, traj_gd),
        'Newton Method': (x_newton, traj_newton),
        'Quasi-Newton (BFGS)': (x_bfgs, traj_bfgs)
    }
    
    print(f"\nOptimization Results:")
    for name, (x_final, trajectory) in methods.items():
        final_value = quadratic_function(x_final)
        steps = len(trajectory) - 1
        
        print(f"  {name}:")
        print(f"    Final point: [{x_final[0]:.6f}, {x_final[1]:.6f}]")
        print(f"    Final value: {final_value:.6f}")
        print(f"    Steps: {steps}")
    
    # Theoretical optimum
    H = quadratic_hessian()
    b = torch.tensor([1.0, -1.0])
    x_opt = -0.5 * torch.inverse(H) @ b
    f_opt = quadratic_function(x_opt)
    
    print(f"\nTheoretical optimum:")
    print(f"  Point: [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
    print(f"  Value: {f_opt:.6f}")
    
    return methods, x_opt

methods, x_opt = second_order_optimization()
```

**4. Specialized CNN Operations:**

**4.1 Depthwise and Pointwise Convolutions:**
```python
def depthwise_pointwise_convolutions():
    """Demonstrate depthwise and pointwise convolutions"""
    
    print("\nDepthwise and Pointwise Convolutions:")
    print("-" * 37)
    
    # Input parameters
    batch_size, in_channels, height, width = 1, 32, 64, 64
    out_channels = 64
    kernel_size = 3
    
    # Create input
    X = torch.randn(batch_size, in_channels, height, width)
    
    print(f"Input shape: {X.shape}")
    
    # 1. Standard Convolution
    conv_standard = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
    output_standard = conv_standard(X)
    
    # Count parameters
    params_standard = sum(p.numel() for p in conv_standard.parameters())
    
    print(f"\nStandard Convolution:")
    print(f"  Output shape: {output_standard.shape}")
    print(f"  Parameters: {params_standard:,}")
    
    # 2. Depthwise Separable Convolution
    class DepthwiseSeparableConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding=0):
            super().__init__()
            
            # Depthwise convolution
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size, 
                padding=padding, groups=in_channels
            )
            
            # Pointwise convolution
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x
    
    conv_depthwise = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, padding=1)
    output_depthwise = conv_depthwise(X)
    
    # Count parameters
    params_depthwise = sum(p.numel() for p in conv_depthwise.parameters())
    
    print(f"\nDepthwise Separable Convolution:")
    print(f"  Output shape: {output_depthwise.shape}")
    print(f"  Parameters: {params_depthwise:,}")
    print(f"  Parameter reduction: {params_standard / params_depthwise:.2f}x")
    
    # 3. Computational complexity analysis
    def compute_flops(input_shape, output_shape, kernel_size, in_channels, out_channels, conv_type='standard'):
        """Compute FLOPs for different convolution types"""
        
        batch_size, _, out_h, out_w = output_shape
        
        if conv_type == 'standard':
            # Standard convolution: output_elements * kernel_area * in_channels
            flops = batch_size * out_h * out_w * kernel_size * kernel_size * in_channels * out_channels
        
        elif conv_type == 'depthwise_separable':
            # Depthwise: output_elements * kernel_area * in_channels
            depthwise_flops = batch_size * out_h * out_w * kernel_size * kernel_size * in_channels
            
            # Pointwise: output_elements * in_channels * out_channels
            pointwise_flops = batch_size * out_h * out_w * in_channels * out_channels
            
            flops = depthwise_flops + pointwise_flops
        
        return flops
    
    flops_standard = compute_flops(X.shape, output_standard.shape, kernel_size, in_channels, out_channels, 'standard')
    flops_depthwise = compute_flops(X.shape, output_depthwise.shape, kernel_size, in_channels, out_channels, 'depthwise_separable')
    
    print(f"\nComputational Complexity:")
    print(f"  Standard convolution FLOPs: {flops_standard:,}")
    print(f"  Depthwise separable FLOPs: {flops_depthwise:,}")
    print(f"  FLOP reduction: {flops_standard / flops_depthwise:.2f}x")
    
    return conv_standard, conv_depthwise, params_standard, params_depthwise

conv_std, conv_dw, params_std, params_dw = depthwise_pointwise_convolutions()
```

**4.2 Dilated Convolutions and Receptive Fields:**
```python
def dilated_convolutions():
    """Demonstrate dilated convolutions and receptive field analysis"""
    
    print("\nDilated Convolutions and Receptive Fields:")
    print("-" * 40)
    
    # Input parameters
    batch_size, channels, height, width = 1, 1, 32, 32
    X = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {X.shape}")
    
    # Different dilation rates
    dilation_rates = [1, 2, 4, 8]
    kernel_size = 3
    
    convolutions = {}
    
    for dilation in dilation_rates:
        # Create dilated convolution
        conv = nn.Conv2d(channels, channels, kernel_size, padding=dilation, dilation=dilation)
        output = conv(X)
        
        convolutions[dilation] = {
            'conv': conv,
            'output': output,
            'output_shape': output.shape
        }
        
        print(f"\nDilation rate {dilation}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Padding used: {dilation}")
    
    # Receptive field calculation
    def calculate_receptive_field(layers):
        """Calculate effective receptive field through network layers"""
        
        rf = 1  # Initial receptive field
        stride_product = 1
        
        for layer in layers:
            kernel_size, stride, dilation = layer
            
            # Update receptive field
            rf = rf + (kernel_size - 1) * dilation * stride_product
            stride_product *= stride
        
        return rf
    
    print(f"\nReceptive Field Analysis:")
    
    # Single layer receptive fields
    for dilation in dilation_rates:
        layers = [(kernel_size, 1, dilation)]
        rf = calculate_receptive_field(layers)
        
        print(f"  Dilation {dilation}: Receptive field = {rf}x{rf}")
    
    # Multi-layer receptive field example
    print(f"\nMulti-layer Examples:")
    
    # Stack of dilated convolutions
    dilated_stack = [(3, 1, 1), (3, 1, 2), (3, 1, 4), (3, 1, 8)]
    rf_dilated = calculate_receptive_field(dilated_stack)
    
    # Standard convolutions with pooling
    standard_stack = [(3, 1, 1), (2, 2, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1)]
    rf_standard = calculate_receptive_field(standard_stack)
    
    print(f"  Dilated stack (1,2,4,8): RF = {rf_dilated}x{rf_dilated}")
    print(f"  Standard + pooling: RF = {rf_standard}x{rf_standard}")
    
    # Parameter comparison
    def count_dilated_params(num_layers, in_channels, out_channels, kernel_size):
        """Count parameters in dilated convolution stack"""
        return num_layers * in_channels * out_channels * kernel_size * kernel_size
    
    params_dilated = count_dilated_params(4, channels, channels, kernel_size)
    params_standard = count_dilated_params(3, channels, channels, kernel_size)  # Excluding pooling layers
    
    print(f"\nParameter Comparison:")
    print(f"  Dilated stack: {params_dilated} parameters")
    print(f"  Standard stack: {params_standard} parameters")
    print(f"  RF/param ratio (dilated): {rf_dilated / params_dilated:.4f}")
    print(f"  RF/param ratio (standard): {rf_standard / params_standard:.4f}")
    
    return convolutions, rf_dilated, rf_standard

convs_dilated, rf_dilated, rf_standard = dilated_convolutions()
```

**5. Memory Optimization and Efficient Training:**

**5.1 Gradient Checkpointing:**
```python
def gradient_checkpointing():
    """Demonstrate gradient checkpointing for memory optimization"""
    
    print("\nGradient Checkpointing for Memory Optimization:")
    print("-" * 46)
    
    import torch.utils.checkpoint as checkpoint
    
    # Deep network for demonstration
    class DeepNetwork(nn.Module):
        def __init__(self, num_layers=20, hidden_size=1024):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
            ])
            self.relu = nn.ReLU()
            
        def forward(self, x, use_checkpointing=False):
            if use_checkpointing:
                # Use gradient checkpointing
                for layer in self.layers:
                    x = checkpoint.checkpoint(self._forward_layer, x, layer)
            else:
                # Normal forward pass
                for layer in self.layers:
                    x = self._forward_layer(x, layer)
            
            return x
        
        def _forward_layer(self, x, layer):
            return self.relu(layer(x))
    
    # Create model and data
    hidden_size = 1024
    batch_size = 32
    num_layers = 20
    
    model = DeepNetwork(num_layers, hidden_size)
    X = torch.randn(batch_size, hidden_size, requires_grad=True)
    target = torch.randn(batch_size, hidden_size)
    
    print(f"Model depth: {num_layers} layers")
    print(f"Hidden size: {hidden_size}")
    print(f"Batch size: {batch_size}")
    
    # Function to measure memory usage
    def measure_memory_usage(model, x, target, use_checkpointing=False):
        """Measure peak memory usage during forward/backward pass"""
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Forward pass
        output = model(x, use_checkpointing=use_checkpointing)
        loss = F.mse_loss(output, target)
        
        # Measure memory before backward pass
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = 0
        
        # Backward pass
        loss.backward()
        
        # Measure peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory = 0
        
        return peak_memory, loss.item()
    
    # Compare memory usage
    if torch.cuda.is_available():
        model = model.cuda()
        X = X.cuda()
        target = target.cuda()
        
        print(f"\nMemory Usage Comparison (GPU):")
        
        # Without checkpointing
        model.zero_grad()
        memory_normal, loss_normal = measure_memory_usage(model, X, target, False)
        
        # With checkpointing
        model.zero_grad()
        memory_checkpoint, loss_checkpoint = measure_memory_usage(model, X, target, True)
        
        print(f"  Normal training: {memory_normal / 1024**2:.1f} MB")
        print(f"  With checkpointing: {memory_checkpoint / 1024**2:.1f} MB")
        print(f"  Memory savings: {memory_normal / memory_checkpoint:.2f}x")
        print(f"  Loss difference: {abs(loss_normal - loss_checkpoint):.2e}")
        
    else:
        print("  GPU not available for memory measurement")
    
    # Time comparison
    def time_training(model, x, target, use_checkpointing=False, num_runs=5):
        """Time the training process"""
        
        times = []
        
        for _ in range(num_runs):
            model.zero_grad()
            
            start_time = time.time()
            
            output = model(x, use_checkpointing=use_checkpointing)
            loss = F.mse_loss(output, target)
            loss.backward()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return np.mean(times), np.std(times)
    
    print(f"\nTiming Comparison:")
    
    time_normal, std_normal = time_training(model, X, target, False)
    time_checkpoint, std_checkpoint = time_training(model, X, target, True)
    
    print(f"  Normal training: {time_normal:.4f} ± {std_normal:.4f} seconds")
    print(f"  With checkpointing: {time_checkpoint:.4f} ± {std_checkpoint:.4f} seconds")
    print(f"  Time overhead: {time_checkpoint / time_normal:.2f}x")
    
    return model, time_normal, time_checkpoint

model_checkpoint, time_normal, time_checkpoint = gradient_checkpointing()
```

**6. Advanced CNN Architectures and Linear Algebra:**

```python
def advanced_cnn_architectures():
    """Demonstrate advanced CNN architectures using linear algebra"""
    
    print("\nAdvanced CNN Architectures:")
    print("-" * 28)
    
    # 1. Residual Connections (ResNet-style)
    class ResidualBlock(nn.Module):
        def __init__(self, channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)
            
            # Skip connection
            if stride != 1:
                self.skip = nn.Conv2d(channels, channels, 1, stride=stride)
            else:
                self.skip = nn.Identity()
        
        def forward(self, x):
            identity = self.skip(x)
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            out = self.conv2(out)
            out = self.bn2(out)
            
            # Residual connection
            out += identity
            out = self.relu(out)
            
            return out
    
    # 2. Attention Mechanism
    class SpatialAttention(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.query = nn.Conv2d(channels, channels // 8, 1)
            self.key = nn.Conv2d(channels, channels // 8, 1)
            self.value = nn.Conv2d(channels, channels, 1)
            self.softmax = nn.Softmax(dim=-1)
            self.gamma = nn.Parameter(torch.zeros(1))
        
        def forward(self, x):
            batch_size, channels, height, width = x.shape
            
            # Generate query, key, value
            q = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
            k = self.key(x).view(batch_size, -1, width * height)
            v = self.value(x).view(batch_size, -1, width * height)
            
            # Attention computation
            attention = torch.bmm(q, k)
            attention = self.softmax(attention)
            
            # Apply attention to values
            out = torch.bmm(v, attention.permute(0, 2, 1))
            out = out.view(batch_size, channels, height, width)
            
            # Residual connection with learnable weight
            out = self.gamma * out + x
            
            return out
    
    # 3. Complete architecture combining techniques
    class AdvancedCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            
            # Initial convolution
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)
            
            # Residual blocks
            self.res_block1 = ResidualBlock(64)
            self.res_block2 = ResidualBlock(64)
            
            # Attention module
            self.attention = SpatialAttention(64)
            
            # Classification head
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(64, num_classes)
        
        def forward(self, x):
            # Initial processing
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)
            
            # Residual processing
            x = self.res_block1(x)
            x = self.res_block2(x)
            
            # Attention
            x = self.attention(x)
            
            # Classification
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            
            return x
    
    # Test the architecture
    model = AdvancedCNN(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass with intermediate analysis
    with torch.no_grad():
        # Initial conv
        x1 = model.conv1(x)
        x1 = model.bn1(x1)
        x1 = model.relu(x1)
        x1 = model.pool(x1)
        
        print(f"After initial processing: {x1.shape}")
        
        # Residual blocks
        x2 = model.res_block1(x1)
        print(f"After residual block 1: {x2.shape}")
        
        x3 = model.res_block2(x2)
        print(f"After residual block 2: {x3.shape}")
        
        # Attention
        x4 = model.attention(x3)
        print(f"After attention: {x4.shape}")
        
        # Final classification
        x5 = model.global_pool(x4)
        x5 = x5.view(x5.size(0), -1)
        output = model.classifier(x5)
        
        print(f"Final output: {output.shape}")
    
    # Parameter analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Memory and computation analysis
    def analyze_model_complexity(model, input_shape):
        """Analyze model computational complexity"""
        
        model.eval()
        input_tensor = torch.randn(1, *input_shape[1:])
        
        # Count FLOPs (simplified estimation)
        total_flops = 0
        
        def flop_count_hook(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Conv2d):
                # Convolution FLOPs
                batch_size, out_channels, out_h, out_w = output.shape
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                total_flops += batch_size * out_channels * out_h * out_w * kernel_flops
            
            elif isinstance(module, nn.Linear):
                # Linear layer FLOPs
                total_flops += input[0].numel() * module.out_features
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(flop_count_hook))
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return total_flops
    
    flops = analyze_model_complexity(model, x.shape)
    print(f"  Estimated FLOPs: {flops:,}")
    print(f"  FLOPs per parameter: {flops / total_params:.2f}")
    
    return model, total_params, flops

advanced_model, total_params, flops = advanced_cnn_architectures()
```

**Summary and Key Insights:**

**Linear Algebra's Critical Role in Deep Learning and CNNs:**

1. **Computational Foundation**: Matrix operations enable efficient parallel processing of convolutions, activations, and gradients
2. **Memory Efficiency**: Techniques like gradient checkpointing and sparse representations optimize memory usage
3. **Optimization**: Matrix-based optimizers (Adam, RMSprop) leverage second-order information for faster convergence
4. **Architecture Innovation**: Advanced techniques like attention mechanisms and residual connections rely on linear algebraic principles
5. **Scalability**: Vectorized operations and batch processing enable training on large datasets and models

**Key Applications in Modern Deep Learning:**
- **Transformer Architectures**: Attention mechanisms as matrix operations
- **Computer Vision**: CNNs for object detection, segmentation, recognition
- **Efficient Networks**: Depthwise separable convolutions, dilated convolutions
- **Optimization**: Advanced gradient-based methods for training stability
- **Memory Management**: Gradient checkpointing, mixed precision training

The mathematical rigor of linear algebra provides both the theoretical foundation and computational efficiency necessary for modern deep learning systems.

---

## Question 7

**Propose strategies to visualizehigh-dimensional datausinglinear algebratechniques.**

**Answer:** _[To be filled]_

---

## Question 8

**Discuss an approach for optimizingmemory usageinmatrix computationsfor a large-scalemachine learning application.**

**Answer:** _[To be filled]_

---

