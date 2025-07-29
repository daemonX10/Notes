# Python Ml Interview Questions - Coding Questions

## Question 1

**Give an example of how to implement a gradient descent algorithm in Python.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional

class GradientDescent:
    """
    Complete implementation of gradient descent with various optimizations
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, optimizer: str = 'standard'):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.optimizer = optimizer
        self.cost_history = []
        self.gradient_history = []
        
        # For momentum
        self.velocity = None
        self.momentum = 0.9
        
        # For Adam optimizer
        self.m = None  # First moment
        self.v = None  # Second moment
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Time step
    
    def minimize(self, cost_function: Callable, gradient_function: Callable, 
                 initial_params: np.ndarray, X: np.ndarray = None, y: np.ndarray = None) -> Tuple[np.ndarray, List[float]]:
        """
        Minimize a cost function using gradient descent
        
        Args:
            cost_function: Function that computes cost given parameters
            gradient_function: Function that computes gradients
            initial_params: Starting parameters
            X: Input features (optional)
            y: Target values (optional)
            
        Returns:
            Tuple of (optimized_parameters, cost_history)
        """
        params = initial_params.copy()
        self.cost_history = []
        self.gradient_history = []
        
        # Initialize optimizer-specific variables
        if self.optimizer in ['momentum', 'adam']:
            self.velocity = np.zeros_like(params)
        
        if self.optimizer == 'adam':
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.t = 0
        
        for iteration in range(self.max_iterations):
            # Compute cost and gradient
            if X is not None and y is not None:
                cost = cost_function(params, X, y)
                gradient = gradient_function(params, X, y)
            else:
                cost = cost_function(params)
                gradient = gradient_function(params)
            
            # Store history
            self.cost_history.append(cost)
            self.gradient_history.append(np.linalg.norm(gradient))
            
            # Check convergence
            if np.linalg.norm(gradient) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            # Update parameters based on optimizer
            params = self._update_parameters(params, gradient)
            
            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}, ||Gradient|| = {np.linalg.norm(gradient):.6f}")
        
        return params, self.cost_history
    
    def _update_parameters(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Update parameters based on selected optimizer"""
        
        if self.optimizer == 'standard':
            return params - self.learning_rate * gradient
        
        elif self.optimizer == 'momentum':
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            return params + self.velocity
        
        elif self.optimizer == 'adam':
            self.t += 1
            
            # Update biased first moment estimate
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            
            # Update biased second raw moment estimate
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v / (1 - self.beta2 ** self.t)
            
            # Update parameters
            return params - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
    
    def plot_convergence(self):
        """Plot cost and gradient norm over iterations"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot cost
        ax1.plot(self.cost_history)
        ax1.set_title('Cost Function Over Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.grid(True)
        
        # Plot gradient norm
        ax2.plot(self.gradient_history)
        ax2.set_title('Gradient Norm Over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('||Gradient||')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example 1: Minimize a simple quadratic function
def quadratic_cost(params):
    """Simple quadratic function: f(x, y) = x^2 + y^2"""
    return np.sum(params ** 2)

def quadratic_gradient(params):
    """Gradient of quadratic function"""
    return 2 * params

# Example 2: Linear Regression with Gradient Descent
class LinearRegressionGD:
    """Linear Regression using Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """Fit linear regression model"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute cost (MSE)
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (2 / n_samples) * X.T.dot(y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        return X.dot(self.weights) + self.bias
    
    def plot_cost(self):
        """Plot cost function"""
        plt.figure(figsize=(8, 5))
        plt.plot(self.cost_history)
        plt.title('Cost Function During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()

# Example 3: Logistic Regression with Gradient Descent
class LogisticRegressionGD:
    """Logistic Regression using Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Fit logistic regression model"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = X.dot(self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)
            
            # Compute cost (cross-entropy)
            cost = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        linear_pred = X.dot(self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict_classes(self, X):
        """Predict binary classes"""
        return (self.predict(X) > 0.5).astype(int)

# Demonstration Examples
def demonstrate_gradient_descent():
    """Demonstrate different gradient descent implementations"""
    
    print("=== Gradient Descent Demonstrations ===\n")
    
    # Example 1: Simple quadratic function
    print("1. Minimizing Quadratic Function f(x,y) = x² + y²")
    print("-" * 50)
    
    gd = GradientDescent(learning_rate=0.1, max_iterations=100, optimizer='standard')
    initial_params = np.array([5.0, -3.0])
    
    optimal_params, cost_history = gd.minimize(quadratic_cost, quadratic_gradient, initial_params)
    
    print(f"Initial parameters: {initial_params}")
    print(f"Optimal parameters: {optimal_params}")
    print(f"Final cost: {cost_history[-1]:.8f}")
    
    # Compare optimizers
    print("\n2. Comparing Different Optimizers")
    print("-" * 50)
    
    optimizers = ['standard', 'momentum', 'adam']
    results = {}
    
    for opt in optimizers:
        gd_opt = GradientDescent(learning_rate=0.01, max_iterations=200, optimizer=opt)
        params_opt, cost_hist = gd_opt.minimize(quadratic_cost, quadratic_gradient, 
                                                np.array([5.0, -3.0]))
        results[opt] = {
            'final_params': params_opt,
            'final_cost': cost_hist[-1],
            'iterations_to_converge': len(cost_hist)
        }
        print(f"{opt.capitalize()}: Final cost = {cost_hist[-1]:.8f}, "
              f"Iterations = {len(cost_hist)}")
    
    # Example 3: Linear Regression
    print("\n3. Linear Regression with Gradient Descent")
    print("-" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    true_weights = np.array([3.0, -2.0])
    true_bias = 1.0
    y = X.dot(true_weights) + true_bias + 0.1 * np.random.randn(n_samples)
    
    # Fit model
    lr_model = LinearRegressionGD(learning_rate=0.01, max_iterations=1000)
    lr_model.fit(X, y)
    
    print(f"True weights: {true_weights}")
    print(f"Learned weights: {lr_model.weights}")
    print(f"True bias: {true_bias}")
    print(f"Learned bias: {lr_model.bias:.4f}")
    print(f"Final cost: {lr_model.cost_history[-1]:.6f}")
    
    # Example 4: Logistic Regression
    print("\n4. Logistic Regression with Gradient Descent")
    print("-" * 50)
    
    # Generate binary classification data
    from sklearn.datasets import make_classification
    X_class, y_class = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                          n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Fit logistic regression
    log_reg = LogisticRegressionGD(learning_rate=0.1, max_iterations=1000)
    log_reg.fit(X_class, y_class)
    
    # Evaluate
    predictions = log_reg.predict_classes(X_class)
    accuracy = np.mean(predictions == y_class)
    
    print(f"Final cost: {log_reg.cost_history[-1]:.6f}")
    print(f"Training accuracy: {accuracy:.4f}")
    
    return {
        'quadratic_results': results,
        'linear_regression': lr_model,
        'logistic_regression': log_reg
    }

# Advanced Gradient Descent with Regularization
class RegularizedGradientDescent:
    """Gradient Descent with L1 and L2 regularization"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, 
                 l1_reg=0.0, l2_reg=0.0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """Fit model with regularization"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = X.dot(self.weights) + self.bias
            
            # Compute cost with regularization
            mse_cost = np.mean((y_pred - y) ** 2)
            l1_cost = self.l1_reg * np.sum(np.abs(self.weights))
            l2_cost = self.l2_reg * np.sum(self.weights ** 2)
            total_cost = mse_cost + l1_cost + l2_cost
            
            self.cost_history.append(total_cost)
            
            # Compute gradients with regularization
            dw_mse = (2 / n_samples) * X.T.dot(y_pred - y)
            dw_l1 = self.l1_reg * np.sign(self.weights)
            dw_l2 = 2 * self.l2_reg * self.weights
            dw = dw_mse + dw_l1 + dw_l2
            
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                print(f"Iteration {i}: Total Cost = {total_cost:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        return X.dot(self.weights) + self.bias

# Run demonstration
if __name__ == "__main__":
    results = demonstrate_gradient_descent()
    print("\n=== Demonstration Complete ===")
```

This implementation provides:

1. **Multiple Optimizers**: Standard, Momentum, and Adam
2. **Convergence Monitoring**: Cost and gradient tracking
3. **Real Applications**: Linear and Logistic Regression
4. **Regularization**: L1 and L2 penalties
5. **Visualization**: Cost function plotting
6. **Type Hints**: Professional code structure
7. **Error Handling**: Numerical stability measures

---

## Question 2

**Write a Python function that normalizes an array of data to the range [0, 1].**

**Answer:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
import warnings

class DataNormalizer:
    """
    Comprehensive data normalization toolkit with multiple methods
    """
    
    def __init__(self):
        self.fitted_params = {}
        self.normalization_history = []
    
    def min_max_normalize(self, data: np.ndarray, feature_range: Tuple[float, float] = (0, 1), 
                         axis: Optional[int] = None, keep_params: bool = True) -> np.ndarray:
        """
        Min-Max normalization to scale data to a specific range
        
        Formula: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
        
        Args:
            data: Input data array
            feature_range: Desired range (min, max)
            axis: Axis along which to normalize (None for global, 0 for columns, 1 for rows)
            keep_params: Whether to store parameters for inverse transformation
            
        Returns:
            Normalized data
        """
        data = np.asarray(data)
        
        # Calculate min and max along specified axis
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        
        # Handle case where min == max (constant values)
        data_range = data_max - data_min
        data_range = np.where(data_range == 0, 1, data_range)  # Avoid division by zero
        
        # Normalize to [0, 1]
        normalized = (data - data_min) / data_range
        
        # Scale to desired range
        target_min, target_max = feature_range
        scaled = normalized * (target_max - target_min) + target_min
        
        # Store parameters for inverse transformation
        if keep_params:
            self.fitted_params['min_max'] = {
                'data_min': data_min,
                'data_max': data_max,
                'feature_range': feature_range,
                'axis': axis
            }
            
        return scaled
    
    def z_score_normalize(self, data: np.ndarray, axis: Optional[int] = None, 
                         keep_params: bool = True) -> np.ndarray:
        """
        Z-score normalization (standardization)
        
        Formula: X_scaled = (X - μ) / σ
        
        Args:
            data: Input data array
            axis: Axis along which to normalize
            keep_params: Whether to store parameters
            
        Returns:
            Standardized data
        """
        data = np.asarray(data)
        
        # Calculate mean and standard deviation
        data_mean = np.mean(data, axis=axis, keepdims=True)
        data_std = np.std(data, axis=axis, keepdims=True)
        
        # Handle case where std == 0
        data_std = np.where(data_std == 0, 1, data_std)
        
        # Standardize
        standardized = (data - data_mean) / data_std
        
        # Store parameters
        if keep_params:
            self.fitted_params['z_score'] = {
                'data_mean': data_mean,
                'data_std': data_std,
                'axis': axis
            }
            
        return standardized
    
    def robust_normalize(self, data: np.ndarray, axis: Optional[int] = None, 
                        keep_params: bool = True) -> np.ndarray:
        """
        Robust normalization using median and IQR (less sensitive to outliers)
        
        Formula: X_scaled = (X - median) / IQR
        
        Args:
            data: Input data array
            axis: Axis along which to normalize
            keep_params: Whether to store parameters
            
        Returns:
            Robust normalized data
        """
        data = np.asarray(data)
        
        # Calculate median and IQR
        data_median = np.median(data, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        
        # Handle case where IQR == 0
        iqr = np.where(iqr == 0, 1, iqr)
        
        # Normalize
        normalized = (data - data_median) / iqr
        
        # Store parameters
        if keep_params:
            self.fitted_params['robust'] = {
                'data_median': data_median,
                'iqr': iqr,
                'axis': axis
            }
            
        return normalized
    
    def unit_vector_normalize(self, data: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Unit vector normalization (L2 normalization)
        
        Formula: X_scaled = X / ||X||_2
        
        Args:
            data: Input data array
            axis: Axis along which to normalize (1 for row-wise, 0 for column-wise)
            
        Returns:
            Unit normalized data
        """
        data = np.asarray(data)
        
        # Calculate L2 norm
        l2_norm = np.linalg.norm(data, axis=axis, keepdims=True)
        
        # Handle zero vectors
        l2_norm = np.where(l2_norm == 0, 1, l2_norm)
        
        # Normalize
        normalized = data / l2_norm
        
        return normalized
    
    def quantile_normalize(self, data: np.ndarray, n_quantiles: int = 100, 
                          axis: Optional[int] = None) -> np.ndarray:
        """
        Quantile normalization to uniform distribution
        
        Args:
            data: Input data array
            n_quantiles: Number of quantiles to use
            axis: Axis along which to normalize
            
        Returns:
            Quantile normalized data
        """
        data = np.asarray(data)
        
        if axis is None:
            # Flatten data for global quantile normalization
            flat_data = data.flatten()
            quantiles = np.linspace(0, 100, n_quantiles)
            quantile_values = np.percentile(flat_data, quantiles)
            
            # Map each value to its quantile
            normalized = np.interp(data, quantile_values, quantiles / 100)
        else:
            # Apply along specified axis
            normalized = np.apply_along_axis(
                lambda x: np.interp(x, np.percentile(x, np.linspace(0, 100, n_quantiles)), 
                                   np.linspace(0, 1, n_quantiles)), 
                axis, data
            )
        
        return normalized
    
    def inverse_transform(self, normalized_data: np.ndarray, method: str) -> np.ndarray:
        """
        Inverse transformation to original scale
        
        Args:
            normalized_data: Normalized data to transform back
            method: Normalization method used ('min_max', 'z_score', 'robust')
            
        Returns:
            Data in original scale
        """
        if method not in self.fitted_params:
            raise ValueError(f"No fitted parameters found for method '{method}'")
        
        params = self.fitted_params[method]
        
        if method == 'min_max':
            target_min, target_max = params['feature_range']
            # Reverse scaling to [0, 1]
            unit_scaled = (normalized_data - target_min) / (target_max - target_min)
            # Reverse normalization
            original = unit_scaled * (params['data_max'] - params['data_min']) + params['data_min']
            
        elif method == 'z_score':
            original = normalized_data * params['data_std'] + params['data_mean']
            
        elif method == 'robust':
            original = normalized_data * params['iqr'] + params['data_median']
            
        else:
            raise ValueError(f"Inverse transformation not supported for method '{method}'")
        
        return original
    
    def compare_normalizations(self, data: np.ndarray, methods: list = None) -> dict:
        """
        Compare different normalization methods on the same data
        
        Args:
            data: Input data
            methods: List of methods to compare
            
        Returns:
            Dictionary with normalized data for each method
        """
        if methods is None:
            methods = ['min_max', 'z_score', 'robust', 'unit_vector', 'quantile']
        
        results = {}
        
        for method in methods:
            if method == 'min_max':
                results[method] = self.min_max_normalize(data)
            elif method == 'z_score':
                results[method] = self.z_score_normalize(data)
            elif method == 'robust':
                results[method] = self.robust_normalize(data)
            elif method == 'unit_vector':
                results[method] = self.unit_vector_normalize(data)
            elif method == 'quantile':
                results[method] = self.quantile_normalize(data)
        
        return results

# Simple functions for quick use
def normalize_to_01(data: Union[list, np.ndarray], method: str = 'min_max') -> np.ndarray:
    """
    Simple function to normalize data to [0, 1] range
    
    Args:
        data: Input data (list or numpy array)
        method: Normalization method ('min_max', 'quantile')
        
    Returns:
        Normalized data in [0, 1] range
    """
    data = np.asarray(data)
    
    if method == 'min_max':
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max == data_min:
            return np.zeros_like(data)
        
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'quantile':
        # Use quantile normalization
        flat_data = data.flatten()
        sorted_indices = np.argsort(flat_data)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(flat_data))
        
        normalized = ranks / (len(flat_data) - 1)
        return normalized.reshape(data.shape)
    
    else:
        raise ValueError(f"Unsupported method: {method}")

def normalize_dataframe(df: pd.DataFrame, columns: list = None, 
                       method: str = 'min_max', **kwargs) -> pd.DataFrame:
    """
    Normalize columns in a pandas DataFrame
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize (None for all numeric columns)
        method: Normalization method
        **kwargs: Additional arguments for normalization
        
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        # Auto-select numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    normalizer = DataNormalizer()
    
    for col in columns:
        if col in df.columns:
            if method == 'min_max':
                df_normalized[col] = normalizer.min_max_normalize(df[col].values, **kwargs)
            elif method == 'z_score':
                df_normalized[col] = normalizer.z_score_normalize(df[col].values, **kwargs)
            elif method == 'robust':
                df_normalized[col] = normalizer.robust_normalize(df[col].values, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
    
    return df_normalized

# Demonstration and examples
def demonstrate_normalization():
    """Demonstrate different normalization techniques"""
    
    print("=== Data Normalization Demonstrations ===\n")
    
    # Generate sample data with different characteristics
    np.random.seed(42)
    
    # Normal distribution data
    normal_data = np.random.normal(50, 15, 100)
    
    # Data with outliers
    outlier_data = np.concatenate([
        np.random.normal(10, 2, 90),
        np.array([50, 55, 60])  # Outliers
    ])
    
    # Skewed data
    skewed_data = np.random.exponential(2, 100)
    
    # Create normalizer
    normalizer = DataNormalizer()
    
    print("1. Basic Min-Max Normalization to [0, 1]")
    print("-" * 50)
    
    normalized_01 = normalize_to_01(normal_data)
    print(f"Original range: [{np.min(normal_data):.2f}, {np.max(normal_data):.2f}]")
    print(f"Normalized range: [{np.min(normalized_01):.2f}, {np.max(normalized_01):.2f}]")
    
    print("\n2. Different Normalization Methods Comparison")
    print("-" * 50)
    
    methods = ['min_max', 'z_score', 'robust']
    results = normalizer.compare_normalizations(normal_data, methods)
    
    for method, normalized in results.items():
        print(f"{method.replace('_', ' ').title()}:")
        print(f"  Range: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
        print(f"  Mean: {np.mean(normalized):.3f}, Std: {np.std(normalized):.3f}")
    
    print("\n3. Handling Data with Outliers")
    print("-" * 50)
    
    # Compare robust vs min-max on outlier data
    minmax_outliers = normalizer.min_max_normalize(outlier_data)
    robust_outliers = normalizer.robust_normalize(outlier_data)
    
    print(f"Original data range: [{np.min(outlier_data):.2f}, {np.max(outlier_data):.2f}]")
    print(f"Min-max normalized: [{np.min(minmax_outliers):.3f}, {np.max(minmax_outliers):.3f}]")
    print(f"Robust normalized: [{np.min(robust_outliers):.3f}, {np.max(robust_outliers):.3f}]")
    
    # Show how outliers affect the distribution
    main_data_minmax = minmax_outliers[:-3]  # Exclude outliers
    main_data_robust = robust_outliers[:-3]
    
    print(f"Main data (90%) - Min-max: [{np.min(main_data_minmax):.3f}, {np.max(main_data_minmax):.3f}]")
    print(f"Main data (90%) - Robust: [{np.min(main_data_robust):.3f}, {np.max(main_data_robust):.3f}]")
    
    print("\n4. DataFrame Normalization")
    print("-" * 50)
    
    # Create sample DataFrame
    df_data = pd.DataFrame({
        'feature1': np.random.normal(100, 20, 50),
        'feature2': np.random.uniform(0, 1000, 50),
        'feature3': np.random.exponential(5, 50),
        'category': np.random.choice(['A', 'B', 'C'], 50)
    })
    
    print("Original DataFrame statistics:")
    print(df_data.describe())
    
    # Normalize numeric columns
    df_normalized = normalize_dataframe(df_data, method='min_max')
    
    print("\nNormalized DataFrame statistics:")
    print(df_normalized.describe())
    
    print("\n5. Inverse Transformation")
    print("-" * 50)
    
    # Demonstrate inverse transformation
    original_sample = np.array([1, 5, 10, 15, 20])
    normalized_sample = normalizer.min_max_normalize(original_sample, keep_params=True)
    reconstructed_sample = normalizer.inverse_transform(normalized_sample, 'min_max')
    
    print(f"Original: {original_sample}")
    print(f"Normalized: {normalized_sample}")
    print(f"Reconstructed: {reconstructed_sample}")
    print(f"Reconstruction error: {np.mean(np.abs(original_sample - reconstructed_sample)):.10f}")
    
    return {
        'normalizer': normalizer,
        'normal_data': normal_data,
        'outlier_data': outlier_data,
        'skewed_data': skewed_data,
        'results': results
    }

# Advanced normalization for machine learning pipelines
class MLNormalizer:
    """
    Normalization specifically designed for ML pipelines
    """
    
    def __init__(self, method='min_max', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.fitted = False
        self.feature_params = {}
    
    def fit(self, X, y=None):
        """Fit normalizer on training data"""
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.feature_params = {}
        
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            
            if self.method == 'min_max':
                self.feature_params[i] = {
                    'min': np.min(feature_data),
                    'max': np.max(feature_data),
                    'range': np.max(feature_data) - np.min(feature_data)
                }
            elif self.method == 'z_score':
                self.feature_params[i] = {
                    'mean': np.mean(feature_data),
                    'std': np.std(feature_data)
                }
            elif self.method == 'robust':
                self.feature_params[i] = {
                    'median': np.median(feature_data),
                    'iqr': np.percentile(feature_data, 75) - np.percentile(feature_data, 25)
                }
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_normalized = X.copy().astype(float)
        
        for i in range(X.shape[1]):
            params = self.feature_params[i]
            
            if self.method == 'min_max':
                if params['range'] != 0:
                    X_normalized[:, i] = (X[:, i] - params['min']) / params['range']
                else:
                    X_normalized[:, i] = 0
                    
            elif self.method == 'z_score':
                if params['std'] != 0:
                    X_normalized[:, i] = (X[:, i] - params['mean']) / params['std']
                else:
                    X_normalized[:, i] = 0
                    
            elif self.method == 'robust':
                if params['iqr'] != 0:
                    X_normalized[:, i] = (X[:, i] - params['median']) / params['iqr']
                else:
                    X_normalized[:, i] = 0
        
        return X_normalized
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X_normalized):
        """Transform back to original scale"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")
        
        X_normalized = np.asarray(X_normalized)
        if X_normalized.ndim == 1:
            X_normalized = X_normalized.reshape(-1, 1)
        
        X_original = X_normalized.copy()
        
        for i in range(X_normalized.shape[1]):
            params = self.feature_params[i]
            
            if self.method == 'min_max':
                X_original[:, i] = X_normalized[:, i] * params['range'] + params['min']
                
            elif self.method == 'z_score':
                X_original[:, i] = X_normalized[:, i] * params['std'] + params['mean']
                
            elif self.method == 'robust':
                X_original[:, i] = X_normalized[:, i] * params['iqr'] + params['median']
        
        return X_original

# Example usage in ML pipeline
def ml_pipeline_example():
    """Example of using normalization in ML pipeline"""
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("=== ML Pipeline Normalization Example ===")
    print(f"Original X_train range: {np.min(X_train):.2f} to {np.max(X_train):.2f}")
    print(f"Original X_test range: {np.min(X_test):.2f} to {np.max(X_test):.2f}")
    
    # Fit normalizer on training data only
    normalizer = MLNormalizer(method='min_max')
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)  # Use same parameters
    
    print(f"Normalized X_train range: {np.min(X_train_norm):.2f} to {np.max(X_train_norm):.2f}")
    print(f"Normalized X_test range: {np.min(X_test_norm):.2f} to {np.max(X_test_norm):.2f}")
    
    # Verify inverse transformation
    X_train_recovered = normalizer.inverse_transform(X_train_norm)
    error = np.mean(np.abs(X_train - X_train_recovered))
    print(f"Reconstruction error: {error:.10f}")

# Run demonstrations
if __name__ == "__main__":
    demo_results = demonstrate_normalization()
    print("\n" + "="*60)
    ml_pipeline_example()
```

This comprehensive implementation provides:

1. **Multiple normalization methods**: Min-Max, Z-score, Robust, Unit Vector, Quantile
2. **Flexible axis handling**: Global, row-wise, or column-wise normalization
3. **Inverse transformations**: Convert back to original scale
4. **DataFrame support**: Direct pandas integration
5. **ML pipeline integration**: Fit/transform pattern for proper train/test splits
6. **Outlier handling**: Robust methods for data with outliers
7. **Error handling**: Safe division and edge case management
8. **Comprehensive examples**: Multiple use cases and demonstrations

---

## Question 3

**Construct a Python class structure for a simple perceptron model.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Perceptron:
    """
    Simple Perceptron implementation for binary classification
    
    The perceptron is the simplest type of artificial neural network,
    consisting of a single neuron that makes predictions based on a linear
    combination of input features.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 random_state: Optional[int] = None, fit_intercept: bool = True):
        """
        Initialize perceptron
        
        Args:
            learning_rate: Step size for weight updates
            max_iterations: Maximum number of training iterations
            random_state: Random seed for reproducibility
            fit_intercept: Whether to add bias term
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        
        # Model parameters
        self.weights = None
        self.bias = None
        
        # Training history
        self.training_history = {
            'iteration': [],
            'errors': [],
            'accuracy': [],
            'weights': [],
            'bias': []
        }
        
        self.is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add bias column to feature matrix"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _initialize_weights(self, n_features: int) -> None:
        """Initialize weights randomly"""
        # Small random weights for better convergence
        self.weights = np.random.normal(0, 0.01, n_features)
        
        if self.fit_intercept:
            self.bias = np.random.normal(0, 0.01)
        else:
            self.bias = 0
    
    def _activation(self, z: np.ndarray) -> np.ndarray:
        """
        Step activation function (Heaviside function)
        
        Args:
            z: Linear combination of inputs
            
        Returns:
            Binary predictions (0 or 1)
        """
        return np.where(z >= 0, 1, 0)
    
    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input (linear combination)"""
        return np.dot(X, self.weights) + self.bias
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'Perceptron':
        """
        Train the perceptron using the perceptron learning rule
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (0 or 1)
            verbose: Whether to print training progress
            
        Returns:
            self: Fitted perceptron
        """
        # Validate inputs
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Ensure binary labels
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(f"Expected 2 classes, got {len(unique_labels)}")
        
        # Convert labels to 0/1 if needed
        if set(unique_labels) != {0, 1}:
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            y = np.array([label_map[label] for label in y])
            self.label_map = label_map
            self.inverse_label_map = {v: k for k, v in label_map.items()}
        else:
            self.label_map = None
            self.inverse_label_map = None
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        self._initialize_weights(n_features)
        
        # Training loop
        for iteration in range(self.max_iterations):
            errors = 0
            iteration_weights = []
            iteration_bias = []
            
            # Process each sample
            for i in range(n_samples):
                # Forward pass
                xi = X[i]
                yi = y[i]
                
                # Calculate net input and prediction
                net_input = self._net_input(xi)
                prediction = self._activation(net_input)
                
                # Calculate error
                error = yi - prediction
                
                # Update weights if there's an error (perceptron learning rule)
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    if self.fit_intercept:
                        self.bias += self.learning_rate * error
                    errors += 1
                
                # Store weights for this sample update
                iteration_weights.append(self.weights.copy())
                iteration_bias.append(self.bias)
            
            # Calculate accuracy for this iteration
            predictions = self.predict(X)
            accuracy = accuracy_score(y, predictions)
            
            # Store training history
            self.training_history['iteration'].append(iteration + 1)
            self.training_history['errors'].append(errors)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['weights'].append(self.weights.copy())
            self.training_history['bias'].append(self.bias)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: {errors} errors, accuracy: {accuracy:.4f}")
            
            # Early stopping if no errors
            if errors == 0:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Perceptron must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Calculate predictions
        net_input = self._net_input(X)
        predictions = self._activation(net_input)
        
        # Convert back to original labels if needed
        if self.inverse_label_map is not None:
            predictions = np.array([self.inverse_label_map[pred] for pred in predictions])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (simplified for perceptron)
        
        Note: Standard perceptron doesn't output probabilities,
        this is a simplified approximation based on distance from decision boundary
        """
        if not self.is_fitted:
            raise ValueError("Perceptron must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Calculate net input (distance from decision boundary)
        net_input = self._net_input(X)
        
        # Simple approximation: use sigmoid to convert to probabilities
        probabilities = 1 / (1 + np.exp(-net_input))
        
        # Return probabilities for both classes
        proba_matrix = np.column_stack([1 - probabilities, probabilities])
        return proba_matrix
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get the decision function values (net input)
        
        Args:
            X: Input features
            
        Returns:
            Decision function values
        """
        if not self.is_fitted:
            raise ValueError("Perceptron must be fitted before computing decision function")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return self._net_input(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """Plot training history"""
        if not self.is_fitted:
            raise ValueError("Perceptron must be fitted before plotting history")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot errors
        axes[0].plot(self.training_history['iteration'], self.training_history['errors'], 'b-')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Number of Errors')
        axes[0].set_title('Training Errors')
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.training_history['iteration'], self.training_history['accuracy'], 'g-')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].grid(True)
        
        # Plot weight evolution (if 2D)
        if len(self.training_history['weights'][0]) == 2:
            weights_array = np.array(self.training_history['weights'])
            axes[2].plot(weights_array[:, 0], label='Weight 1')
            axes[2].plot(weights_array[:, 1], label='Weight 2')
            if self.fit_intercept:
                axes[2].plot(self.training_history['bias'], label='Bias')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Weight Value')
            axes[2].set_title('Weight Evolution')
            axes[2].legend()
            axes[2].grid(True)
        else:
            axes[2].text(0.5, 0.5, 'Weight evolution\n(>2D features)', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Weight Evolution')
        
        plt.tight_layout()
        plt.show()

class PerceptronVisualizer:
    """Visualization utilities for perceptron"""
    
    @staticmethod
    def plot_decision_boundary(perceptron: Perceptron, X: np.ndarray, y: np.ndarray,
                             title: str = "Perceptron Decision Boundary",
                             figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot decision boundary for 2D data
        
        Args:
            perceptron: Fitted perceptron
            X: Feature data (must be 2D)
            y: Labels
            title: Plot title
            figsize: Figure size
        """
        if X.shape[1] != 2:
            raise ValueError("Can only plot decision boundary for 2D data")
        
        if not perceptron.is_fitted:
            raise ValueError("Perceptron must be fitted")
        
        plt.figure(figsize=figsize)
        
        # Create a mesh for plotting decision boundary
        h = 0.02  # Step size
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = perceptron.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['lightcoral', 'lightblue'])
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
        plt.colorbar(scatter)
        
        # Add weight vector if available
        if perceptron.weights is not None and len(perceptron.weights) == 2:
            # Plot normal vector to decision boundary
            plt.arrow(np.mean(X[:, 0]), np.mean(X[:, 1]),
                     perceptron.weights[0], perceptron.weights[1],
                     head_width=0.1, head_length=0.1, fc='red', ec='red',
                     label=f'Weight vector: ({perceptron.weights[0]:.2f}, {perceptron.weights[1]:.2f})')
            plt.legend()
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def compare_perceptrons(perceptrons: List[Perceptron], names: List[str],
                          X: np.ndarray, y: np.ndarray) -> None:
        """Compare multiple perceptrons"""
        n_perceptrons = len(perceptrons)
        fig, axes = plt.subplots(1, n_perceptrons, figsize=(5*n_perceptrons, 4))
        
        if n_perceptrons == 1:
            axes = [axes]
        
        for i, (perceptron, name) in enumerate(zip(perceptrons, names)):
            if X.shape[1] == 2:
                # Plot decision boundary
                h = 0.02
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                    np.arange(y_min, y_max, h))
                
                mesh_points = np.c_[xx.ravel(), yy.ravel()]
                Z = perceptron.predict(mesh_points)
                Z = Z.reshape(xx.shape)
                
                axes[i].contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['lightcoral', 'lightblue'])
                axes[i].contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
                axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
                
                # Calculate and display accuracy
                accuracy = perceptron.score(X, y)
                axes[i].set_title(f'{name}\nAccuracy: {accuracy:.3f}')
                axes[i].set_xlabel('Feature 1')
                axes[i].set_ylabel('Feature 2')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Enhanced perceptron variants
class AdaptivePerceptron(Perceptron):
    """Perceptron with adaptive learning rate"""
    
    def __init__(self, initial_learning_rate: float = 0.1, decay_rate: float = 0.99,
                 min_learning_rate: float = 0.001, **kwargs):
        super().__init__(learning_rate=initial_learning_rate, **kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.min_learning_rate = min_learning_rate
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'AdaptivePerceptron':
        """Fit with adaptive learning rate"""
        # Reset learning rate
        self.learning_rate = self.initial_learning_rate
        
        # Store learning rate history
        self.learning_rate_history = []
        
        return super().fit(X, y, verbose)
    
    def _update_learning_rate(self, iteration: int) -> None:
        """Update learning rate based on iteration"""
        self.learning_rate = max(
            self.min_learning_rate,
            self.initial_learning_rate * (self.decay_rate ** iteration)
        )
        self.learning_rate_history.append(self.learning_rate)

class AveragedPerceptron(Perceptron):
    """Averaged perceptron for better generalization"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avg_weights = None
        self.avg_bias = None
        self.weight_sum = None
        self.bias_sum = None
        self.update_count = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'AveragedPerceptron':
        """Fit averaged perceptron"""
        # Initialize averaging variables
        n_features = X.shape[1] if X.ndim > 1 else 1
        self.weight_sum = np.zeros(n_features)
        self.bias_sum = 0
        self.update_count = 0
        
        # Call parent fit method
        super().fit(X, y, verbose)
        
        # Compute averaged weights
        if self.update_count > 0:
            self.avg_weights = self.weight_sum / self.update_count
            self.avg_bias = self.bias_sum / self.update_count
        else:
            self.avg_weights = self.weights.copy()
            self.avg_bias = self.bias
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using averaged weights"""
        if not self.is_fitted:
            raise ValueError("Perceptron must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Use averaged weights for prediction
        net_input = np.dot(X, self.avg_weights) + self.avg_bias
        predictions = self._activation(net_input)
        
        if self.inverse_label_map is not None:
            predictions = np.array([self.inverse_label_map[pred] for pred in predictions])
        
        return predictions

# Demonstration and examples
def demonstrate_perceptron():
    """Comprehensive perceptron demonstration"""
    
    print("=== Perceptron Implementation Demonstrations ===\n")
    
    # 1. Simple linearly separable data
    print("1. Basic Perceptron on Linearly Separable Data")
    print("-" * 50)
    
    # Generate simple 2D linearly separable data
    np.random.seed(42)
    X_simple, y_simple = make_classification(
        n_samples=100, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=42
    )
    
    # Train perceptron
    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000, random_state=42)
    perceptron.fit(X_simple, y_simple, verbose=True)
    
    # Evaluate
    accuracy = perceptron.score(X_simple, y_simple)
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Final weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias:.4f}")
    
    # 2. Demonstrate different learning rates
    print("\n2. Effect of Different Learning Rates")
    print("-" * 50)
    
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    perceptrons = []
    
    for lr in learning_rates:
        p = Perceptron(learning_rate=lr, max_iterations=1000, random_state=42)
        p.fit(X_simple, y_simple)
        perceptrons.append(p)
        
        iterations_to_converge = len(p.training_history['iteration'])
        final_accuracy = p.training_history['accuracy'][-1]
        
        print(f"Learning rate {lr}: {iterations_to_converge} iterations, accuracy: {final_accuracy:.4f}")
    
    # 3. Non-linearly separable data (XOR problem)
    print("\n3. Perceptron Limitations: XOR Problem")
    print("-" * 50)
    
    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    perceptron_xor = Perceptron(learning_rate=0.1, max_iterations=1000, random_state=42)
    perceptron_xor.fit(X_xor, y_xor, verbose=True)
    
    accuracy_xor = perceptron_xor.score(X_xor, y_xor)
    print(f"XOR accuracy: {accuracy_xor:.4f} (Expected: ~0.5 due to linear inseparability)")
    
    # 4. Compare with different datasets
    print("\n4. Perceptron Performance on Different Datasets")
    print("-" * 50)
    
    datasets = {
        'Linearly Separable': make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                                 n_informative=2, n_clusters_per_class=1, random_state=42),
        'Overlapping Classes': make_classification(n_samples=200, n_features=2, n_redundant=0,
                                                 n_informative=2, n_clusters_per_class=1,
                                                 class_sep=0.5, random_state=42),
        'Two Blobs': make_blobs(n_samples=200, centers=2, n_features=2, random_state=42)
    }
    
    results = {}
    for name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        p = Perceptron(learning_rate=0.1, max_iterations=1000, random_state=42)
        p.fit(X_train, y_train)
        
        train_acc = p.score(X_train, y_train)
        test_acc = p.score(X_test, y_test)
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'iterations': len(p.training_history['iteration'])
        }
        
        print(f"{name}:")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        print(f"  Iterations to converge: {results[name]['iterations']}")
    
    # 5. Advanced perceptron variants
    print("\n5. Advanced Perceptron Variants")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.3, random_state=42)
    
    # Standard perceptron
    standard_p = Perceptron(learning_rate=0.1, max_iterations=1000, random_state=42)
    standard_p.fit(X_train, y_train)
    
    # Adaptive perceptron
    adaptive_p = AdaptivePerceptron(initial_learning_rate=0.5, decay_rate=0.95, 
                                   max_iterations=1000, random_state=42)
    adaptive_p.fit(X_train, y_train)
    
    # Averaged perceptron
    averaged_p = AveragedPerceptron(learning_rate=0.1, max_iterations=1000, random_state=42)
    averaged_p.fit(X_train, y_train)
    
    variants = {
        'Standard': standard_p,
        'Adaptive': adaptive_p,
        'Averaged': averaged_p
    }
    
    for name, p in variants.items():
        train_acc = p.score(X_train, y_train)
        test_acc = p.score(X_test, y_test)
        print(f"{name} Perceptron:")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
    
    return {
        'simple_data': (X_simple, y_simple),
        'xor_data': (X_xor, y_xor),
        'perceptron': perceptron,
        'variants': variants,
        'datasets': datasets,
        'results': results
    }

# Real-world application example
def perceptron_spam_detection_example():
    """Example: Simple spam detection using perceptron"""
    
    print("\n=== Perceptron Application: Simple Spam Detection ===")
    print("-" * 60)
    
    # Simulate simple email features (normally would use TF-IDF or similar)
    np.random.seed(42)
    
    # Features: [word_count, exclamation_marks, caps_ratio, suspicious_words]
    # Spam emails: higher values
    spam_emails = np.random.normal([50, 3, 0.3, 2], [10, 1, 0.1, 1], (100, 4))
    spam_labels = np.ones(100)
    
    # Ham emails: lower values
    ham_emails = np.random.normal([20, 0.5, 0.1, 0.2], [5, 0.3, 0.05, 0.3], (100, 4))
    ham_labels = np.zeros(100)
    
    # Combine data
    X = np.vstack([spam_emails, ham_emails])
    y = np.hstack([spam_labels, ham_labels])
    
    # Add some noise to make it more realistic
    X += np.random.normal(0, 1, X.shape)
    
    # Ensure no negative values for count features
    X = np.abs(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Normalize features (important for perceptron)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train perceptron
    spam_perceptron = Perceptron(learning_rate=0.01, max_iterations=1000, random_state=42)
    spam_perceptron.fit(X_train_scaled, y_train, verbose=True)
    
    # Evaluate
    train_accuracy = spam_perceptron.score(X_train_scaled, y_train)
    test_accuracy = spam_perceptron.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions on test set
    y_pred = spam_perceptron.predict(X_test_scaled)
    
    # Show classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Show feature importance (weights)
    feature_names = ['Word Count', 'Exclamation Marks', 'Caps Ratio', 'Suspicious Words']
    print("\nFeature Weights (importance):")
    for name, weight in zip(feature_names, spam_perceptron.weights):
        print(f"  {name}: {weight:.4f}")
    
    return {
        'perceptron': spam_perceptron,
        'scaler': scaler,
        'test_accuracy': test_accuracy,
        'feature_names': feature_names
    }

# Visualization examples
def create_perceptron_visualizations():
    """Create comprehensive visualizations"""
    
    # Generate 2D data for visualization
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Train perceptron
    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000, random_state=42)
    perceptron.fit(X, y)
    
    # Create visualizer
    visualizer = PerceptronVisualizer()
    
    # Plot decision boundary
    visualizer.plot_decision_boundary(perceptron, X, y, "Perceptron Decision Boundary")
    
    # Plot training history
    perceptron.plot_training_history()
    
    # Compare different learning rates
    learning_rates = [0.01, 0.1, 1.0]
    perceptrons_lr = []
    names_lr = []
    
    for lr in learning_rates:
        p = Perceptron(learning_rate=lr, max_iterations=1000, random_state=42)
        p.fit(X, y)
        perceptrons_lr.append(p)
        names_lr.append(f'LR = {lr}')
    
    visualizer.compare_perceptrons(perceptrons_lr, names_lr, X, y)

# Run demonstrations
if __name__ == "__main__":
    demo_results = demonstrate_perceptron()
    spam_example = perceptron_spam_detection_example()
    
    print("\n" + "="*60)
    print("Creating visualizations...")
    create_perceptron_visualizations()
```

This comprehensive perceptron implementation includes:

1. **Core Perceptron class**: Complete implementation with training history tracking
2. **Advanced variants**: Adaptive learning rate and averaged perceptron
3. **Visualization tools**: Decision boundary plotting and training history visualization
4. **Real-world example**: Spam detection application
5. **Multiple demonstrations**: Different datasets and learning rates
6. **Educational features**: Clear explanations and parameter tracking
7. **Production-ready code**: Error handling, type hints, and comprehensive documentation

---

## Question 4

**Implement the k-means clustering algorithm using only standard Python libraries.**

**Answer:**

```python
import math
import random
import itertools
from typing import List, Tuple, Optional, Dict, Any, Union
import json

class Point:
    """Represents a point in n-dimensional space"""
    
    def __init__(self, coordinates: List[float], label: Optional[str] = None):
        """
        Initialize a point
        
        Args:
            coordinates: List of coordinate values
            label: Optional label for the point
        """
        self.coordinates = coordinates
        self.label = label
        self.cluster_id = None
        self.dimension = len(coordinates)
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point"""
        if self.dimension != other.dimension:
            raise ValueError("Points must have the same dimension")
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.coordinates, other.coordinates)))
    
    def manhattan_distance_to(self, other: 'Point') -> float:
        """Calculate Manhattan distance to another point"""
        if self.dimension != other.dimension:
            raise ValueError("Points must have the same dimension")
        
        return sum(abs(a - b) for a, b in zip(self.coordinates, other.coordinates))
    
    def __repr__(self) -> str:
        coords_str = ', '.join(f'{x:.3f}' for x in self.coordinates)
        cluster_str = f', cluster={self.cluster_id}' if self.cluster_id is not None else ''
        label_str = f', label={self.label}' if self.label else ''
        return f'Point([{coords_str}]{cluster_str}{label_str})'
    
    def __eq__(self, other: 'Point') -> bool:
        if not isinstance(other, Point):
            return False
        return self.coordinates == other.coordinates
    
    def __hash__(self) -> int:
        return hash(tuple(self.coordinates))

class Cluster:
    """Represents a cluster of points"""
    
    def __init__(self, centroid: Point, cluster_id: int):
        """
        Initialize a cluster
        
        Args:
            centroid: Center point of the cluster
            cluster_id: Unique identifier for the cluster
        """
        self.centroid = centroid
        self.cluster_id = cluster_id
        self.points = []
        self.previous_centroid = None
    
    def add_point(self, point: Point) -> None:
        """Add a point to this cluster"""
        point.cluster_id = self.cluster_id
        self.points.append(point)
    
    def remove_point(self, point: Point) -> None:
        """Remove a point from this cluster"""
        if point in self.points:
            self.points.remove(point)
            point.cluster_id = None
    
    def clear_points(self) -> None:
        """Remove all points from this cluster"""
        for point in self.points:
            point.cluster_id = None
        self.points.clear()
    
    def update_centroid(self) -> float:
        """
        Update centroid based on current points
        
        Returns:
            Distance moved by centroid
        """
        if not self.points:
            return 0.0
        
        # Store previous centroid
        self.previous_centroid = Point(self.centroid.coordinates[:])
        
        # Calculate new centroid
        dimension = self.points[0].dimension
        new_coordinates = []
        
        for i in range(dimension):
            coord_sum = sum(point.coordinates[i] for point in self.points)
            new_coordinates.append(coord_sum / len(self.points))
        
        self.centroid = Point(new_coordinates)
        
        # Return distance moved
        if self.previous_centroid:
            return self.centroid.distance_to(self.previous_centroid)
        return 0.0
    
    def calculate_sse(self) -> float:
        """Calculate sum of squared errors for this cluster"""
        if not self.points:
            return 0.0
        
        return sum(point.distance_to(self.centroid) ** 2 for point in self.points)
    
    def calculate_diameter(self) -> float:
        """Calculate diameter (maximum distance between any two points)"""
        if len(self.points) < 2:
            return 0.0
        
        max_distance = 0.0
        for i, point1 in enumerate(self.points):
            for point2 in self.points[i+1:]:
                distance = point1.distance_to(point2)
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def __repr__(self) -> str:
        return f'Cluster(id={self.cluster_id}, points={len(self.points)}, centroid={self.centroid})'

class KMeansClusterer:
    """
    K-Means clustering implementation using only standard Python libraries
    """
    
    def __init__(self, k: int, max_iterations: int = 100, tolerance: float = 1e-6,
                 random_seed: Optional[int] = None, init_method: str = 'random'):
        """
        Initialize K-Means clusterer
        
        Args:
            k: Number of clusters
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            random_seed: Random seed for reproducibility
            init_method: Initialization method ('random', 'kmeans++', 'furthest')
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init_method = init_method
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
        
        # Model state
        self.clusters = []
        self.points = []
        self.fitted = False
        
        # Training history
        self.history = {
            'iteration': [],
            'sse': [],
            'centroid_movement': [],
            'cluster_assignments': []
        }
    
    def _initialize_centroids(self, points: List[Point]) -> List[Point]:
        """Initialize centroids using specified method"""
        if self.init_method == 'random':
            return self._random_initialization(points)
        elif self.init_method == 'kmeans++':
            return self._kmeans_plus_plus_initialization(points)
        elif self.init_method == 'furthest':
            return self._furthest_first_initialization(points)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
    
    def _random_initialization(self, points: List[Point]) -> List[Point]:
        """Randomly select k points as initial centroids"""
        if len(points) < self.k:
            raise ValueError(f"Not enough points ({len(points)}) for {self.k} clusters")
        
        selected_points = random.sample(points, self.k)
        return [Point(point.coordinates[:]) for point in selected_points]
    
    def _kmeans_plus_plus_initialization(self, points: List[Point]) -> List[Point]:
        """K-means++ initialization for better initial centroids"""
        if len(points) < self.k:
            raise ValueError(f"Not enough points ({len(points)}) for {self.k} clusters")
        
        centroids = []
        
        # Choose first centroid randomly
        first_centroid = random.choice(points)
        centroids.append(Point(first_centroid.coordinates[:]))
        
        # Choose remaining centroids
        for _ in range(self.k - 1):
            distances = []
            
            # Calculate distance to nearest centroid for each point
            for point in points:
                min_distance = min(point.distance_to(centroid) for centroid in centroids)
                distances.append(min_distance ** 2)
            
            # Choose next centroid with probability proportional to squared distance
            total_distance = sum(distances)
            if total_distance == 0:
                # Fallback to random selection
                remaining_points = [p for p in points if Point(p.coordinates[:]) not in centroids]
                if remaining_points:
                    next_centroid = random.choice(remaining_points)
                    centroids.append(Point(next_centroid.coordinates[:]))
            else:
                probabilities = [d / total_distance for d in distances]
                cumulative_probs = []
                cumsum = 0
                for prob in probabilities:
                    cumsum += prob
                    cumulative_probs.append(cumsum)
                
                # Select based on cumulative probability
                rand_val = random.random()
                selected_index = 0
                for i, cum_prob in enumerate(cumulative_probs):
                    if rand_val <= cum_prob:
                        selected_index = i
                        break
                
                next_centroid = points[selected_index]
                centroids.append(Point(next_centroid.coordinates[:]))
        
        return centroids
    
    def _furthest_first_initialization(self, points: List[Point]) -> List[Point]:
        """Choose centroids that are furthest from existing centroids"""
        if len(points) < self.k:
            raise ValueError(f"Not enough points ({len(points)}) for {self.k} clusters")
        
        centroids = []
        
        # Choose first centroid randomly
        first_centroid = random.choice(points)
        centroids.append(Point(first_centroid.coordinates[:]))
        
        # Choose remaining centroids
        for _ in range(self.k - 1):
            max_min_distance = -1
            furthest_point = None
            
            # Find point with maximum distance to nearest centroid
            for point in points:
                min_distance = min(point.distance_to(centroid) for centroid in centroids)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    furthest_point = point
            
            if furthest_point:
                centroids.append(Point(furthest_point.coordinates[:]))
        
        return centroids
    
    def _assign_points_to_clusters(self) -> int:
        """
        Assign each point to the nearest cluster
        
        Returns:
            Number of points that changed clusters
        """
        changes = 0
        
        for point in self.points:
            # Find nearest cluster
            min_distance = float('inf')
            nearest_cluster = None
            
            for cluster in self.clusters:
                distance = point.distance_to(cluster.centroid)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster = cluster
            
            # Check if assignment changed
            if point.cluster_id != nearest_cluster.cluster_id:
                # Remove from old cluster
                if point.cluster_id is not None:
                    old_cluster = self.clusters[point.cluster_id]
                    old_cluster.remove_point(point)
                
                # Add to new cluster
                nearest_cluster.add_point(point)
                changes += 1
        
        return changes
    
    def _update_centroids(self) -> float:
        """
        Update all cluster centroids
        
        Returns:
            Maximum distance moved by any centroid
        """
        max_movement = 0.0
        
        for cluster in self.clusters:
            movement = cluster.update_centroid()
            max_movement = max(max_movement, movement)
        
        return max_movement
    
    def _calculate_total_sse(self) -> float:
        """Calculate total sum of squared errors"""
        return sum(cluster.calculate_sse() for cluster in self.clusters)
    
    def fit(self, data: List[List[float]], verbose: bool = False) -> 'KMeansClusterer':
        """
        Fit the K-means model to the data
        
        Args:
            data: List of data points (each point is a list of coordinates)
            verbose: Whether to print iteration details
            
        Returns:
            self: Fitted clusterer
        """
        # Convert data to Point objects
        self.points = [Point(coordinates) for coordinates in data]
        
        if len(self.points) < self.k:
            raise ValueError(f"Number of points ({len(self.points)}) must be >= k ({self.k})")
        
        # Initialize centroids
        initial_centroids = self._initialize_centroids(self.points)
        
        # Create clusters
        self.clusters = [Cluster(centroid, i) for i, centroid in enumerate(initial_centroids)]
        
        # Clear history
        self.history = {
            'iteration': [],
            'sse': [],
            'centroid_movement': [],
            'cluster_assignments': []
        }
        
        if verbose:
            print(f"Starting K-means clustering with k={self.k}")
            print(f"Initialization method: {self.init_method}")
        
        # Main clustering loop
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            assignment_changes = self._assign_points_to_clusters()
            
            # Update centroids
            centroid_movement = self._update_centroids()
            
            # Calculate SSE
            total_sse = self._calculate_total_sse()
            
            # Store history
            self.history['iteration'].append(iteration + 1)
            self.history['sse'].append(total_sse)
            self.history['centroid_movement'].append(centroid_movement)
            self.history['cluster_assignments'].append(
                [[point.coordinates for point in cluster.points] for cluster in self.clusters]
            )
            
            if verbose:
                print(f"Iteration {iteration + 1}: SSE={total_sse:.6f}, "
                      f"Max centroid movement={centroid_movement:.6f}, "
                      f"Assignment changes={assignment_changes}")
            
            # Check for convergence
            if centroid_movement < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        self.fitted = True
        return self
    
    def predict(self, data: List[List[float]]) -> List[int]:
        """
        Predict cluster assignments for new data
        
        Args:
            data: List of data points to assign to clusters
            
        Returns:
            List of cluster assignments
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for coordinates in data:
            point = Point(coordinates)
            
            # Find nearest cluster
            min_distance = float('inf')
            nearest_cluster_id = 0
            
            for cluster in self.clusters:
                distance = point.distance_to(cluster.centroid)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster_id = cluster.cluster_id
            
            predictions.append(nearest_cluster_id)
        
        return predictions
    
    def get_cluster_centers(self) -> List[List[float]]:
        """Get the coordinates of cluster centers"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting cluster centers")
        
        return [cluster.centroid.coordinates for cluster in self.clusters]
    
    def get_labels(self) -> List[int]:
        """Get cluster labels for training data"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting labels")
        
        labels = [None] * len(self.points)
        for point in self.points:
            # Find index of this point in original data
            for i, original_point in enumerate(self.points):
                if point == original_point:
                    labels[i] = point.cluster_id
                    break
        
        return labels
    
    def calculate_inertia(self) -> float:
        """Calculate total within-cluster sum of squares (inertia)"""
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating inertia")
        
        return self._calculate_total_sse()
    
    def calculate_silhouette_score(self) -> float:
        """Calculate simplified silhouette score"""
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating silhouette score")
        
        if self.k == 1:
            return 0.0
        
        silhouette_scores = []
        
        for point in self.points:
            # Calculate average distance to points in same cluster (a)
            same_cluster_points = [p for p in self.clusters[point.cluster_id].points if p != point]
            if same_cluster_points:
                a = sum(point.distance_to(p) for p in same_cluster_points) / len(same_cluster_points)
            else:
                a = 0.0
            
            # Calculate average distance to points in nearest other cluster (b)
            min_avg_distance = float('inf')
            for cluster in self.clusters:
                if cluster.cluster_id != point.cluster_id and cluster.points:
                    avg_distance = sum(point.distance_to(p) for p in cluster.points) / len(cluster.points)
                    min_avg_distance = min(min_avg_distance, avg_distance)
            
            b = min_avg_distance if min_avg_distance != float('inf') else 0.0
            
            # Calculate silhouette score for this point
            if max(a, b) == 0:
                silhouette_scores.append(0.0)
            else:
                silhouette_scores.append((b - a) / max(a, b))
        
        return sum(silhouette_scores) / len(silhouette_scores) if silhouette_scores else 0.0
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get detailed information about clusters"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting cluster info")
        
        info = {
            'total_clusters': self.k,
            'total_points': len(self.points),
            'total_sse': self._calculate_total_sse(),
            'silhouette_score': self.calculate_silhouette_score(),
            'clusters': []
        }
        
        for cluster in self.clusters:
            cluster_info = {
                'id': cluster.cluster_id,
                'centroid': cluster.centroid.coordinates,
                'size': len(cluster.points),
                'sse': cluster.calculate_sse(),
                'diameter': cluster.calculate_diameter()
            }
            info['clusters'].append(cluster_info)
        
        return info

class KMeansVisualizer:
    """Visualization utilities for K-means clustering (text-based)"""
    
    @staticmethod
    def print_clusters(clusterer: KMeansClusterer) -> None:
        """Print cluster information in a readable format"""
        if not clusterer.fitted:
            print("Model not fitted yet")
            return
        
        print(f"\n=== K-Means Clustering Results (k={clusterer.k}) ===")
        print(f"Total points: {len(clusterer.points)}")
        print(f"Total SSE: {clusterer._calculate_total_sse():.6f}")
        print(f"Silhouette Score: {clusterer.calculate_silhouette_score():.6f}")
        
        for cluster in clusterer.clusters:
            print(f"\nCluster {cluster.cluster_id}:")
            print(f"  Centroid: {[f'{x:.3f}' for x in cluster.centroid.coordinates]}")
            print(f"  Size: {len(cluster.points)} points")
            print(f"  SSE: {cluster.calculate_sse():.6f}")
            print(f"  Diameter: {cluster.calculate_diameter():.6f}")
            
            if len(cluster.points) <= 5:
                print("  Points:")
                for point in cluster.points:
                    coords_str = [f'{x:.3f}' for x in point.coordinates]
                    print(f"    {coords_str}")
            else:
                print(f"  Points: {len(cluster.points)} points (showing first 3)")
                for i, point in enumerate(cluster.points[:3]):
                    coords_str = [f'{x:.3f}' for x in point.coordinates]
                    print(f"    {coords_str}")
    
    @staticmethod
    def print_convergence_history(clusterer: KMeansClusterer) -> None:
        """Print convergence history"""
        if not clusterer.fitted:
            print("Model not fitted yet")
            return
        
        print(f"\n=== Convergence History ===")
        print("Iter\tSSE\t\tMovement")
        print("-" * 30)
        
        for i, (sse, movement) in enumerate(zip(clusterer.history['sse'], 
                                                clusterer.history['centroid_movement'])):
            print(f"{i+1}\t{sse:.6f}\t{movement:.6f}")
    
    @staticmethod
    def create_ascii_plot_2d(clusterer: KMeansClusterer, width: int = 50, height: int = 20) -> None:
        """Create simple ASCII plot for 2D data"""
        if not clusterer.fitted:
            print("Model not fitted yet")
            return
        
        # Check if data is 2D
        if not clusterer.points or clusterer.points[0].dimension != 2:
            print("ASCII plot only available for 2D data")
            return
        
        # Find data bounds
        x_coords = [point.coordinates[0] for point in clusterer.points]
        y_coords = [point.coordinates[1] for point in clusterer.points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points
        symbols = ['*', '#', '@', '$', '%', '&', '+', '=', '!', '?']
        
        for point in clusterer.points:
            x, y = point.coordinates
            
            # Convert to grid coordinates
            grid_x = int((x - x_min) / (x_max - x_min) * (width - 1))
            grid_y = int((y - y_min) / (y_max - y_min) * (height - 1))
            
            # Ensure within bounds
            grid_x = max(0, min(width - 1, grid_x))
            grid_y = max(0, min(height - 1, grid_y))
            
            # Use different symbol for each cluster
            symbol = symbols[point.cluster_id % len(symbols)]
            grid[height - 1 - grid_y][grid_x] = symbol
        
        # Plot centroids
        for cluster in clusterer.clusters:
            x, y = cluster.centroid.coordinates
            
            grid_x = int((x - x_min) / (x_max - x_min) * (width - 1))
            grid_y = int((y - y_min) / (y_max - y_min) * (height - 1))
            
            grid_x = max(0, min(width - 1, grid_x))
            grid_y = max(0, min(height - 1, grid_y))
            
            grid[height - 1 - grid_y][grid_x] = 'C'  # C for centroid
        
        # Print grid
        print(f"\n=== ASCII Plot (Centroids marked with 'C') ===")
        print(f"X range: [{x_min:.2f}, {x_max:.2f}]")
        print(f"Y range: [{y_min:.2f}, {y_max:.2f}]")
        print()
        
        for row in grid:
            print(''.join(row))
        
        # Print legend
        print("\nLegend:")
        for i, cluster in enumerate(clusterer.clusters):
            symbol = symbols[i % len(symbols)]
            print(f"  Cluster {i}: {symbol} ({len(cluster.points)} points)")
        print("  Centroids: C")

# Utility functions
def generate_sample_data(n_points: int = 100, n_clusters: int = 3, 
                        dimension: int = 2, cluster_spread: float = 1.0,
                        random_seed: Optional[int] = None) -> List[List[float]]:
    """Generate sample data for clustering"""
    if random_seed is not None:
        random.seed(random_seed)
    
    data = []
    points_per_cluster = n_points // n_clusters
    
    # Generate cluster centers
    cluster_centers = []
    for _ in range(n_clusters):
        center = [random.uniform(-5, 5) for _ in range(dimension)]
        cluster_centers.append(center)
    
    # Generate points around each center
    for i, center in enumerate(cluster_centers):
        cluster_points = points_per_cluster
        if i == len(cluster_centers) - 1:  # Last cluster gets remaining points
            cluster_points = n_points - (n_clusters - 1) * points_per_cluster
        
        for _ in range(cluster_points):
            point = []
            for j in range(dimension):
                # Add random noise around cluster center
                coord = center[j] + random.gauss(0, cluster_spread)
                point.append(coord)
            data.append(point)
    
    # Shuffle the data
    random.shuffle(data)
    return data

def find_optimal_k(data: List[List[float]], max_k: int = 10, 
                  random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Find optimal number of clusters using elbow method
    
    Args:
        data: Training data
        max_k: Maximum number of clusters to try
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results for each k
    """
    results = {
        'k_values': [],
        'sse_values': [],
        'silhouette_scores': [],
        'models': {}
    }
    
    print("Finding optimal k using elbow method...")
    
    for k in range(1, max_k + 1):
        print(f"Testing k={k}...")
        
        # Fit K-means
        kmeans = KMeansClusterer(k=k, random_seed=random_seed)
        kmeans.fit(data)
        
        # Calculate metrics
        sse = kmeans.calculate_inertia()
        silhouette = kmeans.calculate_silhouette_score() if k > 1 else 0.0
        
        # Store results
        results['k_values'].append(k)
        results['sse_values'].append(sse)
        results['silhouette_scores'].append(silhouette)
        results['models'][k] = kmeans
        
        print(f"  SSE: {sse:.6f}, Silhouette: {silhouette:.6f}")
    
    return results

# Demonstration and examples
def demonstrate_kmeans():
    """Comprehensive K-means demonstration"""
    
    print("=== K-Means Clustering Implementation Demonstrations ===\n")
    
    # 1. Basic K-means on 2D data
    print("1. Basic K-means on 2D Generated Data")
    print("-" * 50)
    
    # Generate sample data
    data_2d = generate_sample_data(n_points=150, n_clusters=3, dimension=2, 
                                  cluster_spread=1.0, random_seed=42)
    
    # Fit K-means
    kmeans = KMeansClusterer(k=3, random_seed=42, init_method='kmeans++')
    kmeans.fit(data_2d, verbose=True)
    
    # Print results
    visualizer = KMeansVisualizer()
    visualizer.print_clusters(kmeans)
    visualizer.print_convergence_history(kmeans)
    visualizer.create_ascii_plot_2d(kmeans)
    
    # 2. Compare initialization methods
    print("\n" + "="*60)
    print("2. Comparison of Initialization Methods")
    print("-" * 50)
    
    init_methods = ['random', 'kmeans++', 'furthest']
    for method in init_methods:
        print(f"\nTesting {method} initialization:")
        kmeans_init = KMeansClusterer(k=3, random_seed=42, init_method=method)
        kmeans_init.fit(data_2d)
        
        sse = kmeans_init.calculate_inertia()
        silhouette = kmeans_init.calculate_silhouette_score()
        iterations = len(kmeans_init.history['iteration'])
        
        print(f"  SSE: {sse:.6f}")
        print(f"  Silhouette Score: {silhouette:.6f}")
        print(f"  Iterations to converge: {iterations}")
    
    # 3. Find optimal k
    print("\n" + "="*60)
    print("3. Finding Optimal Number of Clusters")
    print("-" * 50)
    
    optimal_k_results = find_optimal_k(data_2d, max_k=8, random_seed=42)
    
    print("\nSSE by k:")
    for k, sse in zip(optimal_k_results['k_values'], optimal_k_results['sse_values']):
        print(f"  k={k}: SSE={sse:.6f}")
    
    print("\nSilhouette Score by k:")
    for k, silhouette in zip(optimal_k_results['k_values'], optimal_k_results['silhouette_scores']):
        if k > 1:
            print(f"  k={k}: Silhouette={silhouette:.6f}")
    
    # 4. High-dimensional data
    print("\n" + "="*60)
    print("4. K-means on High-Dimensional Data")
    print("-" * 50)
    
    data_5d = generate_sample_data(n_points=200, n_clusters=4, dimension=5, 
                                  cluster_spread=0.8, random_seed=42)
    
    kmeans_5d = KMeansClusterer(k=4, random_seed=42, init_method='kmeans++')
    kmeans_5d.fit(data_5d, verbose=True)
    
    visualizer.print_clusters(kmeans_5d)
    
    # 5. Edge cases and robustness
    print("\n" + "="*60)
    print("5. Edge Cases and Robustness Testing")
    print("-" * 50)
    
    # Single cluster
    print("Testing k=1 (single cluster):")
    single_cluster = KMeansClusterer(k=1, random_seed=42)
    single_cluster.fit(data_2d)
    print(f"  SSE: {single_cluster.calculate_inertia():.6f}")
    
    # K equals number of points
    small_data = data_2d[:5]
    print(f"\nTesting k=5 on {len(small_data)} points:")
    many_clusters = KMeansClusterer(k=5, random_seed=42)
    many_clusters.fit(small_data)
    print(f"  SSE: {many_clusters.calculate_inertia():.6f}")
    
    return {
        'data_2d': data_2d,
        'data_5d': data_5d,
        'basic_kmeans': kmeans,
        'optimal_k_results': optimal_k_results,
        'high_dim_kmeans': kmeans_5d
    }

# Real-world application example
def customer_segmentation_example():
    """Example: Customer segmentation using K-means"""
    
    print("\n=== Customer Segmentation Example ===")
    print("-" * 50)
    
    # Simulate customer data
    random.seed(42)
    
    # Features: [annual_spending, frequency_of_visits, average_transaction, age]
    customers = []
    
    # High-value customers
    for _ in range(50):
        customer = [
            random.uniform(5000, 15000),  # Annual spending
            random.uniform(20, 50),       # Visits per year
            random.uniform(100, 300),     # Average transaction
            random.uniform(30, 60)        # Age
        ]
        customers.append(customer)
    
    # Medium-value customers
    for _ in range(80):
        customer = [
            random.uniform(1000, 5000),   # Annual spending
            random.uniform(5, 25),        # Visits per year
            random.uniform(50, 150),      # Average transaction
            random.uniform(25, 55)        # Age
        ]
        customers.append(customer)
    
    # Low-value customers
    for _ in range(70):
        customer = [
            random.uniform(100, 1000),    # Annual spending
            random.uniform(1, 10),        # Visits per year
            random.uniform(20, 80),       # Average transaction
            random.uniform(18, 45)        # Age
        ]
        customers.append(customer)
    
    # Shuffle customers
    random.shuffle(customers)
    
    print(f"Customer dataset: {len(customers)} customers with 4 features")
    
    # Find optimal number of segments
    segmentation_results = find_optimal_k(customers, max_k=6, random_seed=42)
    
    # Use k=3 for customer segmentation
    customer_kmeans = KMeansClusterer(k=3, random_seed=42, init_method='kmeans++')
    customer_kmeans.fit(customers, verbose=True)
    
    # Analyze segments
    print("\n=== Customer Segment Analysis ===")
    visualizer = KMeansVisualizer()
    visualizer.print_clusters(customer_kmeans)
    
    # Calculate segment characteristics
    feature_names = ['Annual Spending', 'Visit Frequency', 'Avg Transaction', 'Age']
    
    print("\nSegment Characteristics:")
    for cluster in customer_kmeans.clusters:
        print(f"\nSegment {cluster.cluster_id} ({len(cluster.points)} customers):")
        
        if cluster.points:
            # Calculate averages for each feature
            feature_sums = [0] * 4
            for point in cluster.points:
                for i, value in enumerate(point.coordinates):
                    feature_sums[i] += value
            
            feature_averages = [s / len(cluster.points) for s in feature_sums]
            
            for name, avg in zip(feature_names, feature_averages):
                print(f"  Average {name}: {avg:.2f}")
    
    return {
        'customers': customers,
        'kmeans': customer_kmeans,
        'segmentation_results': segmentation_results
    }

# Run demonstrations
if __name__ == "__main__":
    demo_results = demonstrate_kmeans()
    customer_example = customer_segmentation_example()
    
    print("\n" + "="*60)
    print("K-means clustering implementation complete!")
    print("This implementation uses only standard Python libraries.")
```

This comprehensive K-means implementation includes:

1. **Core K-means algorithm**: Complete implementation using only standard Python libraries
2. **Multiple initialization methods**: Random, K-means++, and furthest-first
3. **Comprehensive evaluation metrics**: SSE, silhouette score, cluster diameter
4. **Visualization tools**: ASCII plotting for 2D data and detailed cluster information
5. **Utility functions**: Data generation and optimal k finding
6. **Real-world example**: Customer segmentation application
7. **Edge case handling**: Single clusters, high-dimensional data
8. **Educational features**: Detailed history tracking and verbose output

---

## Question 5

**Create a Python script that performs linear regression on a dataset using NumPy.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Union
import pandas as pd
from dataclasses import dataclass
import warnings

@dataclass
class RegressionMetrics:
    """Container for regression evaluation metrics"""
    mse: float
    rmse: float
    mae: float
    r2: float
    adjusted_r2: float
    aic: float
    bic: float

class LinearRegression:
    """
    Linear Regression implementation using NumPy with comprehensive features
    
    Supports multiple solving methods, regularization, and detailed diagnostics
    """
    
    def __init__(self, fit_intercept: bool = True, method: str = 'normal_equation', 
                 regularization: Optional[str] = None, alpha: float = 1.0,
                 normalize: bool = False):
        """
        Initialize Linear Regression model
        
        Args:
            fit_intercept: Whether to add intercept term
            method: Solving method ('normal_equation', 'gradient_descent', 'svd', 'qr')
            regularization: Regularization type (None, 'ridge', 'lasso')
            alpha: Regularization strength
            normalize: Whether to normalize features
        """
        self.fit_intercept = fit_intercept
        self.method = method
        self.regularization = regularization
        self.alpha = alpha
        self.normalize = normalize
        
        # Model parameters
        self.coefficients = None
        self.intercept = None
        
        # Data preprocessing
        self.feature_means = None
        self.feature_stds = None
        self.target_mean = None
        
        # Model diagnostics
        self.training_history = {}
        self.fitted = False
        
        # Validate inputs
        valid_methods = ['normal_equation', 'gradient_descent', 'svd', 'qr']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        valid_regularization = [None, 'ridge', 'lasso']
        if regularization not in valid_regularization:
            raise ValueError(f"Regularization must be one of {valid_regularization}")
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to feature matrix"""
        ones = np.ones((X.shape[0], 1))
        return np.column_stack([ones, X])
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features to zero mean and unit variance"""
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1
        
        return (X - self.feature_means) / self.feature_stds
    
    def _solve_normal_equation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve using normal equation: θ = (X^T X)^(-1) X^T y"""
        XtX = np.dot(X.T, X)
        
        # Add regularization if specified
        if self.regularization == 'ridge':
            # Add ridge penalty (but not to intercept if present)
            regularization_matrix = self.alpha * np.eye(XtX.shape[0])
            if self.fit_intercept:
                regularization_matrix[0, 0] = 0  # Don't regularize intercept
            XtX += regularization_matrix
        
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            warnings.warn("Matrix is singular, using pseudo-inverse")
            XtX_inv = np.linalg.pinv(XtX)
        
        Xty = np.dot(X.T, y)
        return np.dot(XtX_inv, Xty)
    
    def _solve_svd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve using Singular Value Decomposition"""
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Add small regularization to prevent numerical issues
        s_reg = s / (s**2 + self.alpha if self.regularization == 'ridge' else s**2 + 1e-10)
        
        # Compute pseudo-inverse
        X_pinv = np.dot(Vt.T, np.dot(np.diag(s_reg), U.T))
        
        return np.dot(X_pinv, y)
    
    def _solve_qr(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve using QR decomposition"""
        Q, R = np.linalg.qr(X)
        
        # Add regularization to R if specified
        if self.regularization == 'ridge':
            # Augment R with regularization
            reg_matrix = np.sqrt(self.alpha) * np.eye(R.shape[1])
            if self.fit_intercept:
                reg_matrix[0, 0] = 0  # Don't regularize intercept
            R_aug = np.vstack([R, reg_matrix])
            Q_aug = np.vstack([Q, np.zeros((reg_matrix.shape[0], Q.shape[1]))])
        else:
            R_aug = R
            Q_aug = Q
        
        # Solve R * theta = Q^T * y
        Qty = np.dot(Q_aug.T, np.concatenate([y, np.zeros(reg_matrix.shape[0])]) if self.regularization == 'ridge' else y)
        return np.linalg.solve(R_aug, Qty)
    
    def _solve_gradient_descent(self, X: np.ndarray, y: np.ndarray, 
                               learning_rate: float = 0.01, max_iter: int = 10000,
                               tolerance: float = 1e-8) -> np.ndarray:
        """Solve using gradient descent"""
        m, n = X.shape
        theta = np.zeros(n)
        
        # Store training history
        self.training_history = {
            'iteration': [],
            'cost': [],
            'gradient_norm': []
        }
        
        for iteration in range(max_iter):
            # Forward pass
            predictions = np.dot(X, theta)
            errors = predictions - y
            
            # Calculate cost
            cost = np.mean(errors**2) / 2
            
            # Add regularization to cost
            if self.regularization == 'ridge':
                if self.fit_intercept:
                    # Don't regularize intercept
                    reg_cost = self.alpha * np.sum(theta[1:]**2) / (2 * m)
                else:
                    reg_cost = self.alpha * np.sum(theta**2) / (2 * m)
                cost += reg_cost
            elif self.regularization == 'lasso':
                if self.fit_intercept:
                    reg_cost = self.alpha * np.sum(np.abs(theta[1:])) / m
                else:
                    reg_cost = self.alpha * np.sum(np.abs(theta)) / m
                cost += reg_cost
            
            # Calculate gradients
            gradient = np.dot(X.T, errors) / m
            
            # Add regularization to gradient
            if self.regularization == 'ridge':
                reg_gradient = self.alpha * theta / m
                if self.fit_intercept:
                    reg_gradient[0] = 0  # Don't regularize intercept
                gradient += reg_gradient
            elif self.regularization == 'lasso':
                reg_gradient = self.alpha * np.sign(theta) / m
                if self.fit_intercept:
                    reg_gradient[0] = 0  # Don't regularize intercept
                gradient += reg_gradient
            
            # Update parameters
            theta -= learning_rate * gradient
            
            # Store history
            self.training_history['iteration'].append(iteration)
            self.training_history['cost'].append(cost)
            self.training_history['gradient_norm'].append(np.linalg.norm(gradient))
            
            # Check convergence
            if np.linalg.norm(gradient) < tolerance:
                break
        
        return theta
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            learning_rate: float = 0.01, max_iter: int = 10000) -> 'LinearRegression':
        """
        Fit the linear regression model
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum iterations for gradient descent
            
        Returns:
            self: Fitted model
        """
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Ensure 2D array for X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Store original dimensions
        self.n_samples, self.n_features = X.shape
        
        # Store target mean for R² calculation
        self.target_mean = np.mean(y)
        
        # Normalize features if requested
        if self.normalize:
            X = self._normalize_features(X, fit=True)
        
        # Add intercept if requested
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # Solve using specified method
        if self.method == 'normal_equation':
            theta = self._solve_normal_equation(X, y)
        elif self.method == 'gradient_descent':
            theta = self._solve_gradient_descent(X, y, learning_rate, max_iter)
        elif self.method == 'svd':
            theta = self._solve_svd(X, y)
        elif self.method == 'qr':
            theta = self._solve_qr(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Extract intercept and coefficients
        if self.fit_intercept:
            self.intercept = theta[0]
            self.coefficients = theta[1:]
        else:
            self.intercept = 0
            self.coefficients = theta
        
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Normalize features if model was trained with normalization
        if self.normalize:
            X = self._normalize_features(X, fit=False)
        
        # Make predictions
        predictions = np.dot(X, self.coefficients) + self.intercept
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> RegressionMetrics:
        """Calculate comprehensive regression metrics"""
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating metrics")
        
        predictions = self.predict(X)
        residuals = y - predictions
        n = len(y)
        p = self.n_features
        
        # Basic metrics
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - self.target_mean)**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        
        # Adjusted R-squared
        if n > p + 1:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = r2
        
        # Information criteria
        log_likelihood = -n/2 * np.log(2 * np.pi * mse) - ss_res / (2 * mse)
        aic = 2 * (p + 1) - 2 * log_likelihood
        bic = np.log(n) * (p + 1) - 2 * log_likelihood
        
        return RegressionMetrics(
            mse=mse, rmse=rmse, mae=mae, r2=r2, adjusted_r2=adj_r2, aic=aic, bic=bic
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance based on absolute coefficient values"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return np.abs(self.coefficients)
    
    def plot_training_history(self) -> None:
        """Plot training history for gradient descent"""
        if not self.training_history:
            print("No training history available (only for gradient descent method)")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot cost function
        ax1.plot(self.training_history['iteration'], self.training_history['cost'])
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.set_title('Training Cost')
        ax1.grid(True)
        
        # Plot gradient norm
        ax2.plot(self.training_history['iteration'], self.training_history['gradient_norm'])
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Convergence')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def residual_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform residual analysis"""
        if not self.fitted:
            raise ValueError("Model must be fitted before residual analysis")
        
        predictions = self.predict(X)
        residuals = y - predictions
        standardized_residuals = residuals / np.std(residuals)
        
        return {
            'residuals': residuals,
            'standardized_residuals': standardized_residuals,
            'predictions': predictions,
            'leverage': self._calculate_leverage(X)
        }
    
    def _calculate_leverage(self, X: np.ndarray) -> np.ndarray:
        """Calculate leverage values for each observation"""
        if self.normalize:
            X = self._normalize_features(X, fit=False)
        
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        try:
            H = X @ np.linalg.inv(X.T @ X) @ X.T
            return np.diag(H)
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            H = X @ np.linalg.pinv(X.T @ X) @ X.T
            return np.diag(H)

class MultipleLinearRegression(LinearRegression):
    """Extended Linear Regression with additional features for multiple regression"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None, **kwargs):
        """Fit with optional feature names"""
        super().fit(X, y, **kwargs)
        
        if feature_names is not None:
            if len(feature_names) != self.n_features:
                raise ValueError(f"Number of feature names ({len(feature_names)}) must match number of features ({self.n_features})")
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        
        return self
    
    def get_summary(self, X: np.ndarray, y: np.ndarray) -> str:
        """Get detailed model summary"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        metrics = self.calculate_metrics(X, y)
        
        summary = f"""
Multiple Linear Regression Summary
{'='*50}

Model Configuration:
  Method: {self.method}
  Regularization: {self.regularization or 'None'}
  Alpha: {self.alpha}
  Normalize: {self.normalize}
  Fit Intercept: {self.fit_intercept}

Dataset Information:
  Number of observations: {len(y)}
  Number of features: {self.n_features}

Model Performance:
  R²: {metrics.r2:.6f}
  Adjusted R²: {metrics.adjusted_r2:.6f}
  RMSE: {metrics.rmse:.6f}
  MAE: {metrics.mae:.6f}
  AIC: {metrics.aic:.2f}
  BIC: {metrics.bic:.2f}

Coefficients:
  Intercept: {self.intercept:.6f}
"""
        
        for name, coef in zip(self.feature_names, self.coefficients):
            summary += f"  {name}: {coef:.6f}\n"
        
        return summary

# Utility functions for data generation and analysis
def generate_regression_data(n_samples: int = 100, n_features: int = 1, 
                           noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data"""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate true coefficients
    true_coefficients = np.random.randn(n_features)
    
    # Generate target with linear relationship + noise
    y = X @ true_coefficients + noise * np.random.randn(n_samples)
    
    return X, y

def load_boston_housing_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic Boston-like housing data
    (Since sklearn's Boston dataset is deprecated)
    """
    np.random.seed(42)
    n_samples = 506
    
    # Feature names
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
        'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Generate correlated features that mimic real housing data
    X = np.random.randn(n_samples, len(feature_names))
    
    # Create some realistic relationships
    X[:, 0] = np.abs(X[:, 0] * 2)  # Crime rate (positive)
    X[:, 1] = np.abs(X[:, 1] * 20)  # Residential zoned land
    X[:, 3] = np.random.binomial(1, 0.1, n_samples)  # Charles River dummy
    X[:, 5] = 6 + X[:, 5] * 0.5  # Average rooms (around 6)
    X[:, 6] = np.abs(X[:, 6] * 20)  # Age of buildings
    
    # Create target variable with realistic relationships
    y = (24 + 
         -0.1 * X[:, 0] +  # Crime decreases price
         0.1 * X[:, 1] +   # Zoning increases price
         -0.5 * X[:, 2] +  # Industry decreases price
         3 * X[:, 3] +     # River access increases price
         -10 * X[:, 4] +   # NOX decreases price
         8 * X[:, 5] +     # Rooms increase price
         -0.1 * X[:, 6] +  # Age decreases price
         0.5 * X[:, 7] +   # Distance effects
         -0.3 * X[:, 8] +  # Highway access
         -0.01 * X[:, 9] + # Tax decreases price
         -0.5 * X[:, 10] + # Pupil-teacher ratio
         0.01 * X[:, 11] + # B index
         -0.5 * X[:, 12] + # LSTAT decreases price
         np.random.randn(n_samples) * 2)  # Noise
    
    # Ensure positive prices
    y = np.maximum(y, 5)
    
    return X, y, feature_names

def compare_regression_methods(X: np.ndarray, y: np.ndarray) -> Dict[str, LinearRegression]:
    """Compare different regression solving methods"""
    methods = ['normal_equation', 'svd', 'qr', 'gradient_descent']
    results = {}
    
    print("Comparing regression methods:")
    print("-" * 40)
    
    for method in methods:
        print(f"Testing {method}...")
        
        # Create and fit model
        model = LinearRegression(method=method)
        model.fit(X, y)
        
        # Calculate metrics
        metrics = model.calculate_metrics(X, y)
        
        results[method] = model
        
        print(f"  R²: {metrics.r2:.6f}")
        print(f"  RMSE: {metrics.rmse:.6f}")
        
        if method == 'gradient_descent' and model.training_history:
            iterations = len(model.training_history['iteration'])
            print(f"  Iterations: {iterations}")
    
    return results

# Demonstration and examples
def demonstrate_linear_regression():
    """Comprehensive linear regression demonstration"""
    
    print("=== Linear Regression Implementation Demonstrations ===\n")
    
    # 1. Simple linear regression (1 feature)
    print("1. Simple Linear Regression (1 Feature)")
    print("-" * 50)
    
    # Generate simple data
    X_simple, y_simple = generate_regression_data(n_samples=100, n_features=1, 
                                                 noise=0.2, random_state=42)
    
    # Fit model
    simple_lr = LinearRegression(method='normal_equation')
    simple_lr.fit(X_simple, y_simple)
    
    # Evaluate
    metrics = simple_lr.calculate_metrics(X_simple, y_simple)
    print(f"Coefficient: {simple_lr.coefficients[0]:.6f}")
    print(f"Intercept: {simple_lr.intercept:.6f}")
    print(f"R²: {metrics.r2:.6f}")
    print(f"RMSE: {metrics.rmse:.6f}")
    
    # 2. Multiple linear regression
    print("\n2. Multiple Linear Regression")
    print("-" * 50)
    
    # Generate multi-feature data
    X_multi, y_multi = generate_regression_data(n_samples=200, n_features=5, 
                                               noise=0.3, random_state=42)
    
    feature_names = [f'Feature_{i+1}' for i in range(5)]
    
    # Fit model
    multi_lr = MultipleLinearRegression(method='normal_equation')
    multi_lr.fit(X_multi, y_multi, feature_names=feature_names)
    
    # Print summary
    summary = multi_lr.get_summary(X_multi, y_multi)
    print(summary)
    
    # 3. Compare different solving methods
    print("\n3. Comparison of Solving Methods")
    print("-" * 50)
    
    method_results = compare_regression_methods(X_multi, y_multi)
    
    # 4. Regularization comparison
    print("\n4. Regularization Comparison")
    print("-" * 50)
    
    regularizations = [None, 'ridge']
    alphas = [0.1, 1.0, 10.0]
    
    for reg in regularizations:
        if reg is None:
            print("No regularization:")
            model = LinearRegression(regularization=None)
            model.fit(X_multi, y_multi)
            metrics = model.calculate_metrics(X_multi, y_multi)
            print(f"  R²: {metrics.r2:.6f}, RMSE: {metrics.rmse:.6f}")
        else:
            print(f"{reg.title()} regularization:")
            for alpha in alphas:
                model = LinearRegression(regularization=reg, alpha=alpha)
                model.fit(X_multi, y_multi)
                metrics = model.calculate_metrics(X_multi, y_multi)
                print(f"  α={alpha}: R²={metrics.r2:.6f}, RMSE={metrics.rmse:.6f}")
    
    # 5. Real-world example: Housing prices
    print("\n5. Real-World Example: Housing Price Prediction")
    print("-" * 50)
    
    X_housing, y_housing, housing_features = load_boston_housing_data()
    
    # Split data
    n_train = int(0.8 * len(X_housing))
    X_train, X_test = X_housing[:n_train], X_housing[n_train:]
    y_train, y_test = y_housing[:n_train], y_housing[n_train:]
    
    # Fit model with normalization
    housing_model = MultipleLinearRegression(normalize=True, method='normal_equation')
    housing_model.fit(X_train, y_train, feature_names=housing_features)
    
    # Evaluate on test set
    train_metrics = housing_model.calculate_metrics(X_train, y_train)
    test_metrics = housing_model.calculate_metrics(X_test, y_test)
    
    print(f"Training R²: {train_metrics.r2:.6f}")
    print(f"Test R²: {test_metrics.r2:.6f}")
    print(f"Test RMSE: {test_metrics.rmse:.6f}")
    
    # Feature importance
    importance = housing_model.get_feature_importance()
    feature_importance = list(zip(housing_features, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Most Important Features:")
    for name, imp in feature_importance[:5]:
        print(f"  {name}: {imp:.6f}")
    
    # 6. Residual analysis
    print("\n6. Residual Analysis")
    print("-" * 50)
    
    residual_data = housing_model.residual_analysis(X_test, y_test)
    
    print(f"Mean residual: {np.mean(residual_data['residuals']):.6f}")
    print(f"Std residual: {np.std(residual_data['residuals']):.6f}")
    print(f"Max leverage: {np.max(residual_data['leverage']):.6f}")
    
    return {
        'simple_model': simple_lr,
        'multi_model': multi_lr,
        'housing_model': housing_model,
        'method_results': method_results,
        'housing_data': (X_housing, y_housing, housing_features)
    }

# Visualization functions
def plot_regression_results(model: LinearRegression, X: np.ndarray, y: np.ndarray, 
                          title: str = "Linear Regression Results") -> None:
    """Plot regression results for 1D or 2D data"""
    
    if X.shape[1] == 1:
        # 1D plot
        plt.figure(figsize=(10, 6))
        
        # Sort data for plotting line
        sort_indices = np.argsort(X[:, 0])
        X_sorted = X[sort_indices]
        y_sorted = y[sort_indices]
        
        # Plot data points
        plt.scatter(X[:, 0], y, alpha=0.6, label='Data')
        
        # Plot regression line
        y_pred = model.predict(X_sorted)
        plt.plot(X_sorted[:, 0], y_pred, 'r-', linewidth=2, label='Regression Line')
        
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics to plot
        metrics = model.calculate_metrics(X, y)
        plt.text(0.05, 0.95, f'R² = {metrics.r2:.3f}\nRMSE = {metrics.rmse:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
    
    else:
        print("Plotting only available for 1D features")

def plot_residuals(model: LinearRegression, X: np.ndarray, y: np.ndarray) -> None:
    """Plot residual analysis"""
    residual_data = model.residual_analysis(X, y)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(residual_data['predictions'], residual_data['residuals'], alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot (approximate)
    from scipy import stats
    stats.probplot(residual_data['standardized_residuals'], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    # Residuals histogram
    axes[1, 0].hist(residual_data['residuals'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Leverage plot
    axes[1, 1].scatter(range(len(residual_data['leverage'])), residual_data['leverage'], alpha=0.6)
    axes[1, 1].set_xlabel('Observation Index')
    axes[1, 1].set_ylabel('Leverage')
    axes[1, 1].set_title('Leverage Values')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run demonstrations
if __name__ == "__main__":
    demo_results = demonstrate_linear_regression()
    
    print("\n" + "="*60)
    print("Creating visualizations...")
    
    # Plot simple regression
    X_simple, y_simple = generate_regression_data(100, 1, 0.2, 42)
    simple_model = LinearRegression()
    simple_model.fit(X_simple, y_simple)
    plot_regression_results(simple_model, X_simple, y_simple, "Simple Linear Regression")
    
    # Plot residuals for housing model
    housing_model = demo_results['housing_model']
    X_housing, y_housing, _ = demo_results['housing_data']
    plot_residuals(housing_model, X_housing, y_housing)
```

This comprehensive linear regression implementation provides:

1. **Multiple solving methods**: Normal equation, SVD, QR decomposition, and gradient descent
2. **Regularization support**: Ridge and Lasso regularization options
3. **Comprehensive metrics**: R², adjusted R², AIC, BIC, RMSE, MAE
4. **Feature normalization**: Optional feature scaling
5. **Residual analysis**: Leverage calculation and diagnostic plots
6. **Real-world examples**: Housing price prediction with synthetic realistic data
7. **Visualization tools**: Regression line plots and residual analysis plots
8. **Educational features**: Method comparison and detailed model summaries

---

## Question 6

**Write a function that optimizes a given cost function using gradient descent.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, Dict, List, Any, Union
from dataclasses import dataclass
import warnings

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    x_optimal: np.ndarray
    f_optimal: float
    n_iterations: int
    converged: bool
    history: Dict[str, List]

class GradientDescentOptimizer:
    """
    Comprehensive gradient descent optimizer for general cost functions
    
    Supports multiple variants of gradient descent and adaptive methods
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 10000,
                 tolerance: float = 1e-8, method: str = 'standard',
                 momentum: float = 0.9, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Initialize gradient descent optimizer
        
        Args:
            learning_rate: Step size for parameter updates
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for gradient norm
            method: Optimization method ('standard', 'momentum', 'nesterov', 'adam', 'rmsprop', 'adagrad')
            momentum: Momentum factor for momentum-based methods
            beta1: Exponential decay rate for first moment (Adam)
            beta2: Exponential decay rate for second moment (Adam, RMSprop)
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.method = method
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Validate method
        valid_methods = ['standard', 'momentum', 'nesterov', 'adam', 'rmsprop', 'adagrad']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        # Initialize state variables
        self.reset_state()
    
    def reset_state(self):
        """Reset optimizer state"""
        self.velocity = None
        self.squared_gradients = None
        self.m = None  # First moment (Adam)
        self.v = None  # Second moment (Adam)
        self.t = 0     # Time step
    
    def _initialize_state(self, x: np.ndarray):
        """Initialize state variables based on parameter shape"""
        if self.method in ['momentum', 'nesterov']:
            self.velocity = np.zeros_like(x)
        elif self.method == 'adagrad':
            self.squared_gradients = np.zeros_like(x)
        elif self.method == 'rmsprop':
            self.squared_gradients = np.zeros_like(x)
        elif self.method == 'adam':
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
    
    def optimize(self, cost_function: Callable[[np.ndarray], float],
                gradient_function: Callable[[np.ndarray], np.ndarray],
                x_initial: np.ndarray,
                callback: Optional[Callable[[int, np.ndarray, float], None]] = None) -> OptimizationResult:
        """
        Optimize a cost function using gradient descent
        
        Args:
            cost_function: Function to minimize f(x) -> scalar
            gradient_function: Gradient function grad_f(x) -> vector
            x_initial: Initial parameter values
            callback: Optional callback function called each iteration
            
        Returns:
            OptimizationResult with optimization details
        """
        # Initialize
        x = np.array(x_initial, dtype=float)
        self.reset_state()
        self._initialize_state(x)
        
        # History tracking
        history = {
            'iteration': [],
            'x': [],
            'cost': [],
            'gradient_norm': [],
            'learning_rate': []
        }
        
        converged = False
        
        for iteration in range(self.max_iterations):
            # Evaluate cost and gradient
            cost = cost_function(x)
            gradient = gradient_function(x)
            gradient_norm = np.linalg.norm(gradient)
            
            # Store history
            history['iteration'].append(iteration)
            history['x'].append(x.copy())
            history['cost'].append(cost)
            history['gradient_norm'].append(gradient_norm)
            history['learning_rate'].append(self.learning_rate)
            
            # Check convergence
            if gradient_norm < self.tolerance:
                converged = True
                break
            
            # Apply optimization step
            x = self._optimization_step(x, gradient)
            
            # Call callback if provided
            if callback is not None:
                callback(iteration, x, cost)
        
        return OptimizationResult(
            x_optimal=x,
            f_optimal=cost_function(x),
            n_iterations=iteration + 1,
            converged=converged,
            history=history
        )
    
    def _optimization_step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Apply single optimization step based on method"""
        
        if self.method == 'standard':
            return x - self.learning_rate * gradient
        
        elif self.method == 'momentum':
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            return x + self.velocity
        
        elif self.method == 'nesterov':
            # Nesterov accelerated gradient
            x_lookahead = x + self.momentum * self.velocity
            # Note: This is a simplified version; full NAG requires gradient at lookahead point
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            return x + self.velocity
        
        elif self.method == 'adagrad':
            self.squared_gradients += gradient ** 2
            adapted_lr = self.learning_rate / (np.sqrt(self.squared_gradients) + self.epsilon)
            return x - adapted_lr * gradient
        
        elif self.method == 'rmsprop':
            self.squared_gradients = (self.beta2 * self.squared_gradients + 
                                    (1 - self.beta2) * gradient ** 2)
            adapted_lr = self.learning_rate / (np.sqrt(self.squared_gradients) + self.epsilon)
            return x - adapted_lr * gradient
        
        elif self.method == 'adam':
            self.t += 1
            
            # Update biased first and second moment estimates
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
            
            # Compute bias-corrected moment estimates
            m_corrected = self.m / (1 - self.beta1 ** self.t)
            v_corrected = self.v / (1 - self.beta2 ** self.t)
            
            # Update parameters
            return x - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")

class AdaptiveLearningRateOptimizer(GradientDescentOptimizer):
    """Gradient descent with adaptive learning rate strategies"""
    
    def __init__(self, lr_schedule: str = 'exponential', decay_rate: float = 0.95,
                 decay_steps: int = 100, **kwargs):
        """
        Initialize with learning rate scheduling
        
        Args:
            lr_schedule: Learning rate schedule ('exponential', 'step', 'cosine', 'polynomial')
            decay_rate: Decay rate for exponential/step decay
            decay_steps: Steps between decay for step schedule
        """
        super().__init__(**kwargs)
        self.lr_schedule = lr_schedule
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.initial_lr = self.learning_rate
    
    def _update_learning_rate(self, iteration: int):
        """Update learning rate based on schedule"""
        if self.lr_schedule == 'exponential':
            self.learning_rate = self.initial_lr * (self.decay_rate ** iteration)
        elif self.lr_schedule == 'step':
            self.learning_rate = self.initial_lr * (self.decay_rate ** (iteration // self.decay_steps))
        elif self.lr_schedule == 'cosine':
            self.learning_rate = self.initial_lr * 0.5 * (1 + np.cos(np.pi * iteration / self.max_iterations))
        elif self.lr_schedule == 'polynomial':
            self.learning_rate = self.initial_lr * (1 - iteration / self.max_iterations) ** 2
    
    def _optimization_step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Apply optimization step with learning rate update"""
        self._update_learning_rate(self.t if hasattr(self, 't') else 0)
        return super()._optimization_step(x, gradient)

# Numerical gradient computation
def numerical_gradient(func: Callable[[np.ndarray], float], x: np.ndarray, 
                      h: float = 1e-7) -> np.ndarray:
    """
    Compute numerical gradient using finite differences
    
    Args:
        func: Function to compute gradient for
        x: Point at which to compute gradient
        h: Step size for finite differences
        
    Returns:
        Numerical gradient
    """
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
    
    return grad

# Common cost functions and their gradients
class CostFunctions:
    """Collection of common cost functions with analytical gradients"""
    
    @staticmethod
    def quadratic(x: np.ndarray, A: Optional[np.ndarray] = None, 
                  b: Optional[np.ndarray] = None, c: float = 0) -> float:
        """
        Quadratic function: f(x) = 0.5 * x^T A x + b^T x + c
        
        Args:
            x: Input vector
            A: Quadratic coefficient matrix (default: identity)
            b: Linear coefficient vector (default: zero)
            c: Constant term
        """
        if A is None:
            A = np.eye(len(x))
        if b is None:
            b = np.zeros_like(x)
        
        return 0.5 * x.T @ A @ x + b.T @ x + c
    
    @staticmethod
    def quadratic_gradient(x: np.ndarray, A: Optional[np.ndarray] = None, 
                          b: Optional[np.ndarray] = None, c: float = 0) -> np.ndarray:
        """Gradient of quadratic function: grad_f(x) = A x + b"""
        if A is None:
            A = np.eye(len(x))
        if b is None:
            b = np.zeros_like(x)
        
        return A @ x + b
    
    @staticmethod
    def rosenbrock(x: np.ndarray, a: float = 1, b: float = 100) -> float:
        """
        Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
        
        Global minimum at (a, a^2) with value 0
        """
        if len(x) != 2:
            raise ValueError("Rosenbrock function is defined for 2D input")
        
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    @staticmethod
    def rosenbrock_gradient(x: np.ndarray, a: float = 1, b: float = 100) -> np.ndarray:
        """Gradient of Rosenbrock function"""
        if len(x) != 2:
            raise ValueError("Rosenbrock function is defined for 2D input")
        
        grad = np.zeros(2)
        grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
        grad[1] = 2 * b * (x[1] - x[0]**2)
        return grad
    
    @staticmethod
    def himmelblau(x: np.ndarray) -> float:
        """
        Himmelblau's function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
        
        Has four global minima
        """
        if len(x) != 2:
            raise ValueError("Himmelblau function is defined for 2D input")
        
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    
    @staticmethod
    def himmelblau_gradient(x: np.ndarray) -> np.ndarray:
        """Gradient of Himmelblau's function"""
        if len(x) != 2:
            raise ValueError("Himmelblau function is defined for 2D input")
        
        grad = np.zeros(2)
        grad[0] = 4 * x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7)
        grad[1] = 2 * (x[0]**2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1]**2 - 7)
        return grad

# Visualization utilities
class OptimizationVisualizer:
    """Visualization tools for optimization results"""
    
    @staticmethod
    def plot_convergence(results: List[OptimizationResult], labels: List[str],
                        figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot convergence comparison for multiple optimization runs"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Cost function convergence
        axes[0, 0].set_title('Cost Function Convergence')
        for result, label in zip(results, labels):
            axes[0, 0].semilogy(result.history['iteration'], result.history['cost'], 
                               label=label, marker='o', markersize=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Cost (log scale)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Gradient norm convergence
        axes[0, 1].set_title('Gradient Norm Convergence')
        for result, label in zip(results, labels):
            axes[0, 1].semilogy(result.history['iteration'], result.history['gradient_norm'], 
                               label=label, marker='o', markersize=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Gradient Norm (log scale)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Parameter evolution (first component)
        axes[1, 0].set_title('Parameter Evolution (x[0])')
        for result, label in zip(results, labels):
            x_values = [x[0] for x in result.history['x']]
            axes[1, 0].plot(result.history['iteration'], x_values, 
                           label=label, marker='o', markersize=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('x[0]')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate evolution
        axes[1, 1].set_title('Learning Rate Evolution')
        for result, label in zip(results, labels):
            axes[1, 1].plot(result.history['iteration'], result.history['learning_rate'], 
                           label=label, marker='o', markersize=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_2d_optimization_path(cost_func: Callable[[np.ndarray], float],
                                 results: List[OptimizationResult],
                                 labels: List[str],
                                 bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                                 figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot optimization paths on 2D cost function contour"""
        x_bounds, y_bounds = bounds
        
        # Create mesh for contour plot
        x = np.linspace(x_bounds[0], x_bounds[1], 100)
        y = np.linspace(y_bounds[0], y_bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate cost function
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = cost_func(np.array([X[i, j], Y[i, j]]))
        
        # Plot contours
        plt.figure(figsize=figsize)
        contour = plt.contour(X, Y, Z, levels=20, alpha=0.6)
        plt.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
        plt.colorbar(label='Cost Function Value')
        
        # Plot optimization paths
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (result, label) in enumerate(zip(results, labels)):
            if len(result.history['x']) > 0 and len(result.history['x'][0]) == 2:
                path_x = [x[0] for x in result.history['x']]
                path_y = [x[1] for x in result.history['x']]
                
                plt.plot(path_x, path_y, colors[i % len(colors)], 
                        linewidth=2, marker='o', markersize=3, label=label)
                
                # Mark start and end points
                plt.plot(path_x[0], path_y[0], colors[i % len(colors)], 
                        marker='s', markersize=8, label=f'{label} Start')
                plt.plot(path_x[-1], path_y[-1], colors[i % len(colors)], 
                        marker='*', markersize=10, label=f'{label} End')
        
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')
        plt.title('Optimization Paths on Cost Function Contour')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Demonstration and examples
def demonstrate_gradient_descent():
    """Comprehensive gradient descent demonstration"""
    
    print("=== Gradient Descent Optimization Demonstrations ===\n")
    
    # 1. Simple quadratic function
    print("1. Optimizing Simple Quadratic Function")
    print("-" * 50)
    
    # Define quadratic function f(x) = x^2 + 2x + 1 = (x+1)^2
    def simple_quadratic(x):
        return CostFunctions.quadratic(x, A=np.array([[2]]), b=np.array([2]), c=1)
    
    def simple_quadratic_grad(x):
        return CostFunctions.quadratic_gradient(x, A=np.array([[2]]), b=np.array([2]))
    
    # Test different methods
    methods = ['standard', 'momentum', 'adam']
    results = []
    
    for method in methods:
        optimizer = GradientDescentOptimizer(
            learning_rate=0.1, 
            max_iterations=100, 
            method=method
        )
        
        result = optimizer.optimize(
            simple_quadratic, 
            simple_quadratic_grad, 
            np.array([2.0])
        )
        
        results.append(result)
        
        print(f"{method.capitalize()} method:")
        print(f"  Optimal x: {result.x_optimal[0]:.6f}")
        print(f"  Optimal f(x): {result.f_optimal:.6f}")
        print(f"  Iterations: {result.n_iterations}")
        print(f"  Converged: {result.converged}")
    
    # 2. Rosenbrock function optimization
    print("\n2. Rosenbrock Function Optimization")
    print("-" * 50)
    
    # Test multiple starting points
    starting_points = [
        np.array([-1.2, 1.0]),  # Classic starting point
        np.array([0.0, 0.0]),   # Origin
        np.array([2.0, 2.0])    # Away from minimum
    ]
    
    rosenbrock_results = []
    
    for i, start_point in enumerate(starting_points):
        print(f"\nStarting from {start_point}:")
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.001,
            max_iterations=5000,
            method='adam'
        )
        
        result = optimizer.optimize(
            CostFunctions.rosenbrock,
            CostFunctions.rosenbrock_gradient,
            start_point
        )
        
        rosenbrock_results.append(result)
        
        print(f"  Final x: [{result.x_optimal[0]:.6f}, {result.x_optimal[1]:.6f}]")
        print(f"  Final f(x): {result.f_optimal:.6f}")
        print(f"  Iterations: {result.n_iterations}")
        print(f"  Converged: {result.converged}")
    
    # 3. Compare optimization methods on Himmelblau's function
    print("\n3. Method Comparison on Himmelblau's Function")
    print("-" * 50)
    
    methods_config = [
        ('standard', {'learning_rate': 0.01}),
        ('momentum', {'learning_rate': 0.01, 'momentum': 0.9}),
        ('adam', {'learning_rate': 0.01}),
        ('rmsprop', {'learning_rate': 0.01})
    ]
    
    himmel_results = []
    start_point = np.array([0.0, 0.0])
    
    for method, config in methods_config:
        optimizer = GradientDescentOptimizer(
            method=method,
            max_iterations=1000,
            **config
        )
        
        result = optimizer.optimize(
            CostFunctions.himmelblau,
            CostFunctions.himmelblau_gradient,
            start_point
        )
        
        himmel_results.append(result)
        
        print(f"{method.capitalize()}:")
        print(f"  Final x: [{result.x_optimal[0]:.6f}, {result.x_optimal[1]:.6f}]")
        print(f"  Final f(x): {result.f_optimal:.6f}")
        print(f"  Iterations: {result.n_iterations}")
    
    # 4. Adaptive learning rate demonstration
    print("\n4. Adaptive Learning Rate Demonstration")
    print("-" * 50)
    
    lr_schedules = ['exponential', 'step', 'cosine']
    adaptive_results = []
    
    for schedule in lr_schedules:
        optimizer = AdaptiveLearningRateOptimizer(
            initial_lr=0.1,
            lr_schedule=schedule,
            decay_rate=0.95,
            max_iterations=500,
            method='standard'
        )
        
        result = optimizer.optimize(
            CostFunctions.rosenbrock,
            CostFunctions.rosenbrock_gradient,
            np.array([-1.2, 1.0])
        )
        
        adaptive_results.append(result)
        
        print(f"{schedule.capitalize()} schedule:")
        print(f"  Final f(x): {result.f_optimal:.6f}")
        print(f"  Final LR: {result.history['learning_rate'][-1]:.6f}")
        print(f"  Iterations: {result.n_iterations}")
    
    # 5. Numerical vs analytical gradients
    print("\n5. Numerical vs Analytical Gradients")
    print("-" * 50)
    
    test_point = np.array([1.5, 2.0])
    
    # Analytical gradient
    analytical_grad = CostFunctions.rosenbrock_gradient(test_point)
    
    # Numerical gradient
    numerical_grad = numerical_gradient(CostFunctions.rosenbrock, test_point)
    
    print(f"Test point: {test_point}")
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Numerical gradient:  {numerical_grad}")
    print(f"Difference: {np.linalg.norm(analytical_grad - numerical_grad):.2e}")
    
    return {
        'quadratic_results': results,
        'rosenbrock_results': rosenbrock_results,
        'himmelblau_results': himmel_results,
        'adaptive_results': adaptive_results
    }

# Real-world application: Neural network parameter optimization
def neural_network_optimization_example():
    """Example: Optimizing simple neural network parameters"""
    
    print("\n=== Neural Network Parameter Optimization ===")
    print("-" * 60)
    
    # Generate synthetic data for binary classification
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)  # Simple linear decision boundary
    
    def sigmoid(z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def neural_network_cost(params):
        """Cost function for simple 2-layer neural network"""
        # Unpack parameters: [W1 (2x2), b1 (2,), W2 (2x1), b2 (1,)]
        W1 = params[:4].reshape(2, 2)
        b1 = params[4:6]
        W2 = params[6:8].reshape(2, 1)
        b2 = params[8]
        
        # Forward pass
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2).flatten()
        
        # Binary cross-entropy loss
        epsilon = 1e-15
        a2 = np.clip(a2, epsilon, 1 - epsilon)
        cost = -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))
        
        return cost
    
    def neural_network_gradient(params):
        """Gradient of neural network cost function"""
        # This is a simplified numerical gradient
        return numerical_gradient(neural_network_cost, params, h=1e-7)
    
    # Initialize parameters randomly
    np.random.seed(42)
    initial_params = np.random.randn(9) * 0.1
    
    print(f"Dataset: {n_samples} samples, 2 features")
    print(f"Network: 2 -> 2 -> 1 (sigmoid activations)")
    print(f"Parameters to optimize: {len(initial_params)}")
    
    # Optimize using different methods
    methods = ['adam', 'momentum', 'rmsprop']
    nn_results = []
    
    for method in methods:
        print(f"\nOptimizing with {method}...")
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.01 if method == 'adam' else 0.1,
            max_iterations=1000,
            method=method,
            tolerance=1e-6
        )
        
        result = optimizer.optimize(
            neural_network_cost,
            neural_network_gradient,
            initial_params
        )
        
        nn_results.append(result)
        
        print(f"  Initial cost: {neural_network_cost(initial_params):.6f}")
        print(f"  Final cost: {result.f_optimal:.6f}")
        print(f"  Improvement: {neural_network_cost(initial_params) - result.f_optimal:.6f}")
        print(f"  Iterations: {result.n_iterations}")
        print(f"  Converged: {result.converged}")
    
    return nn_results

# Run demonstrations
if __name__ == "__main__":
    # Main demonstration
    demo_results = demonstrate_gradient_descent()
    
    # Neural network example
    nn_results = neural_network_optimization_example()
    
    print("\n" + "="*60)
    print("Creating visualizations...")
    
    # Visualize convergence for Rosenbrock function
    visualizer = OptimizationVisualizer()
    
    if demo_results['rosenbrock_results']:
        visualizer.plot_convergence(
            demo_results['rosenbrock_results'][:2],  # First two results
            ['Starting at (-1.2, 1.0)', 'Starting at (0.0, 0.0)']
        )
        
        # Visualize optimization path on Rosenbrock function
        visualizer.plot_2d_optimization_path(
            CostFunctions.rosenbrock,
            demo_results['rosenbrock_results'][:2],
            ['Starting at (-1.2, 1.0)', 'Starting at (0.0, 0.0)'],
            bounds=((-2, 2), (-1, 3))
        )
    
    print("Gradient descent optimization demonstrations complete!")
```

This comprehensive gradient descent optimizer includes:

1. **Multiple optimization algorithms**: Standard, momentum, Nesterov, Adam, RMSprop, Adagrad
2. **Adaptive learning rates**: Exponential, step, cosine, and polynomial decay schedules
3. **Common test functions**: Quadratic, Rosenbrock, Himmelblau's function
4. **Numerical gradients**: For functions without analytical gradients
5. **Comprehensive visualization**: Convergence plots and 2D optimization paths
6. **Real-world example**: Neural network parameter optimization
7. **Production-ready features**: Error handling, convergence checking, detailed history tracking

---

## Question 7

**Use Pandas to read a CSV file, clean the data, and prepare it for analysis.**

**Answer:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import io

class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing toolkit for pandas DataFrames
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.cleaning_report = {
            'original_shape': None,
            'final_shape': None,
            'operations_performed': [],
            'columns_dropped': [],
            'rows_dropped': 0,
            'missing_values_handled': {},
            'outliers_removed': {},
            'data_types_converted': {},
            'duplicates_removed': 0
        }
        
        # Store transformations for potential reversal
        self.transformations = {
            'scalers': {},
            'encoders': {},
            'imputations': {}
        }
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats with automatic encoding detection
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Determine file type
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'csv':
                # Try different encodings if default fails
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                        if self.verbose:
                            print(f"Successfully loaded CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode file with any common encoding")
                    
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, **kwargs)
                
            elif file_extension == 'json':
                df = pd.read_json(file_path, **kwargs)
                
            elif file_extension in ['tsv', 'txt']:
                df = pd.read_csv(file_path, sep='\t', **kwargs)
                
            else:
                # Default to CSV
                df = pd.read_csv(file_path, **kwargs)
            
            self.cleaning_report['original_shape'] = df.shape
            
            if self.verbose:
                print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        analysis = {
            'basic_info': {},
            'missing_values': {},
            'data_types': {},
            'duplicates': {},
            'outliers': {},
            'categorical_analysis': {},
            'numerical_analysis': {}
        }
        
        # Basic information
        analysis['basic_info'] = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'total_cells': df.shape[0] * df.shape[1],
            'non_null_cells': df.count().sum(),
            'null_cells': df.isnull().sum().sum()
        }
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        analysis['missing_values'] = {
            'columns_with_missing': missing_data[missing_data > 0].to_dict(),
            'missing_percentages': missing_percent[missing_percent > 0].to_dict(),
            'total_missing': missing_data.sum(),
            'completely_missing_columns': missing_data[missing_data == len(df)].index.tolist()
        }
        
        # Data types
        analysis['data_types'] = df.dtypes.value_counts().to_dict()
        
        # Duplicates
        analysis['duplicates'] = {
            'total_duplicates': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Outlier analysis for numerical columns
        outlier_analysis = {}
        for col in numerical_cols:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_analysis[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
        
        analysis['outliers'] = outlier_analysis
        
        # Categorical analysis
        categorical_analysis = {}
        for col in categorical_cols:
            unique_values = df[col].nunique()
            categorical_analysis[col] = {
                'unique_count': unique_values,
                'unique_percentage': (unique_values / len(df)) * 100,
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            }
        
        analysis['categorical_analysis'] = categorical_analysis
        
        # Numerical analysis
        numerical_analysis = {}
        for col in numerical_cols:
            numerical_analysis[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
        
        analysis['numerical_analysis'] = numerical_analysis
        
        return analysis
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto', 
                            columns: Optional[List[str]] = None, 
                            custom_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Handle missing values with various strategies
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                     'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill', 
                     'backward_fill', 'interpolate', 'knn', 'custom'
            columns: Specific columns to handle (None for all)
            custom_values: Custom values for specific columns
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df.columns[df.isnull().any()].tolist()
        
        missing_before = df_clean.isnull().sum().sum()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            missing_count = df_clean[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'auto':
                # Auto-select strategy based on data type and missing percentage
                missing_pct = missing_count / len(df_clean)
                
                if missing_pct > 0.5:
                    # Drop column if >50% missing
                    df_clean = df_clean.drop(columns=[col])
                    self.cleaning_report['columns_dropped'].append(col)
                    continue
                elif df_clean[col].dtype in ['object', 'category']:
                    # Use mode for categorical
                    fill_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_value)
                else:
                    # Use median for numerical
                    fill_value = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                
            elif strategy == 'mean' and df_clean[col].dtype in ['int64', 'float64']:
                fill_value = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(fill_value)
                
            elif strategy == 'median' and df_clean[col].dtype in ['int64', 'float64']:
                fill_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_value)
                
            elif strategy == 'mode':
                fill_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_value)
                
            elif strategy == 'forward_fill':
                df_clean[col] = df_clean[col].fillna(method='ffill')
                
            elif strategy == 'backward_fill':
                df_clean[col] = df_clean[col].fillna(method='bfill')
                
            elif strategy == 'interpolate' and df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].interpolate()
                
            elif strategy == 'knn' and df_clean[col].dtype in ['int64', 'float64']:
                # Use KNN imputation
                numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 1:
                    imputer = KNNImputer(n_neighbors=5)
                    df_clean[numerical_cols] = imputer.fit_transform(df_clean[numerical_cols])
                    
            elif strategy == 'custom' and custom_values and col in custom_values:
                df_clean[col] = df_clean[col].fillna(custom_values[col])
            
            # Record the imputation
            if col in df_clean.columns:
                self.cleaning_report['missing_values_handled'][col] = {
                    'strategy': strategy,
                    'missing_before': missing_count,
                    'missing_after': df_clean[col].isnull().sum()
                }
        
        missing_after = df_clean.isnull().sum().sum()
        
        if self.verbose:
            print(f"Missing values handled: {missing_before} → {missing_after}")
        
        self.cleaning_report['operations_performed'].append('handle_missing_values')
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            DataFrame without duplicates
        """
        df_clean = df.copy()
        
        duplicates_before = df_clean.duplicated(subset=subset).sum()
        df_clean = df_clean.drop_duplicates(subset=subset, keep=keep)
        duplicates_removed = duplicates_before
        
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        self.cleaning_report['operations_performed'].append('remove_duplicates')
        
        if self.verbose:
            print(f"Duplicates removed: {duplicates_removed}")
        
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr',
                       columns: Optional[List[str]] = None,
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in numerical columns
        
        Args:
            df: Input DataFrame
            method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            columns: Columns to check for outliers
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_removed = {}
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            before_count = len(df_clean)
            
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                df_clean = df_clean[~outlier_mask]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                outlier_mask = z_scores > threshold
                # Apply mask to non-null values only
                non_null_indices = df_clean[col].dropna().index
                outlier_indices = non_null_indices[outlier_mask]
                df_clean = df_clean.drop(outlier_indices)
            
            after_count = len(df_clean)
            outliers_removed[col] = before_count - after_count
        
        self.cleaning_report['outliers_removed'] = outliers_removed
        self.cleaning_report['operations_performed'].append('handle_outliers')
        
        total_outliers = sum(outliers_removed.values())
        if self.verbose and total_outliers > 0:
            print(f"Outliers removed: {total_outliers} rows")
        
        return df_clean
    
    def convert_data_types(self, df: pd.DataFrame, 
                          type_mapping: Optional[Dict[str, str]] = None,
                          auto_convert: bool = True) -> pd.DataFrame:
        """
        Convert data types for better memory usage and analysis
        
        Args:
            df: Input DataFrame
            type_mapping: Manual type mapping {column: type}
            auto_convert: Whether to automatically optimize types
            
        Returns:
            DataFrame with optimized data types
        """
        df_clean = df.copy()
        conversions = {}
        
        if auto_convert:
            # Auto-convert numerical columns
            for col in df_clean.select_dtypes(include=['int64']).columns:
                if df_clean[col].min() >= 0:
                    if df_clean[col].max() <= 255:
                        df_clean[col] = df_clean[col].astype('uint8')
                        conversions[col] = 'int64 → uint8'
                    elif df_clean[col].max() <= 65535:
                        df_clean[col] = df_clean[col].astype('uint16')
                        conversions[col] = 'int64 → uint16'
                    elif df_clean[col].max() <= 4294967295:
                        df_clean[col] = df_clean[col].astype('uint32')
                        conversions[col] = 'int64 → uint32'
                else:
                    if df_clean[col].min() >= -128 and df_clean[col].max() <= 127:
                        df_clean[col] = df_clean[col].astype('int8')
                        conversions[col] = 'int64 → int8'
                    elif df_clean[col].min() >= -32768 and df_clean[col].max() <= 32767:
                        df_clean[col] = df_clean[col].astype('int16')
                        conversions[col] = 'int64 → int16'
                    elif df_clean[col].min() >= -2147483648 and df_clean[col].max() <= 2147483647:
                        df_clean[col] = df_clean[col].astype('int32')
                        conversions[col] = 'int64 → int32'
            
            # Convert object columns with few unique values to category
            for col in df_clean.select_dtypes(include=['object']).columns:
                unique_ratio = df_clean[col].nunique() / len(df_clean)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df_clean[col] = df_clean[col].astype('category')
                    conversions[col] = 'object → category'
        
        # Apply manual type mapping
        if type_mapping:
            for col, dtype in type_mapping.items():
                if col in df_clean.columns:
                    try:
                        original_type = str(df_clean[col].dtype)
                        df_clean[col] = df_clean[col].astype(dtype)
                        conversions[col] = f'{original_type} → {dtype}'
                    except Exception as e:
                        if self.verbose:
                            print(f"Could not convert {col} to {dtype}: {e}")
        
        self.cleaning_report['data_types_converted'] = conversions
        self.cleaning_report['operations_performed'].append('convert_data_types')
        
        if self.verbose and conversions:
            print("Data type conversions:")
            for col, conversion in conversions.items():
                print(f"  {col}: {conversion}")
        
        return df_clean
    
    def clean_text_columns(self, df: pd.DataFrame, 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean text columns by removing extra whitespace, fixing encoding issues, etc.
        
        Args:
            df: Input DataFrame
            columns: Text columns to clean
            
        Returns:
            DataFrame with cleaned text
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            # Remove leading/trailing whitespace
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # Remove extra internal whitespace
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
            
            # Fix common encoding issues
            df_clean[col] = df_clean[col].str.replace('â€™', "'", regex=False)
            df_clean[col] = df_clean[col].str.replace('â€œ', '"', regex=False)
            df_clean[col] = df_clean[col].str.replace('â€\x9d', '"', regex=False)
            
            # Convert back to NaN if result is empty string
            df_clean[col] = df_clean[col].replace('', np.nan)
            df_clean[col] = df_clean[col].replace('nan', np.nan)
        
        self.cleaning_report['operations_performed'].append('clean_text_columns')
        
        return df_clean
    
    def create_date_features(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """
        Extract useful features from date columns
        
        Args:
            df: Input DataFrame
            date_columns: List of date column names
            
        Returns:
            DataFrame with additional date features
        """
        df_clean = df.copy()
        
        for col in date_columns:
            if col not in df_clean.columns:
                continue
            
            try:
                # Convert to datetime if not already
                if df_clean[col].dtype != 'datetime64[ns]':
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                
                # Extract features
                df_clean[f'{col}_year'] = df_clean[col].dt.year
                df_clean[f'{col}_month'] = df_clean[col].dt.month
                df_clean[f'{col}_day'] = df_clean[col].dt.day
                df_clean[f'{col}_dayofweek'] = df_clean[col].dt.dayofweek
                df_clean[f'{col}_quarter'] = df_clean[col].dt.quarter
                df_clean[f'{col}_is_weekend'] = df_clean[col].dt.dayofweek >= 5
                
                # Calculate days since epoch for numerical analysis
                df_clean[f'{col}_days_since_epoch'] = (
                    df_clean[col] - pd.Timestamp('1970-01-01')
                ).dt.days
                
            except Exception as e:
                if self.verbose:
                    print(f"Could not process date column {col}: {e}")
        
        self.cleaning_report['operations_performed'].append('create_date_features')
        
        return df_clean
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None,
                                   method: str = 'auto') -> pd.DataFrame:
        """
        Encode categorical variables for machine learning
        
        Args:
            df: Input DataFrame
            columns: Columns to encode
            method: Encoding method ('auto', 'onehot', 'label', 'target')
            
        Returns:
            DataFrame with encoded variables
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            unique_count = df_clean[col].nunique()
            
            if method == 'auto':
                # Choose encoding method based on cardinality
                if unique_count <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
                    df_clean = pd.concat([df_clean, dummies], axis=1)
                    df_clean = df_clean.drop(columns=[col])
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
                    self.transformations['encoders'][col] = le
                    
            elif method == 'onehot':
                dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
                df_clean = pd.concat([df_clean, dummies], axis=1)
                df_clean = df_clean.drop(columns=[col])
                
            elif method == 'label':
                le = LabelEncoder()
                df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
                self.transformations['encoders'][col] = le
        
        self.cleaning_report['operations_performed'].append('encode_categorical_variables')
        
        return df_clean
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                               columns: Optional[List[str]] = None,
                               method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            df_clean[f'{col}_scaled'] = scaler.fit_transform(df_clean[[col]])
            self.transformations['scalers'][col] = scaler
        
        self.cleaning_report['operations_performed'].append('scale_numerical_features')
        
        return df_clean
    
    def complete_cleaning_pipeline(self, df: pd.DataFrame, 
                                 config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Complete data cleaning pipeline
        
        Args:
            df: Input DataFrame
            config: Configuration for cleaning steps
            
        Returns:
            Cleaned DataFrame
        """
        if config is None:
            config = {
                'remove_duplicates': True,
                'handle_missing': True,
                'missing_strategy': 'auto',
                'handle_outliers': True,
                'outlier_method': 'iqr',
                'convert_types': True,
                'clean_text': True,
                'create_date_features': False,
                'encode_categorical': False,
                'scale_numerical': False
            }
        
        df_clean = df.copy()
        
        if self.verbose:
            print(f"Starting cleaning pipeline for dataset: {df_clean.shape}")
        
        # Remove duplicates
        if config.get('remove_duplicates', True):
            df_clean = self.remove_duplicates(df_clean)
        
        # Handle missing values
        if config.get('handle_missing', True):
            df_clean = self.handle_missing_values(
                df_clean, 
                strategy=config.get('missing_strategy', 'auto')
            )
        
        # Handle outliers
        if config.get('handle_outliers', True):
            df_clean = self.handle_outliers(
                df_clean, 
                method=config.get('outlier_method', 'iqr')
            )
        
        # Convert data types
        if config.get('convert_types', True):
            df_clean = self.convert_data_types(df_clean)
        
        # Clean text columns
        if config.get('clean_text', True):
            df_clean = self.clean_text_columns(df_clean)
        
        # Create date features
        if config.get('create_date_features', False) and 'date_columns' in config:
            df_clean = self.create_date_features(df_clean, config['date_columns'])
        
        # Encode categorical variables
        if config.get('encode_categorical', False):
            df_clean = self.encode_categorical_variables(df_clean)
        
        # Scale numerical features
        if config.get('scale_numerical', False):
            df_clean = self.scale_numerical_features(df_clean)
        
        self.cleaning_report['final_shape'] = df_clean.shape
        self.cleaning_report['rows_dropped'] = (
            self.cleaning_report['original_shape'][0] - df_clean.shape[0]
        )
        
        if self.verbose:
            print(f"Cleaning complete: {df_clean.shape}")
            print(f"Rows removed: {self.cleaning_report['rows_dropped']}")
        
        return df_clean
    
    def generate_cleaning_report(self) -> str:
        """Generate a comprehensive cleaning report"""
        report = "=== DATA CLEANING REPORT ===\n\n"
        
        # Basic statistics
        report += f"Original shape: {self.cleaning_report['original_shape']}\n"
        report += f"Final shape: {self.cleaning_report['final_shape']}\n"
        report += f"Rows removed: {self.cleaning_report['rows_dropped']}\n"
        report += f"Data reduction: {(self.cleaning_report['rows_dropped'] / self.cleaning_report['original_shape'][0] * 100):.2f}%\n\n"
        
        # Operations performed
        report += "Operations performed:\n"
        for op in self.cleaning_report['operations_performed']:
            report += f"  - {op}\n"
        report += "\n"
        
        # Missing values
        if self.cleaning_report['missing_values_handled']:
            report += "Missing values handled:\n"
            for col, info in self.cleaning_report['missing_values_handled'].items():
                report += f"  - {col}: {info['missing_before']} → {info['missing_after']} ({info['strategy']})\n"
            report += "\n"
        
        # Outliers
        if self.cleaning_report['outliers_removed']:
            report += "Outliers removed:\n"
            for col, count in self.cleaning_report['outliers_removed'].items():
                report += f"  - {col}: {count} rows\n"
            report += "\n"
        
        # Data type conversions
        if self.cleaning_report['data_types_converted']:
            report += "Data type conversions:\n"
            for col, conversion in self.cleaning_report['data_types_converted'].items():
                report += f"  - {col}: {conversion}\n"
            report += "\n"
        
        # Duplicates
        if self.cleaning_report['duplicates_removed'] > 0:
            report += f"Duplicates removed: {self.cleaning_report['duplicates_removed']}\n\n"
        
        # Columns dropped
        if self.cleaning_report['columns_dropped']:
            report += "Columns dropped:\n"
            for col in self.cleaning_report['columns_dropped']:
                report += f"  - {col}\n"
        
        return report

# Utility functions for quick data operations
def quick_clean(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Quick cleaning function for common use cases
    
    Args:
        file_path: Path to data file
        **kwargs: Additional cleaning configuration
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner(verbose=kwargs.get('verbose', True))
    df = cleaner.load_data(file_path)
    
    # Default cleaning configuration
    config = {
        'remove_duplicates': True,
        'handle_missing': True,
        'missing_strategy': 'auto',
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'convert_types': True,
        'clean_text': True
    }
    
    # Update with user configuration
    config.update(kwargs)
    
    df_clean = cleaner.complete_cleaning_pipeline(df, config)
    
    if kwargs.get('show_report', False):
        print(cleaner.generate_cleaning_report())
    
    return df_clean

def create_sample_dataset(save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a sample dataset with various data quality issues for demonstration
    
    Args:
        save_path: Optional path to save the sample dataset
        
    Returns:
        Sample DataFrame with data quality issues
    """
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    
    data = {
        'id': range(1, n_samples + 1),
        'name': [f'Person_{i}' for i in range(1, n_samples + 1)],
        'age': np.random.normal(35, 12, n_samples),
        'salary': np.random.lognormal(10, 0.5, n_samples),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples),
        'join_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'email': [f'person{i}@company.com' for i in range(1, n_samples + 1)],
        'performance_score': np.random.uniform(1, 5, n_samples),
        'is_remote': np.random.choice([True, False], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    
    # Missing values
    missing_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[missing_indices, 'salary'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[missing_indices, 'department'] = np.nan
    
    # Outliers
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'age'] = np.random.uniform(100, 120, 10)
    
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'salary'] = np.random.uniform(1000000, 2000000, 15)
    
    # Duplicates
    duplicate_rows = df.sample(20).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Text issues
    text_issues_indices = np.random.choice(df.index, size=25, replace=False)
    df.loc[text_issues_indices, 'name'] = df.loc[text_issues_indices, 'name'] + '   '  # Extra whitespace
    
    # Invalid email formats
    invalid_email_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[invalid_email_indices, 'email'] = df.loc[invalid_email_indices, 'email'].str.replace('@', '_at_')
    
    # Inconsistent department names
    inconsistent_indices = np.random.choice(
        df[df['department'] == 'Engineering'].index, size=10, replace=False
    )
    df.loc[inconsistent_indices, 'department'] = 'Eng'
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Sample dataset saved to: {save_path}")
    
    return df

# Advanced analysis functions
def analyze_data_drift(df_original: pd.DataFrame, df_new: pd.DataFrame, 
                      numerical_threshold: float = 0.1,
                      categorical_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Analyze data drift between two datasets
    
    Args:
        df_original: Original dataset
        df_new: New dataset to compare
        numerical_threshold: Threshold for numerical drift detection
        categorical_threshold: Threshold for categorical drift detection
        
    Returns:
        Dictionary with drift analysis results
    """
    drift_report = {
        'numerical_drift': {},
        'categorical_drift': {},
        'new_columns': [],
        'missing_columns': [],
        'schema_changes': {}
    }
    
    # Schema comparison
    original_cols = set(df_original.columns)
    new_cols = set(df_new.columns)
    
    drift_report['new_columns'] = list(new_cols - original_cols)
    drift_report['missing_columns'] = list(original_cols - new_cols)
    
    common_cols = original_cols & new_cols
    
    # Data type changes
    for col in common_cols:
        original_type = str(df_original[col].dtype)
        new_type = str(df_new[col].dtype)
        if original_type != new_type:
            drift_report['schema_changes'][col] = {
                'original_type': original_type,
                'new_type': new_type
            }
    
    # Numerical drift analysis
    numerical_cols = df_original.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col in common_cols]
    
    for col in numerical_cols:
        original_mean = df_original[col].mean()
        new_mean = df_new[col].mean()
        original_std = df_original[col].std()
        new_std = df_new[col].std()
        
        mean_change = abs((new_mean - original_mean) / original_mean) if original_mean != 0 else 0
        std_change = abs((new_std - original_std) / original_std) if original_std != 0 else 0
        
        drift_report['numerical_drift'][col] = {
            'mean_change': mean_change,
            'std_change': std_change,
            'drift_detected': mean_change > numerical_threshold or std_change > numerical_threshold
        }
    
    # Categorical drift analysis
    categorical_cols = df_original.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col in common_cols]
    
    for col in categorical_cols:
        original_dist = df_original[col].value_counts(normalize=True)
        new_dist = df_new[col].value_counts(normalize=True)
        
        # Calculate distribution changes
        all_categories = set(original_dist.index) | set(new_dist.index)
        max_change = 0
        
        for category in all_categories:
            original_freq = original_dist.get(category, 0)
            new_freq = new_dist.get(category, 0)
            change = abs(new_freq - original_freq)
            max_change = max(max_change, change)
        
        drift_report['categorical_drift'][col] = {
            'max_frequency_change': max_change,
            'drift_detected': max_change > categorical_threshold
        }
    
    return drift_report

# Comprehensive demonstration
def demonstrate_data_cleaning():
    """Comprehensive demonstration of data cleaning capabilities"""
    
    print("=== Comprehensive Data Cleaning Demonstration ===\n")
    
    # Create sample dataset with issues
    print("1. Creating Sample Dataset with Data Quality Issues")
    print("-" * 60)
    
    sample_data = create_sample_dataset()
    print(f"Created dataset with shape: {sample_data.shape}")
    
    # Initial data quality analysis
    print("\n2. Initial Data Quality Analysis")
    print("-" * 60)
    
    cleaner = DataCleaner(verbose=True)
    quality_analysis = cleaner.analyze_data_quality(sample_data)
    
    print(f"Missing values: {quality_analysis['missing_values']['total_missing']}")
    print(f"Duplicates: {quality_analysis['duplicates']['total_duplicates']}")
    print(f"Memory usage: {quality_analysis['basic_info']['memory_usage_mb']:.2f} MB")
    
    # Apply cleaning pipeline
    print("\n3. Applying Comprehensive Cleaning Pipeline")
    print("-" * 60)
    
    cleaned_data = cleaner.complete_cleaning_pipeline(sample_data)
    
    # Generate and display cleaning report
    print("\n4. Cleaning Report")
    print("-" * 60)
    print(cleaner.generate_cleaning_report())
    
    # Before vs After comparison
    print("\n5. Before vs After Comparison")
    print("-" * 60)
    
    print("BEFORE CLEANING:")
    print(f"  Shape: {sample_data.shape}")
    print(f"  Missing values: {sample_data.isnull().sum().sum()}")
    print(f"  Duplicates: {sample_data.duplicated().sum()}")
    print(f"  Memory usage: {sample_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nAFTER CLEANING:")
    print(f"  Shape: {cleaned_data.shape}")
    print(f"  Missing values: {cleaned_data.isnull().sum().sum()}")
    print(f"  Duplicates: {cleaned_data.duplicated().sum()}")
    print(f"  Memory usage: {cleaned_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Demonstrate specific cleaning functions
    print("\n6. Specific Cleaning Function Demonstrations")
    print("-" * 60)
    
    # Missing value handling strategies
    test_data = sample_data.copy()
    
    print("Missing value strategies comparison:")
    strategies = ['mean', 'median', 'mode', 'auto']
    for strategy in strategies:
        test_cleaner = DataCleaner(verbose=False)
        cleaned_test = test_cleaner.handle_missing_values(test_data, strategy=strategy)
        remaining_missing = cleaned_test.isnull().sum().sum()
        print(f"  {strategy}: {remaining_missing} missing values remaining")
    
    # Outlier handling methods
    print("\nOutlier handling methods comparison:")
    methods = ['iqr', 'zscore']
    for method in methods:
        test_cleaner = DataCleaner(verbose=False)
        cleaned_test = test_cleaner.handle_outliers(test_data, method=method)
        rows_removed = len(test_data) - len(cleaned_test)
        print(f"  {method}: {rows_removed} rows removed")
    
    return {
        'original_data': sample_data,
        'cleaned_data': cleaned_data,
        'cleaner': cleaner,
        'quality_analysis': quality_analysis
    }

# Example with real-world-like dataset
def process_sales_data_example():
    """Example: Processing sales data"""
    
    print("=== Sales Data Processing Example ===\n")
    
    # Create realistic sales dataset
    np.random.seed(42)
    n_records = 5000
    
    # Generate date range
    start_date = pd.Timestamp('2022-01-01')
    end_date = pd.Timestamp('2023-12-31')
    dates = pd.date_range(start_date, end_date, freq='D')
    
    sales_data = []
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    regions = ['North', 'South', 'East', 'West']
    sales_reps = [f'Rep_{i}' for i in range(1, 21)]
    
    for _ in range(n_records):
        record = {
            'date': np.random.choice(dates),
            'product': np.random.choice(products),
            'region': np.random.choice(regions),
            'sales_rep': np.random.choice(sales_reps),
            'quantity': np.random.poisson(10) + 1,
            'unit_price': np.random.uniform(10, 100),
            'customer_satisfaction': np.random.uniform(1, 5),
            'discount_percent': np.random.uniform(0, 0.3)
        }
        
        # Calculate derived fields
        record['revenue'] = record['quantity'] * record['unit_price'] * (1 - record['discount_percent'])
        
        sales_data.append(record)
    
    df_sales = pd.DataFrame(sales_data)
    
    # Introduce some issues
    # Missing values
    missing_indices = np.random.choice(df_sales.index, size=200, replace=False)
    df_sales.loc[missing_indices, 'customer_satisfaction'] = np.nan
    
    # Invalid data
    invalid_indices = np.random.choice(df_sales.index, size=50, replace=False)
    df_sales.loc[invalid_indices, 'quantity'] = -1  # Negative quantities
    
    # Outliers
    outlier_indices = np.random.choice(df_sales.index, size=30, replace=False)
    df_sales.loc[outlier_indices, 'unit_price'] = np.random.uniform(1000, 5000, 30)
    
    print(f"Generated sales dataset: {df_sales.shape}")
    print(f"Date range: {df_sales['date'].min()} to {df_sales['date'].max()}")
    
    # Clean the data
    cleaner = DataCleaner(verbose=True)
    
    # Custom cleaning configuration for sales data
    config = {
        'remove_duplicates': True,
        'handle_missing': True,
        'missing_strategy': 'auto',
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'convert_types': True,
        'clean_text': True,
        'create_date_features': True,
        'date_columns': ['date']
    }
    
    df_sales_clean = cleaner.complete_cleaning_pipeline(df_sales, config)
    
    # Remove invalid records (negative quantities)
    df_sales_clean = df_sales_clean[df_sales_clean['quantity'] > 0]
    
    # Add business logic
    df_sales_clean['profit_margin'] = (
        df_sales_clean['revenue'] - df_sales_clean['quantity'] * df_sales_clean['unit_price'] * 0.6
    ) / df_sales_clean['revenue']
    
    print(f"\nCleaned sales dataset: {df_sales_clean.shape}")
    print(f"Revenue range: ${df_sales_clean['revenue'].min():.2f} - ${df_sales_clean['revenue'].max():.2f}")
    print(f"Average profit margin: {df_sales_clean['profit_margin'].mean():.2%}")
    
    # Generate summary statistics
    print("\nSales Summary by Region:")
    print(df_sales_clean.groupby('region')['revenue'].agg(['count', 'sum', 'mean']).round(2))
    
    print("\nTop Products by Revenue:")
    print(df_sales_clean.groupby('product')['revenue'].sum().sort_values(ascending=False))
    
    return df_sales_clean

# Run demonstrations
if __name__ == "__main__":
    # Main demonstration
    demo_results = demonstrate_data_cleaning()
    
    print("\n" + "="*80)
    
    # Sales data example
    sales_data = process_sales_data_example()
    
    print("\n=== Data Cleaning Demonstrations Complete ===")
```

This comprehensive implementation provides:

1. **Complete DataCleaner class** with modular cleaning methods
2. **Automated data quality analysis** with detailed metrics
3. **Multiple missing value strategies** (auto, mean, median, mode, KNN, etc.)
4. **Outlier detection and handling** using IQR and Z-score methods
5. **Data type optimization** for memory efficiency
6. **Text cleaning** and encoding fixes
7. **Date feature engineering** for temporal analysis
8. **Categorical encoding** (one-hot, label encoding)
9. **Numerical scaling** (standard, min-max, robust)
10. **Comprehensive reporting** of all cleaning operations
11. **Data drift analysis** for comparing datasets over time
12. **Real-world examples** including sales data processing
13. **Production-ready features** with error handling and logging

---

## Question 8

**Implement a decision tree from scratch in Python.**

**Answer:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod
import graphviz
import pickle

@dataclass
class NodeInfo:
    """Information stored in each node of the decision tree"""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    value: Optional[Union[int, float]] = None
    samples: int = 0
    impurity: float = 0.0
    class_distribution: Optional[Dict] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    is_leaf: bool = False
    depth: int = 0

class ImpurityMeasure(ABC):
    """Abstract base class for impurity measures"""
    
    @abstractmethod
    def calculate(self, y: np.ndarray) -> float:
        """Calculate impurity for given labels"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return name of the impurity measure"""
        pass

class GiniImpurity(ImpurityMeasure):
    """Gini impurity calculation"""
    
    def calculate(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity
        
        Gini = 1 - Σ(p_i)² where p_i is probability of class i
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        
        return gini
    
    def name(self) -> str:
        return "gini"

class Entropy(ImpurityMeasure):
    """Entropy calculation for information gain"""
    
    def calculate(self, y: np.ndarray) -> float:
        """
        Calculate entropy
        
        Entropy = -Σ(p_i * log2(p_i)) where p_i is probability of class i
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Avoid log(0) by filtering out zero probabilities
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
    
    def name(self) -> str:
        return "entropy"

class VarianceReduction(ImpurityMeasure):
    """Variance for regression trees"""
    
    def calculate(self, y: np.ndarray) -> float:
        """Calculate variance"""
        if len(y) == 0:
            return 0.0
        
        return np.var(y)
    
    def name(self) -> str:
        return "variance"

class Node:
    """Decision tree node"""
    
    def __init__(self, info: NodeInfo):
        self.info = info
        self.left = None
        self.right = None
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return self.info.is_leaf
    
    def predict_sample(self, x: np.ndarray) -> Union[int, float]:
        """Predict a single sample"""
        if self.is_leaf():
            return self.info.value
        
        if x[self.info.feature_idx] <= self.info.threshold:
            return self.left.predict_sample(x)
        else:
            return self.right.predict_sample(x)
    
    def get_rules(self, feature_names: Optional[List[str]] = None, 
                  rules: Optional[List[str]] = None, depth: int = 0) -> List[str]:
        """Extract decision rules from the tree"""
        if rules is None:
            rules = []
        
        if self.is_leaf():
            rule = "  " * depth + f"-> Predict: {self.info.value}"
            if self.info.class_distribution:
                rule += f" (samples: {self.info.samples}, distribution: {self.info.class_distribution})"
            rules.append(rule)
            return rules
        
        feature_name = feature_names[self.info.feature_idx] if feature_names else f"feature_{self.info.feature_idx}"
        
        # Left branch (<=)
        rules.append("  " * depth + f"if {feature_name} <= {self.info.threshold:.3f}:")
        self.left.get_rules(feature_names, rules, depth + 1)
        
        # Right branch (>)
        rules.append("  " * depth + f"else:  # {feature_name} > {self.info.threshold:.3f}")
        self.right.get_rules(feature_names, rules, depth + 1)
        
        return rules

class DecisionTree:
    """
    Decision Tree implementation for both classification and regression
    """
    
    def __init__(self, 
                 task_type: str = 'classification',
                 impurity_measure: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_impurity_decrease: float = 0.0,
                 max_features: Optional[Union[int, float, str]] = None,
                 random_state: Optional[int] = None):
        """
        Initialize Decision Tree
        
        Args:
            task_type: 'classification' or 'regression'
            impurity_measure: 'gini', 'entropy' (classification) or 'variance' (regression)
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            min_impurity_decrease: Minimum impurity decrease required for split
            max_features: Number of features to consider for best split
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        
        # Set impurity measure
        if task_type == 'classification':
            if impurity_measure == 'gini':
                self.impurity_calculator = GiniImpurity()
            elif impurity_measure == 'entropy':
                self.impurity_calculator = Entropy()
            else:
                raise ValueError("For classification, use 'gini' or 'entropy'")
        elif task_type == 'regression':
            self.impurity_calculator = VarianceReduction()
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        # Tree attributes
        self.root = None
        self.feature_importances_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.tree_depth_ = 0
        self.n_nodes_ = 0
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    def _calculate_feature_subset(self, n_features: int) -> np.ndarray:
        """Calculate subset of features to consider for splitting"""
        if self.max_features is None:
            return np.arange(n_features)
        
        if isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            max_features = max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            max_features = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            max_features = max(1, int(np.log2(n_features)))
        else:
            max_features = n_features
        
        return np.random.choice(n_features, size=max_features, replace=False)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the best split for the given data
        
        Returns:
            (best_feature_idx, best_threshold, best_impurity_decrease)
        """
        best_feature_idx = None
        best_threshold = None
        best_impurity_decrease = 0.0
        
        current_impurity = self.impurity_calculator.calculate(y)
        n_samples = len(y)
        
        # Get subset of features to consider
        feature_indices = self._calculate_feature_subset(X.shape[1])
        
        for feature_idx in feature_indices:
            # Get unique values for this feature and sort them
            unique_values = np.unique(X[:, feature_idx])
            
            # Consider thresholds between unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                # Check minimum samples constraint
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity after split
                left_impurity = self.impurity_calculator.calculate(y[left_mask])
                right_impurity = self.impurity_calculator.calculate(y[right_mask])
                
                weighted_impurity = (n_left / n_samples) * left_impurity + \
                                  (n_right / n_samples) * right_impurity
                
                # Calculate impurity decrease
                impurity_decrease = current_impurity - weighted_impurity
                
                # Update best split if this is better
                if impurity_decrease > best_impurity_decrease:
                    best_impurity_decrease = impurity_decrease
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_impurity_decrease
    
    def _create_leaf(self, y: np.ndarray, depth: int) -> Node:
        """Create a leaf node"""
        if self.task_type == 'classification':
            # Most common class
            unique_classes, counts = np.unique(y, return_counts=True)
            most_common_idx = np.argmax(counts)
            value = unique_classes[most_common_idx]
            
            # Class distribution
            class_dist = dict(zip(unique_classes, counts))
        else:
            # Mean for regression
            value = np.mean(y)
            class_dist = None
        
        info = NodeInfo(
            value=value,
            samples=len(y),
            impurity=self.impurity_calculator.calculate(y),
            class_distribution=class_dist,
            is_leaf=True,
            depth=depth
        )
        
        return Node(info)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Update tree statistics
        self.n_nodes_ += 1
        self.tree_depth_ = max(self.tree_depth_, depth)
        
        # Check stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (len(np.unique(y)) == 1):  # Pure node
            return self._create_leaf(y, depth)
        
        # Find best split
        best_feature_idx, best_threshold, best_impurity_decrease = self._find_best_split(X, y)
        
        # Check if split improves impurity enough
        if best_feature_idx is None or best_impurity_decrease < self.min_impurity_decrease:
            return self._create_leaf(y, depth)
        
        # Create internal node
        if self.task_type == 'classification':
            unique_classes, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique_classes, counts))
        else:
            class_dist = None
        
        info = NodeInfo(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            samples=n_samples,
            impurity=self.impurity_calculator.calculate(y),
            class_distribution=class_dist,
            is_leaf=False,
            depth=depth
        )
        
        node = Node(info)
        
        # Split data
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate feature importances based on impurity decrease"""
        importances = np.zeros(self.n_features_)
        
        def traverse(node: Node, n_samples_root: int):
            if node.is_leaf():
                return
            
            # Calculate importance for this split
            n_samples = node.info.samples
            impurity_decrease = node.info.impurity - \
                              (node.left.info.samples / n_samples) * node.left.info.impurity - \
                              (node.right.info.samples / n_samples) * node.right.info.impurity
            
            importance = (n_samples / n_samples_root) * impurity_decrease
            importances[node.info.feature_idx] += importance
            
            # Recursively calculate for children
            traverse(node.left, n_samples_root)
            traverse(node.right, n_samples_root)
        
        traverse(self.root, len(y))
        
        # Normalize importances
        total_importance = np.sum(importances)
        if total_importance > 0:
            importances = importances / total_importance
        
        self.feature_importances_ = importances
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Fit the decision tree
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            self: Fitted decision tree
        """
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Store dataset information
        self.n_features_ = X.shape[1]
        
        if self.task_type == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        
        # Reset tree statistics
        self.tree_depth_ = 0
        self.n_nodes_ = 0
        
        # Build the tree
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances
        self._calculate_feature_importances(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.root is None:
            raise ValueError("Tree must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = np.array([self.root.predict_sample(x) for x in X])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)"""
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification")
        
        if self.root is None:
            raise ValueError("Tree must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        def get_leaf_probabilities(node: Node, x: np.ndarray) -> Dict:
            """Get class probabilities from leaf node"""
            if node.is_leaf():
                return node.info.class_distribution
            
            if x[node.info.feature_idx] <= node.info.threshold:
                return get_leaf_probabilities(node.left, x)
            else:
                return get_leaf_probabilities(node.right, x)
        
        probabilities = []
        for x in X:
            leaf_dist = get_leaf_probabilities(self.root, x)
            total_samples = sum(leaf_dist.values())
            
            # Create probability vector for all classes
            prob_vector = np.zeros(self.n_classes_)
            for i, class_label in enumerate(self.classes_):
                prob_vector[i] = leaf_dist.get(class_label, 0) / total_samples
            
            probabilities.append(prob_vector)
        
        return np.array(probabilities)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy (classification) or R² score (regression)"""
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            return np.mean(predictions == y)
        else:
            # R² score for regression
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def get_depth(self) -> int:
        """Get the depth of the tree"""
        return self.tree_depth_
    
    def get_n_nodes(self) -> int:
        """Get the number of nodes in the tree"""
        return self.n_nodes_
    
    def get_rules(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """Extract decision rules from the tree"""
        if self.root is None:
            return []
        
        return self.root.get_rules(feature_names)
    
    def print_tree(self, feature_names: Optional[List[str]] = None) -> None:
        """Print the decision tree rules"""
        rules = self.get_rules(feature_names)
        print("Decision Tree Rules:")
        print("-" * 50)
        for rule in rules:
            print(rule)

class RandomDecisionTree(DecisionTree):
    """Decision Tree with additional randomization for ensemble methods"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bootstrap_sample = True
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_indices: Optional[np.ndarray] = None) -> 'RandomDecisionTree':
        """Fit with optional bootstrap sampling"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Use bootstrap sample if indices provided
        if sample_indices is not None:
            X = X[sample_indices]
            y = y[sample_indices]
        elif self.bootstrap_sample:
            # Create bootstrap sample
            n_samples = len(y)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X = X[indices]
            y = y[indices]
        
        return super().fit(X, y)

# Visualization utilities
class TreeVisualizer:
    """Visualization utilities for decision trees"""
    
    @staticmethod
    def plot_feature_importances(tree: DecisionTree, feature_names: Optional[List[str]] = None,
                               max_features: int = 20, figsize: Tuple[int, int] = (10, 6)):
        """Plot feature importances"""
        if tree.feature_importances_ is None:
            print("Tree must be fitted to plot feature importances")
            return
        
        # Get feature names
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(tree.feature_importances_))]
        
        # Sort features by importance
        indices = np.argsort(tree.feature_importances_)[::-1]
        
        # Limit to top features
        indices = indices[:max_features]
        importances = tree.feature_importances_[indices]
        names = [feature_names[i] for i in indices]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances)
        plt.xticks(range(len(importances)), names, rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_tree_structure(tree: DecisionTree, feature_names: Optional[List[str]] = None,
                          max_depth: int = 3, figsize: Tuple[int, int] = (15, 10)):
        """Plot tree structure (simplified visualization)"""
        if tree.root is None:
            print("Tree must be fitted to plot structure")
            return
        
        def plot_node(node: Node, x: float, y: float, width: float, ax, depth: int = 0):
            """Recursively plot nodes"""
            if depth > max_depth:
                return
            
            # Node appearance
            if node.is_leaf():
                bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7)
                text = f"Leaf\nValue: {node.info.value:.3f}\nSamples: {node.info.samples}"
            else:
                bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7)
                feature_name = feature_names[node.info.feature_idx] if feature_names else f"X[{node.info.feature_idx}]"
                text = f"{feature_name} <= {node.info.threshold:.3f}\nSamples: {node.info.samples}\nImpurity: {node.info.impurity:.3f}"
            
            # Draw node
            ax.text(x, y, text, ha='center', va='center', bbox=bbox_props, fontsize=8)
            
            # Draw connections to children
            if not node.is_leaf() and depth < max_depth:
                child_width = width / 2
                left_x = x - width / 4
                right_x = x + width / 4
                child_y = y - 0.2
                
                # Draw lines
                ax.plot([x, left_x], [y - 0.05, child_y + 0.05], 'k-', alpha=0.6)
                ax.plot([x, right_x], [y - 0.05, child_y + 0.05], 'k-', alpha=0.6)
                
                # Add labels
                ax.text(x - width/8, y - 0.1, 'True', ha='center', fontsize=7, color='blue')
                ax.text(x + width/8, y - 0.1, 'False', ha='center', fontsize=7, color='red')
                
                # Recursively draw children
                plot_node(node.left, left_x, child_y, child_width, ax, depth + 1)
                plot_node(node.right, right_x, child_y, child_width, ax, depth + 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        
        # Plot tree starting from root
        plot_node(tree.root, 0, 0.8, 1.5, ax)
        
        plt.title(f'Decision Tree Structure (max depth {max_depth} shown)')
        plt.tight_layout()
        plt.show()

# Comprehensive demonstration
def demonstrate_decision_trees():
    """Comprehensive demonstration of decision tree implementation"""
    
    print("=== Decision Tree Implementation Demonstration ===\n")
    
    # 1. Classification example
    print("1. Classification Example - Iris Dataset")
    print("-" * 50)
    
    from sklearn.datasets import load_iris, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train decision tree
    clf_tree = DecisionTree(
        task_type='classification',
        impurity_measure='gini',
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    clf_tree.fit(X_train, y_train)
    
    # Make predictions
    train_pred = clf_tree.predict(X_train)
    test_pred = clf_tree.predict(X_test)
    
    # Evaluate
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Testing accuracy: {test_acc:.4f}")
    print(f"Tree depth: {clf_tree.get_depth()}")
    print(f"Number of nodes: {clf_tree.get_n_nodes()}")
    
    print("\nFeature Importances:")
    for i, importance in enumerate(clf_tree.feature_importances_):
        print(f"  {feature_names[i]}: {importance:.4f}")
    
    # Print some decision rules
    print("\nDecision Rules (first few):")
    rules = clf_tree.get_rules(feature_names)
    for rule in rules[:10]:  # Show first 10 rules
        print(rule)
    
    # 2. Regression example
    print("\n2. Regression Example")
    print("-" * 50)
    
    # Generate regression data
    np.random.seed(42)
    n_samples = 200
    X_reg = np.random.uniform(-3, 3, (n_samples, 2))
    y_reg = (X_reg[:, 0] ** 2 + X_reg[:, 1] ** 2 + 
             0.5 * X_reg[:, 0] * X_reg[:, 1] + 
             0.1 * np.random.normal(0, 1, n_samples))
    
    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    # Train regression tree
    reg_tree = DecisionTree(
        task_type='regression',
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    reg_tree.fit(X_train_reg, y_train_reg)
    
    # Make predictions
    train_pred_reg = reg_tree.predict(X_train_reg)
    test_pred_reg = reg_tree.predict(X_test_reg)
    
    # Evaluate
    from sklearn.metrics import mean_squared_error, r2_score
    
    train_mse = mean_squared_error(y_train_reg, train_pred_reg)
    test_mse = mean_squared_error(y_test_reg, test_pred_reg)
    train_r2 = r2_score(y_train_reg, train_pred_reg)
    test_r2 = r2_score(y_test_reg, test_pred_reg)
    
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Testing MSE: {test_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Tree depth: {reg_tree.get_depth()}")
    print(f"Number of nodes: {reg_tree.get_n_nodes()}")
    
    # 3. Compare different impurity measures
    print("\n3. Comparison of Impurity Measures")
    print("-" * 50)
    
    impurity_measures = ['gini', 'entropy']
    results = {}
    
    for measure in impurity_measures:
        tree = DecisionTree(
            task_type='classification',
            impurity_measure=measure,
            max_depth=5,
            random_state=42
        )
        tree.fit(X_train, y_train)
        test_pred = tree.predict(X_test)
        accuracy = accuracy_score(y_test, test_pred)
        
        results[measure] = {
            'accuracy': accuracy,
            'depth': tree.get_depth(),
            'nodes': tree.get_n_nodes()
        }
        
        print(f"{measure.capitalize()}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Depth: {tree.get_depth()}")
        print(f"  Nodes: {tree.get_n_nodes()}")
    
    # 4. Effect of hyperparameters
    print("\n4. Effect of Hyperparameters")
    print("-" * 50)
    
    # Test different max_depth values
    depths = [2, 3, 5, 7, 10, None]
    depth_results = []
    
    for depth in depths:
        tree = DecisionTree(
            task_type='classification',
            max_depth=depth,
            random_state=42
        )
        tree.fit(X_train, y_train)
        
        train_acc = tree.score(X_train, y_train)
        test_acc = tree.score(X_test, y_test)
        
        depth_results.append({
            'max_depth': depth,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'tree_depth': tree.get_depth(),
            'n_nodes': tree.get_n_nodes()
        })
        
        print(f"Max depth {depth}: Train={train_acc:.4f}, Test={test_acc:.4f}, "
              f"Actual depth={tree.get_depth()}, Nodes={tree.get_n_nodes()}")
    
    # 5. Overfitting demonstration
    print("\n5. Overfitting Analysis")
    print("-" * 50)
    
    # Create a more complex dataset
    X_complex, y_complex = make_classification(
        n_samples=500, n_features=10, n_informative=5, n_redundant=2,
        n_clusters_per_class=2, random_state=42
    )
    
    X_train_complex, X_test_complex, y_train_complex, y_test_complex = train_test_split(
        X_complex, y_complex, test_size=0.3, random_state=42
    )
    
    # Train trees with different constraints
    configurations = [
        {'name': 'No constraints', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'name': 'Max depth 5', 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'name': 'Min samples split 20', 'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 1},
        {'name': 'Min samples leaf 10', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 10},
        {'name': 'Combined constraints', 'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 5}
    ]
    
    for config in configurations:
        name = config.pop('name')
        tree = DecisionTree(task_type='classification', random_state=42, **config)
        tree.fit(X_train_complex, y_train_complex)
        
        train_acc = tree.score(X_train_complex, y_train_complex)
        test_acc = tree.score(X_test_complex, y_test_complex)
        
        print(f"{name}:")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        print(f"  Overfitting: {train_acc - test_acc:.4f}")
        print(f"  Tree depth: {tree.get_depth()}")
        print(f"  Nodes: {tree.get_n_nodes()}")
        print()
    
    return {
        'classification_tree': clf_tree,
        'regression_tree': reg_tree,
        'iris_data': (X_test, y_test, feature_names, class_names),
        'regression_data': (X_test_reg, y_test_reg),
        'depth_results': depth_results
    }

# Example of using decision tree for feature selection
def decision_tree_feature_selection(X: np.ndarray, y: np.ndarray, 
                                  feature_names: Optional[List[str]] = None,
                                  top_k: int = 10) -> List[int]:
    """Use decision tree feature importances for feature selection"""
    
    # Train decision tree
    tree = DecisionTree(task_type='classification', random_state=42)
    tree.fit(X, y)
    
    # Get feature importances
    importances = tree.feature_importances_
    
    # Get top k features
    top_indices = np.argsort(importances)[::-1][:top_k]
    
    if feature_names:
        print("Top features by importance:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return top_indices.tolist()

# Model persistence
def save_tree(tree: DecisionTree, filename: str) -> None:
    """Save decision tree to file"""
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)

def load_tree(filename: str) -> DecisionTree:
    """Load decision tree from file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Run demonstration
if __name__ == "__main__":
    results = demonstrate_decision_trees()
    print("\n=== Decision Tree Demonstration Complete ===")
    
    # Optional: Show feature importance plot if matplotlib is available
    try:
        TreeVisualizer.plot_feature_importances(
            results['classification_tree'], 
            results['iris_data'][2]
        )
    except Exception as e:
        print(f"Could not plot feature importances: {e}")
```

This comprehensive implementation provides:

1. **Complete Decision Tree class** with support for both classification and regression
2. **Multiple impurity measures**: Gini, Entropy (Information Gain), and Variance
3. **Flexible hyperparameters**: max_depth, min_samples_split, min_samples_leaf, etc.
4. **Feature importance calculation** based on impurity decrease
5. **Decision rules extraction** for interpretability
6. **Probability predictions** for classification tasks
7. **Tree visualization utilities** for understanding structure
8. **Comprehensive demonstrations** showing different use cases
9. **Overfitting analysis** with various constraint configurations
10. **Feature selection capabilities** using tree-based importance
11. **Model persistence** for saving/loading trained trees
12. **Random tree variant** for ensemble methods preparation
13. **Production-ready features** with proper error handling and type hints

---

## Question 9

**Write a Python function to split a dataset into training and testing sets.**

**Answer:**

```python
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union, Dict, Any
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class DataSplitter:
    """
    Comprehensive data splitting utility for machine learning workflows
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize DataSplitter
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def train_test_split(self, 
                        X: Union[np.ndarray, pd.DataFrame], 
                        y: Union[np.ndarray, pd.Series] = None,
                        test_size: Union[float, int] = 0.2,
                        train_size: Optional[Union[float, int]] = None,
                        stratify: Union[np.ndarray, pd.Series] = None,
                        shuffle: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Split arrays or matrices into random train and test subsets
        
        Args:
            X: Features array/dataframe
            y: Target array/series (optional)
            test_size: Proportion or absolute number of test samples
            train_size: Proportion or absolute number of train samples
            stratify: Array to stratify by (maintains class distribution)
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or (X_train, X_test) if y is None
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        
        n_samples = len(X)
        
        # Validate inputs
        if y is not None and len(y) != n_samples:
            raise ValueError("X and y must have the same number of samples")
        
        # Calculate split sizes
        if isinstance(test_size, float):
            if not 0.0 < test_size < 1.0:
                raise ValueError("test_size must be between 0 and 1 when float")
            n_test = int(test_size * n_samples)
        else:
            n_test = int(test_size)
            if n_test >= n_samples:
                raise ValueError("test_size cannot be larger than dataset size")
        
        if train_size is not None:
            if isinstance(train_size, float):
                n_train = int(train_size * n_samples)
            else:
                n_train = int(train_size)
            
            if n_train + n_test > n_samples:
                raise ValueError("train_size + test_size cannot exceed dataset size")
        else:
            n_train = n_samples - n_test
        
        # Create indices
        indices = np.arange(n_samples)
        
        if stratify is not None:
            # Stratified split
            return self._stratified_split(X, y, indices, n_train, n_test, stratify, shuffle)
        else:
            # Random split
            if shuffle:
                np.random.shuffle(indices)
            
            train_indices = indices[:n_train]
            test_indices = indices[n_train:n_train + n_test]
            
            X_train, X_test = X[train_indices], X[test_indices]
            
            if y is not None:
                y_train, y_test = y[train_indices], y[test_indices]
                return X_train, X_test, y_train, y_test
            else:
                return X_train, X_test
    
    def _stratified_split(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                         n_train: int, n_test: int, stratify: np.ndarray, 
                         shuffle: bool) -> Tuple[np.ndarray, ...]:
        """Perform stratified split maintaining class distribution"""
        
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(stratify, return_counts=True)
        
        train_indices = []
        test_indices = []
        
        for class_label, class_count in zip(unique_classes, class_counts):
            # Get indices for this class
            class_indices = indices[stratify == class_label]
            
            if shuffle:
                np.random.shuffle(class_indices)
            
            # Calculate number of samples for this class
            class_test_size = int(n_test * class_count / len(stratify))
            class_train_size = class_count - class_test_size
            
            # Ensure we don't exceed available samples
            class_test_size = min(class_test_size, class_count)
            class_train_size = min(class_train_size, class_count - class_test_size)
            
            # Split class indices
            train_indices.extend(class_indices[:class_train_size])
            test_indices.extend(class_indices[class_train_size:class_train_size + class_test_size])
        
        # Convert to numpy arrays and shuffle
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
        
        X_train, X_test = X[train_indices], X[test_indices]
        
        if y is not None:
            y_train, y_test = y[train_indices], y[test_indices]
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test
    
    def train_validation_test_split(self, 
                                  X: Union[np.ndarray, pd.DataFrame],
                                  y: Union[np.ndarray, pd.Series] = None,
                                  test_size: Union[float, int] = 0.2,
                                  val_size: Union[float, int] = 0.2,
                                  stratify: Union[np.ndarray, pd.Series] = None,
                                  shuffle: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            X: Features array/dataframe
            y: Target array/series (optional)
            test_size: Proportion or absolute number of test samples
            val_size: Proportion or absolute number of validation samples
            stratify: Array to stratify by
            shuffle: Whether to shuffle data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test) or 
            (X_train, X_val, X_test) if y is None
        """
        # First split: separate test set
        if y is not None:
            X_temp, X_test, y_temp, y_test = self.train_test_split(
                X, y, test_size=test_size, stratify=stratify, shuffle=shuffle
            )
        else:
            X_temp, X_test = self.train_test_split(
                X, test_size=test_size, shuffle=shuffle
            )
            y_temp = None
        
        # Calculate validation size relative to remaining data
        if isinstance(val_size, float):
            remaining_val_size = val_size / (1 - test_size)
        else:
            remaining_val_size = val_size / len(X_temp)
        
        # Second split: separate train and validation
        if y_temp is not None:
            X_train, X_val, y_train, y_val = self.train_test_split(
                X_temp, y_temp, test_size=remaining_val_size, 
                stratify=y_temp if stratify is not None else None, 
                shuffle=shuffle
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_train, X_val = self.train_test_split(
                X_temp, test_size=remaining_val_size, shuffle=shuffle
            )
            return X_train, X_val, X_test
    
    def k_fold_split(self, 
                     X: Union[np.ndarray, pd.DataFrame],
                     y: Union[np.ndarray, pd.Series] = None,
                     n_folds: int = 5,
                     shuffle: bool = True,
                     stratify: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate K-Fold cross-validation splits
        
        Args:
            X: Features array/dataframe
            y: Target array/series (optional)
            n_folds: Number of folds
            shuffle: Whether to shuffle data before folding
            stratify: Whether to maintain class distribution (for classification)
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        if stratify and y is not None:
            return self._stratified_k_fold(indices, y, n_folds)
        else:
            return self._regular_k_fold(indices, n_folds)
    
    def _regular_k_fold(self, indices: np.ndarray, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Regular K-fold split"""
        n_samples = len(indices)
        fold_size = n_samples // n_folds
        
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else n_samples
            
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            folds.append((train_indices, test_indices))
        
        return folds
    
    def _stratified_k_fold(self, indices: np.ndarray, y: np.ndarray, 
                          n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Stratified K-fold split"""
        unique_classes = np.unique(y)
        class_indices = {cls: indices[y[indices] == cls] for cls in unique_classes}
        
        # Split each class into folds
        class_folds = {}
        for cls, cls_indices in class_indices.items():
            np.random.shuffle(cls_indices)
            class_folds[cls] = np.array_split(cls_indices, n_folds)
        
        # Combine folds across classes
        folds = []
        for fold_idx in range(n_folds):
            test_indices = []
            train_indices = []
            
            for cls in unique_classes:
                # Test indices for this fold
                test_indices.extend(class_folds[cls][fold_idx])
                
                # Train indices from other folds
                for other_fold_idx in range(n_folds):
                    if other_fold_idx != fold_idx:
                        train_indices.extend(class_folds[cls][other_fold_idx])
            
            folds.append((np.array(train_indices), np.array(test_indices)))
        
        return folds
    
    def time_series_split(self, 
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series] = None,
                         n_splits: int = 5,
                         test_size: Optional[int] = None,
                         gap: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Time series cross-validation split
        
        Args:
            X: Features array/dataframe
            y: Target array/series (optional)
            n_splits: Number of splits
            test_size: Size of test set for each split
            gap: Gap between train and test sets
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if test_size is None:
            test_size = n_samples // (n_splits + 1)
        
        splits = []
        for i in range(n_splits):
            # Calculate split points
            test_start = n_samples - test_size * (n_splits - i)
            test_end = test_start + test_size
            train_end = test_start - gap
            
            # Ensure valid indices
            if train_end <= 0 or test_start >= n_samples:
                continue
            
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits
    
    def group_split(self, 
                   X: Union[np.ndarray, pd.DataFrame],
                   y: Union[np.ndarray, pd.Series],
                   groups: Union[np.ndarray, pd.Series],
                   test_size: Union[float, int] = 0.2,
                   shuffle: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Split based on groups (ensures no group appears in both train and test)
        
        Args:
            X: Features array/dataframe
            y: Target array/series
            groups: Group labels for each sample
            test_size: Proportion or number of groups for test set
            shuffle: Whether to shuffle groups before splitting
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)
        
        # Get unique groups
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if shuffle:
            np.random.shuffle(unique_groups)
        
        # Calculate number of test groups
        if isinstance(test_size, float):
            n_test_groups = max(1, int(test_size * n_groups))
        else:
            n_test_groups = min(test_size, n_groups - 1)
        
        # Split groups
        test_groups = unique_groups[:n_test_groups]
        train_groups = unique_groups[n_test_groups:]
        
        # Get indices for each split
        test_indices = np.isin(groups, test_groups)
        train_indices = np.isin(groups, train_groups)
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def analyze_split_distribution(self, 
                                  y_train: np.ndarray, 
                                  y_test: np.ndarray,
                                  y_val: Optional[np.ndarray] = None,
                                  plot: bool = True) -> Dict[str, Any]:
        """
        Analyze the distribution of classes across splits
        
        Args:
            y_train: Training labels
            y_test: Test labels
            y_val: Validation labels (optional)
            plot: Whether to create visualization
            
        Returns:
            Dictionary with distribution statistics
        """
        analysis = {}
        
        # Count distributions
        train_dist = Counter(y_train)
        test_dist = Counter(y_test)
        
        analysis['train_distribution'] = train_dist
        analysis['test_distribution'] = test_dist
        analysis['train_size'] = len(y_train)
        analysis['test_size'] = len(y_test)
        
        if y_val is not None:
            val_dist = Counter(y_val)
            analysis['val_distribution'] = val_dist
            analysis['val_size'] = len(y_val)
        
        # Calculate proportions
        all_classes = set(train_dist.keys()) | set(test_dist.keys())
        if y_val is not None:
            all_classes |= set(val_dist.keys())
        
        proportions = {}
        for cls in all_classes:
            train_prop = train_dist.get(cls, 0) / len(y_train)
            test_prop = test_dist.get(cls, 0) / len(y_test)
            
            proportions[cls] = {
                'train': train_prop,
                'test': test_prop
            }
            
            if y_val is not None:
                val_prop = val_dist.get(cls, 0) / len(y_val)
                proportions[cls]['val'] = val_prop
        
        analysis['proportions'] = proportions
        
        # Calculate distribution differences
        max_diff = 0
        for cls in all_classes:
            train_prop = proportions[cls]['train']
            test_prop = proportions[cls]['test']
            diff = abs(train_prop - test_prop)
            max_diff = max(max_diff, diff)
        
        analysis['max_proportion_difference'] = max_diff
        analysis['is_balanced'] = max_diff < 0.05  # Consider balanced if diff < 5%
        
        # Visualization
        if plot and len(all_classes) <= 20:  # Only plot if not too many classes
            self._plot_distribution_comparison(analysis, y_val is not None)
        
        return analysis
    
    def _plot_distribution_comparison(self, analysis: Dict[str, Any], has_val: bool):
        """Plot distribution comparison across splits"""
        proportions = analysis['proportions']
        classes = list(proportions.keys())
        
        # Prepare data for plotting
        train_props = [proportions[cls]['train'] for cls in classes]
        test_props = [proportions[cls]['test'] for cls in classes]
        
        if has_val:
            val_props = [proportions[cls]['val'] for cls in classes]
            x = np.arange(len(classes))
            width = 0.25
            
            plt.figure(figsize=(12, 6))
            plt.bar(x - width, train_props, width, label='Train', alpha=0.8)
            plt.bar(x, val_props, width, label='Validation', alpha=0.8)
            plt.bar(x + width, test_props, width, label='Test', alpha=0.8)
        else:
            x = np.arange(len(classes))
            width = 0.35
            
            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, train_props, width, label='Train', alpha=0.8)
            plt.bar(x + width/2, test_props, width, label='Test', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Proportion')
        plt.title('Class Distribution Across Splits')
        plt.xticks(x, [str(cls) for cls in classes])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Convenience functions for common use cases
def simple_train_test_split(X: Union[np.ndarray, pd.DataFrame], 
                           y: Union[np.ndarray, pd.Series],
                           test_size: float = 0.2,
                           random_state: Optional[int] = None,
                           stratify: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Simple train-test split function
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test data
        random_state: Random seed
        stratify: Whether to stratify by target
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    splitter = DataSplitter(random_state=random_state)
    return splitter.train_test_split(
        X, y, test_size=test_size, 
        stratify=y if stratify else None
    )

def train_val_test_split(X: Union[np.ndarray, pd.DataFrame],
                        y: Union[np.ndarray, pd.Series],
                        test_size: float = 0.2,
                        val_size: float = 0.2,
                        random_state: Optional[int] = None,
                        stratify: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Three-way split into train, validation, and test sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test data
        val_size: Proportion of validation data
        random_state: Random seed
        stratify: Whether to stratify splits
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    splitter = DataSplitter(random_state=random_state)
    return splitter.train_validation_test_split(
        X, y, test_size=test_size, val_size=val_size,
        stratify=y if stratify else None
    )

def create_cv_splits(X: Union[np.ndarray, pd.DataFrame],
                    y: Union[np.ndarray, pd.Series],
                    cv_type: str = 'kfold',
                    n_splits: int = 5,
                    random_state: Optional[int] = None,
                    **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits
    
    Args:
        X: Features
        y: Target
        cv_type: Type of CV ('kfold', 'stratified', 'timeseries', 'group')
        n_splits: Number of splits
        random_state: Random seed
        **kwargs: Additional arguments for specific CV types
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    splitter = DataSplitter(random_state=random_state)
    
    if cv_type == 'kfold':
        return splitter.k_fold_split(X, y, n_folds=n_splits, stratify=False)
    elif cv_type == 'stratified':
        return splitter.k_fold_split(X, y, n_folds=n_splits, stratify=True)
    elif cv_type == 'timeseries':
        return splitter.time_series_split(X, y, n_splits=n_splits, **kwargs)
    elif cv_type == 'group':
        if 'groups' not in kwargs:
            raise ValueError("groups parameter required for group CV")
        # Note: This would need additional implementation for group K-fold
        raise NotImplementedError("Group K-fold not implemented in this example")
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")

# Comprehensive demonstration
def demonstrate_data_splitting():
    """Comprehensive demonstration of data splitting techniques"""
    
    print("=== Data Splitting Demonstration ===\n")
    
    # Generate sample data
    from sklearn.datasets import make_classification, make_regression
    
    # 1. Basic train-test split
    print("1. Basic Train-Test Split")
    print("-" * 40)
    
    X_class, y_class = make_classification(
        n_samples=1000, n_features=10, n_classes=3, 
        n_informative=8, n_redundant=2, random_state=42
    )
    
    splitter = DataSplitter(random_state=42)
    X_train, X_test, y_train, y_test = splitter.train_test_split(
        X_class, y_class, test_size=0.2
    )
    
    print(f"Original dataset: {X_class.shape[0]} samples")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X_class.shape[0]:.1%})")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X_class.shape[0]:.1%})")
    
    # Analyze distribution
    analysis = splitter.analyze_split_distribution(y_train, y_test, plot=False)
    print(f"Class distribution balanced: {analysis['is_balanced']}")
    print(f"Max proportion difference: {analysis['max_proportion_difference']:.3f}")
    
    # 2. Stratified split
    print("\n2. Stratified Train-Test Split")
    print("-" * 40)
    
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = splitter.train_test_split(
        X_class, y_class, test_size=0.2, stratify=y_class
    )
    
    analysis_strat = splitter.analyze_split_distribution(
        y_train_strat, y_test_strat, plot=False
    )
    print(f"Stratified - Class distribution balanced: {analysis_strat['is_balanced']}")
    print(f"Stratified - Max proportion difference: {analysis_strat['max_proportion_difference']:.3f}")
    
    # Compare regular vs stratified
    print("\nComparison:")
    print(f"Regular split max diff: {analysis['max_proportion_difference']:.3f}")
    print(f"Stratified split max diff: {analysis_strat['max_proportion_difference']:.3f}")
    
    # 3. Three-way split
    print("\n3. Train-Validation-Test Split")
    print("-" * 40)
    
    X_train_3, X_val_3, X_test_3, y_train_3, y_val_3, y_test_3 = splitter.train_validation_test_split(
        X_class, y_class, test_size=0.2, val_size=0.2, stratify=y_class
    )
    
    print(f"Training set: {X_train_3.shape[0]} samples ({X_train_3.shape[0]/X_class.shape[0]:.1%})")
    print(f"Validation set: {X_val_3.shape[0]} samples ({X_val_3.shape[0]/X_class.shape[0]:.1%})")
    print(f"Test set: {X_test_3.shape[0]} samples ({X_test_3.shape[0]/X_class.shape[0]:.1%})")
    
    # 4. K-Fold Cross Validation
    print("\n4. K-Fold Cross Validation")
    print("-" * 40)
    
    # Regular K-fold
    kfold_splits = splitter.k_fold_split(X_class, y_class, n_folds=5, stratify=False)
    print(f"Regular K-Fold: {len(kfold_splits)} folds")
    
    for i, (train_idx, test_idx) in enumerate(kfold_splits):
        print(f"  Fold {i+1}: Train {len(train_idx)}, Test {len(test_idx)}")
    
    # Stratified K-fold
    stratified_splits = splitter.k_fold_split(X_class, y_class, n_folds=5, stratify=True)
    print(f"\nStratified K-Fold: {len(stratified_splits)} folds")
    
    # Check class distribution consistency
    for i, (train_idx, test_idx) in enumerate(stratified_splits):
        train_dist = Counter(y_class[train_idx])
        test_dist = Counter(y_class[test_idx])
        print(f"  Fold {i+1}: Train {dict(train_dist)}, Test {dict(test_dist)}")
    
    # 5. Time Series Split
    print("\n5. Time Series Cross Validation")
    print("-" * 40)
    
    # Generate time series data
    n_samples = 100
    X_ts = np.random.randn(n_samples, 5)
    y_ts = np.cumsum(np.random.randn(n_samples)) + 0.1 * np.arange(n_samples)
    
    ts_splits = splitter.time_series_split(X_ts, y_ts, n_splits=5)
    print(f"Time Series CV: {len(ts_splits)} splits")
    
    for i, (train_idx, test_idx) in enumerate(ts_splits):
        print(f"  Split {i+1}: Train [{train_idx[0]}:{train_idx[-1]+1}], "
              f"Test [{test_idx[0]}:{test_idx[-1]+1}]")
    
    # 6. Group-based split
    print("\n6. Group-based Split")
    print("-" * 40)
    
    # Create groups (e.g., different patients, customers, etc.)
    n_groups = 20
    groups = np.random.randint(0, n_groups, size=len(X_class))
    
    X_train_grp, X_test_grp, y_train_grp, y_test_grp = splitter.group_split(
        X_class, y_class, groups, test_size=0.3
    )
    
    train_groups = set(groups[np.isin(np.arange(len(groups)), 
                                    np.where(np.isin(X_class, X_train_grp, assume_unique=False).all(axis=1))[0])])
    test_groups = set(groups[np.isin(np.arange(len(groups)), 
                                   np.where(np.isin(X_class, X_test_grp, assume_unique=False).all(axis=1))[0])])
    
    print(f"Total groups: {len(np.unique(groups))}")
    print(f"Training groups: {len(train_groups)}")
    print(f"Test groups: {len(test_groups)}")
    print(f"Group overlap: {len(train_groups & test_groups)} (should be 0)")
    
    # 7. Demonstrate class imbalance handling
    print("\n7. Handling Class Imbalance")
    print("-" * 40)
    
    # Create imbalanced dataset
    X_imbal, y_imbal = make_classification(
        n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
        n_informative=5, random_state=42
    )
    
    print(f"Original class distribution: {Counter(y_imbal)}")
    
    # Regular split
    X_tr_reg, X_te_reg, y_tr_reg, y_te_reg = splitter.train_test_split(
        X_imbal, y_imbal, test_size=0.2
    )
    
    # Stratified split
    X_tr_strat, X_te_strat, y_tr_strat, y_te_strat = splitter.train_test_split(
        X_imbal, y_imbal, test_size=0.2, stratify=y_imbal
    )
    
    print("Regular split:")
    print(f"  Train: {Counter(y_tr_reg)}")
    print(f"  Test: {Counter(y_te_reg)}")
    
    print("Stratified split:")
    print(f"  Train: {Counter(y_tr_strat)}")
    print(f"  Test: {Counter(y_te_strat)}")
    
    return {
        'basic_split': (X_train, X_test, y_train, y_test),
        'stratified_split': (X_train_strat, X_test_strat, y_train_strat, y_test_strat),
        'three_way_split': (X_train_3, X_val_3, X_test_3, y_train_3, y_val_3, y_test_3),
        'kfold_splits': kfold_splits,
        'ts_splits': ts_splits,
        'splitter': splitter
    }

# Performance comparison with sklearn
def compare_with_sklearn():
    """Compare custom implementation with sklearn"""
    print("\n=== Comparison with sklearn ===")
    
    from sklearn.model_selection import train_test_split as sklearn_split
    from sklearn.datasets import make_classification
    import time
    
    # Generate test data
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    
    # Custom implementation
    splitter = DataSplitter(random_state=42)
    
    start_time = time.time()
    X_train_custom, X_test_custom, y_train_custom, y_test_custom = splitter.train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    custom_time = time.time() - start_time
    
    # sklearn implementation
    start_time = time.time()
    X_train_sklearn, X_test_sklearn, y_train_sklearn, y_test_sklearn = sklearn_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    sklearn_time = time.time() - start_time
    
    print(f"Custom implementation time: {custom_time:.4f} seconds")
    print(f"sklearn implementation time: {sklearn_time:.4f} seconds")
    print(f"Speed ratio: {custom_time/sklearn_time:.2f}x")
    
    # Check if results are similar (they won't be identical due to different RNG)
    print(f"Custom split shapes: Train {X_train_custom.shape}, Test {X_test_custom.shape}")
    print(f"sklearn split shapes: Train {X_train_sklearn.shape}, Test {X_test_sklearn.shape}")
    
    # Check class distributions
    custom_train_dist = Counter(y_train_custom)
    custom_test_dist = Counter(y_test_custom)
    sklearn_train_dist = Counter(y_train_sklearn)
    sklearn_test_dist = Counter(y_test_sklearn)
    
    print(f"Custom train distribution: {custom_train_dist}")
    print(f"sklearn train distribution: {sklearn_train_dist}")

# Run demonstrations
if __name__ == "__main__":
    # Main demonstration
    results = demonstrate_data_splitting()
    
    # Performance comparison
    compare_with_sklearn()
    
    print("\n=== Data Splitting Demonstration Complete ===")
```

This comprehensive implementation provides:

1. **Complete DataSplitter class** with multiple splitting strategies
2. **Basic train-test split** with stratification support
3. **Three-way splitting** (train/validation/test)
4. **K-Fold cross-validation** (regular and stratified)
5. **Time series cross-validation** for temporal data
6. **Group-based splitting** to prevent data leakage
7. **Distribution analysis** to check split quality
8. **Class imbalance handling** with stratification
9. **Comprehensive demonstrations** showing all use cases
10. **Performance comparison** with sklearn
11. **Visualization tools** for analyzing split distributions
12. **Convenience functions** for common scenarios
13. **Production-ready features** with proper error handling

---

## Question 10

**Develop a Python script that automates the process of hyperparameter tuning using grid search.**

**Answer:**

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from itertools import product
import time
import pickle
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import warnings
from abc import ABC, abstractmethod

@dataclass
class ValidationResult:
    """Container for validation results"""
    score: float
    scores: List[float] = field(default_factory=list)
    fit_time: float = 0.0
    score_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    fold_results: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class GridSearchResult:
    """Container for grid search results"""
    best_score: float
    best_params: Dict[str, Any]
    best_estimator: Any
    cv_results: Dict[str, List[Any]]
    total_time: float
    n_combinations: int

class BaseValidator(ABC):
    """Abstract base class for validation strategies"""
    
    @abstractmethod
    def split(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """Generate train/test splits"""
        pass
    
    @abstractmethod
    def get_n_splits(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> int:
        """Get number of splits"""
        pass

class KFoldValidator(BaseValidator):
    """K-Fold cross-validation"""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def split(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """Generate K-Fold splits"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> int:
        return self.n_splits

class StratifiedKFoldValidator(BaseValidator):
    """Stratified K-Fold cross-validation"""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def split(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """Generate stratified K-Fold splits"""
        unique_classes = np.unique(y)
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
        
        # Shuffle class indices
        if self.shuffle:
            for cls_indices in class_indices.values():
                np.random.shuffle(cls_indices)
        
        # Split each class into folds
        class_folds = {}
        for cls, cls_indices in class_indices.items():
            class_folds[cls] = np.array_split(cls_indices, self.n_splits)
        
        # Combine folds across classes
        for fold_idx in range(self.n_splits):
            test_indices = []
            train_indices = []
            
            for cls in unique_classes:
                # Test indices for this fold
                test_indices.extend(class_folds[cls][fold_idx])
                
                # Train indices from other folds
                for other_fold_idx in range(self.n_splits):
                    if other_fold_idx != fold_idx:
                        train_indices.extend(class_folds[cls][other_fold_idx])
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> int:
        return self.n_splits

class TimeSeriesValidator(BaseValidator):
    """Time series cross-validation"""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """Generate time series splits"""
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        for i in range(self.n_splits):
            # Calculate split points
            test_start = n_samples - test_size * (self.n_splits - i)
            test_end = test_start + test_size
            train_end = test_start - self.gap
            
            # Ensure valid indices
            if train_end <= 0 or test_start >= n_samples:
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> int:
        return self.n_splits

class GridSearchCV:
    """
    Comprehensive grid search cross-validation implementation
    """
    
    def __init__(self, 
                 estimator: Any,
                 param_grid: Union[Dict[str, List], List[Dict[str, List]]],
                 scoring: Union[str, Callable] = 'accuracy',
                 cv: Union[int, BaseValidator] = 5,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 refit: bool = True,
                 return_train_score: bool = False,
                 random_state: Optional[int] = None):
        """
        Initialize GridSearchCV
        
        Args:
            estimator: Machine learning estimator
            param_grid: Dictionary or list of dictionaries with parameter values
            scoring: Scoring function or string
            cv: Cross-validation strategy or number of folds
            n_jobs: Number of parallel jobs (not implemented)
            verbose: Verbosity level
            refit: Whether to refit best estimator on full dataset
            return_train_score: Whether to return training scores
            random_state: Random seed
        """
        self.estimator = estimator
        self.param_grid = param_grid if isinstance(param_grid, list) else [param_grid]
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit
        self.return_train_score = return_train_score
        self.random_state = random_state
        
        # Results storage
        self.cv_results_ = {}
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_index_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_scorer(self) -> Callable:
        """Get scoring function"""
        if isinstance(self.scoring, str):
            if self.scoring == 'accuracy':
                return lambda y_true, y_pred: np.mean(y_true == y_pred)
            elif self.scoring == 'mse':
                return lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
            elif self.scoring == 'mae':
                return lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
            elif self.scoring == 'r2':
                def r2_score(y_true, y_pred):
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                return r2_score
            elif self.scoring == 'f1':
                def f1_score(y_true, y_pred):
                    # Binary F1 score
                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    fn = np.sum((y_true == 1) & (y_pred == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                return f1_score
            else:
                raise ValueError(f"Unknown scoring metric: {self.scoring}")
        else:
            return self.scoring
    
    def _get_cv_splitter(self, X: np.ndarray, y: np.ndarray) -> BaseValidator:
        """Get cross-validation splitter"""
        if isinstance(self.cv, int):
            # Determine if classification or regression
            unique_y = np.unique(y)
            if len(unique_y) <= 20 and np.all(unique_y == unique_y.astype(int)):
                # Classification
                return StratifiedKFoldValidator(n_splits=self.cv, random_state=self.random_state)
            else:
                # Regression
                return KFoldValidator(n_splits=self.cv, random_state=self.random_state)
        else:
            return self.cv
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        combinations = []
        
        for grid in self.param_grid:
            # Get all parameter names and their values
            param_names = list(grid.keys())
            param_values = [grid[name] for name in param_names]
            
            # Generate cartesian product
            for combination in product(*param_values):
                param_dict = dict(zip(param_names, combination))
                combinations.append(param_dict)
        
        return combinations
    
    def _clone_estimator(self, params: Dict[str, Any]) -> Any:
        """Clone estimator with new parameters"""
        # Create a new instance of the estimator
        estimator_class = type(self.estimator)
        
        # Get current parameters
        if hasattr(self.estimator, 'get_params'):
            current_params = self.estimator.get_params()
        else:
            current_params = {}
        
        # Update with new parameters
        current_params.update(params)
        
        # Create new instance
        if hasattr(estimator_class, '__init__'):
            try:
                new_estimator = estimator_class(**current_params)
            except:
                # Fallback: copy attributes manually
                new_estimator = estimator_class()
                for param, value in current_params.items():
                    if hasattr(new_estimator, param):
                        setattr(new_estimator, param, value)
        else:
            new_estimator = estimator_class()
        
        return new_estimator
    
    def _validate_parameters(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                           cv_splitter: BaseValidator, scorer: Callable) -> ValidationResult:
        """Validate a single parameter combination"""
        
        scores = []
        fit_times = []
        score_times = []
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and fit estimator
            estimator = self._clone_estimator(params)
            
            # Fit timing
            start_time = time.time()
            estimator.fit(X_train, y_train)
            fit_time = time.time() - start_time
            
            # Score timing
            start_time = time.time()
            if hasattr(estimator, 'predict'):
                y_pred = estimator.predict(X_val)
            else:
                raise ValueError("Estimator must have a 'predict' method")
            
            score = scorer(y_val, y_pred)
            score_time = time.time() - start_time
            
            scores.append(score)
            fit_times.append(fit_time)
            score_times.append(score_time)
            
            # Store fold results
            fold_result = {
                'fold': fold_idx,
                'score': score,
                'fit_time': fit_time,
                'score_time': score_time,
                'train_size': len(X_train),
                'val_size': len(X_val)
            }
            
            # Add training score if requested
            if self.return_train_score:
                y_train_pred = estimator.predict(X_train)
                train_score = scorer(y_train, y_train_pred)
                fold_result['train_score'] = train_score
            
            fold_results.append(fold_result)
        
        return ValidationResult(
            score=np.mean(scores),
            scores=scores,
            fit_time=np.mean(fit_times),
            score_time=np.mean(score_times),
            parameters=params,
            fold_results=fold_results
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'GridSearchCV':
        """
        Fit grid search cross-validation
        
        Args:
            X: Training features
            y: Training labels
            groups: Group labels for group-based CV
            
        Returns:
            self: Fitted GridSearchCV
        """
        start_time = time.time()
        
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Get scorer and CV splitter
        scorer = self._get_scorer()
        cv_splitter = self._get_cv_splitter(X, y)
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        n_combinations = len(param_combinations)
        
        if self.verbose > 0:
            print(f"Fitting {cv_splitter.get_n_splits(X, y)} folds for each of {n_combinations} candidates, "
                  f"totalling {cv_splitter.get_n_splits(X, y) * n_combinations} fits")
        
        # Initialize results storage
        results = {
            'param_' + key: [] for key in param_combinations[0].keys()
        }
        results.update({
            'mean_test_score': [],
            'std_test_score': [],
            'rank_test_score': [],
            'mean_fit_time': [],
            'std_fit_time': [],
            'mean_score_time': [],
            'std_score_time': [],
            'params': []
        })
        
        if self.return_train_score:
            results.update({
                'mean_train_score': [],
                'std_train_score': []
            })
        
        # Validate each parameter combination
        validation_results = []
        
        for i, params in enumerate(param_combinations):
            if self.verbose > 1:
                print(f"[GridSearchCV] Processing combination {i+1}/{n_combinations}: {params}")
            
            # Validate parameters
            val_result = self._validate_parameters(params, X, y, cv_splitter, scorer)
            validation_results.append(val_result)
            
            # Store results
            for param_name, param_value in params.items():
                results['param_' + param_name].append(param_value)
            
            results['params'].append(params)
            results['mean_test_score'].append(val_result.score)
            results['std_test_score'].append(np.std(val_result.scores))
            results['mean_fit_time'].append(val_result.fit_time)
            results['std_fit_time'].append(np.std([fold['fit_time'] for fold in val_result.fold_results]))
            results['mean_score_time'].append(val_result.score_time)
            results['std_score_time'].append(np.std([fold['score_time'] for fold in val_result.fold_results]))
            
            if self.return_train_score:
                train_scores = [fold.get('train_score', 0) for fold in val_result.fold_results]
                results['mean_train_score'].append(np.mean(train_scores))
                results['std_train_score'].append(np.std(train_scores))
        
        # Rank results
        test_scores = np.array(results['mean_test_score'])
        
        # Higher scores are better for most metrics
        if self.scoring in ['mse', 'mae']:
            # Lower is better for error metrics
            ranks = np.argsort(test_scores) + 1
        else:
            # Higher is better for accuracy, r2, f1, etc.
            ranks = np.argsort(-test_scores) + 1
        
        results['rank_test_score'] = ranks.tolist()
        
        # Find best parameters
        best_idx = np.argmin(ranks) if self.scoring in ['mse', 'mae'] else np.argmax(test_scores)
        
        self.best_index_ = best_idx
        self.best_score_ = test_scores[best_idx]
        self.best_params_ = param_combinations[best_idx]
        self.cv_results_ = results
        
        # Refit on full dataset if requested
        if self.refit:
            self.best_estimator_ = self._clone_estimator(self.best_params_)
            self.best_estimator_.fit(X, y)
        
        total_time = time.time() - start_time
        
        if self.verbose > 0:
            print(f"Grid search completed in {total_time:.2f} seconds")
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Best parameters: {self.best_params_}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using best estimator"""
        if self.best_estimator_ is None:
            raise ValueError("Must fit grid search before predicting")
        
        return self.best_estimator_.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score using best estimator"""
        if self.best_estimator_ is None:
            raise ValueError("Must fit grid search before scoring")
        
        y_pred = self.predict(X)
        scorer = self._get_scorer()
        return scorer(y, y_pred)

class RandomizedSearchCV:
    """
    Randomized search cross-validation
    """
    
    def __init__(self, 
                 estimator: Any,
                 param_distributions: Dict[str, Any],
                 n_iter: int = 10,
                 scoring: Union[str, Callable] = 'accuracy',
                 cv: Union[int, BaseValidator] = 5,
                 random_state: Optional[int] = None,
                 verbose: int = 0):
        """
        Initialize RandomizedSearchCV
        
        Args:
            estimator: Machine learning estimator
            param_distributions: Dictionary with parameter distributions
            n_iter: Number of parameter settings to sample
            scoring: Scoring function
            cv: Cross-validation strategy
            random_state: Random seed
            verbose: Verbosity level
        """
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        
        # Convert to GridSearchCV with sampled parameters
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample parameter combinations
        param_list = []
        for _ in range(n_iter):
            params = {}
            for param_name, distribution in param_distributions.items():
                if hasattr(distribution, 'rvs'):
                    # scipy distribution
                    params[param_name] = distribution.rvs()
                elif isinstance(distribution, list):
                    # List of values
                    params[param_name] = np.random.choice(distribution)
                elif hasattr(distribution, '__call__'):
                    # Function
                    params[param_name] = distribution()
                else:
                    raise ValueError(f"Unknown distribution type for {param_name}")
            param_list.append(params)
        
        # Create GridSearchCV with sampled parameters
        self.grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_list,
            scoring=scoring,
            cv=cv,
            verbose=verbose,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomizedSearchCV':
        """Fit randomized search"""
        self.grid_search.fit(X, y)
        
        # Copy results
        self.cv_results_ = self.grid_search.cv_results_
        self.best_estimator_ = self.grid_search.best_estimator_
        self.best_score_ = self.grid_search.best_score_
        self.best_params_ = self.grid_search.best_params_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.grid_search.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score predictions"""
        return self.grid_search.score(X, y)

class HyperparameterOptimizer:
    """
    High-level hyperparameter optimization interface
    """
    
    def __init__(self, search_type: str = 'grid', random_state: Optional[int] = None):
        """
        Initialize optimizer
        
        Args:
            search_type: 'grid' or 'random'
            random_state: Random seed
        """
        self.search_type = search_type
        self.random_state = random_state
        self.search_history = []
        
    def optimize(self, estimator: Any, X: np.ndarray, y: np.ndarray,
                param_space: Dict[str, Any], 
                scoring: str = 'accuracy',
                cv: int = 5,
                n_iter: int = 10,
                verbose: int = 1) -> Dict[str, Any]:
        """
        Optimize hyperparameters
        
        Args:
            estimator: ML estimator
            X: Features
            y: Labels
            param_space: Parameter space
            scoring: Scoring metric
            cv: Cross-validation folds
            n_iter: Number of iterations (for random search)
            verbose: Verbosity level
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        if self.search_type == 'grid':
            searcher = GridSearchCV(
                estimator=estimator,
                param_grid=param_space,
                scoring=scoring,
                cv=cv,
                verbose=verbose,
                random_state=self.random_state
            )
        elif self.search_type == 'random':
            searcher = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_space,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                verbose=verbose,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")
        
        # Fit searcher
        searcher.fit(X, y)
        
        total_time = time.time() - start_time
        
        # Store results
        result = {
            'best_score': searcher.best_score_,
            'best_params': searcher.best_params_,
            'best_estimator': searcher.best_estimator_,
            'cv_results': searcher.cv_results_,
            'total_time': total_time,
            'search_type': self.search_type,
            'scoring': scoring,
            'cv_folds': cv
        }
        
        self.search_history.append(result)
        
        return result
    
    def plot_results(self, result: Dict[str, Any], figsize: Tuple[int, int] = (15, 10)):
        """Plot optimization results"""
        cv_results = result['cv_results']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Score distribution
        scores = cv_results['mean_test_score']
        axes[0, 0].hist(scores, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(result['best_score'], color='red', linestyle='--', 
                          label=f'Best: {result["best_score"]:.4f}')
        axes[0, 0].set_xlabel('Mean Test Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Score vs. rank
        ranks = cv_results['rank_test_score']
        axes[0, 1].scatter(ranks, scores, alpha=0.6)
        axes[0, 1].set_xlabel('Rank')
        axes[0, 1].set_ylabel('Mean Test Score')
        axes[0, 1].set_title('Score vs. Rank')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Fit time distribution
        fit_times = cv_results['mean_fit_time']
        axes[1, 0].hist(fit_times, bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 0].set_xlabel('Mean Fit Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Fit Time Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Score vs. fit time
        axes[1, 1].scatter(fit_times, scores, alpha=0.6, color='green')
        axes[1, 1].set_xlabel('Mean Fit Time (seconds)')
        axes[1, 1].set_ylabel('Mean Test Score')
        axes[1, 1].set_title('Score vs. Fit Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Parameter importance (if single parameter varied)
        self._plot_parameter_importance(cv_results)
    
    def _plot_parameter_importance(self, cv_results: Dict[str, Any]):
        """Plot parameter importance"""
        param_cols = [col for col in cv_results.keys() if col.startswith('param_')]
        
        if len(param_cols) == 1:
            # Single parameter analysis
            param_name = param_cols[0].replace('param_', '')
            param_values = cv_results[param_cols[0]]
            scores = cv_results['mean_test_score']
            
            plt.figure(figsize=(10, 6))
            
            # Check if parameter is numeric
            try:
                param_values_numeric = [float(x) for x in param_values]
                plt.plot(param_values_numeric, scores, 'o-', linewidth=2, markersize=8)
                plt.xlabel(f'{param_name} (numeric)')
            except:
                # Categorical parameter
                unique_values = list(set(param_values))
                mean_scores = []
                std_scores = []
                
                for val in unique_values:
                    val_scores = [scores[i] for i, pval in enumerate(param_values) if pval == val]
                    mean_scores.append(np.mean(val_scores))
                    std_scores.append(np.std(val_scores))
                
                x_pos = range(len(unique_values))
                plt.bar(x_pos, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
                plt.xticks(x_pos, unique_values)
                plt.xlabel(f'{param_name} (categorical)')
            
            plt.ylabel('Mean Test Score')
            plt.title(f'Parameter Sensitivity: {param_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        elif len(param_cols) == 2:
            # Two parameter heatmap
            param1_name = param_cols[0].replace('param_', '')
            param2_name = param_cols[1].replace('param_', '')
            
            param1_values = cv_results[param_cols[0]]
            param2_values = cv_results[param_cols[1]]
            scores = cv_results['mean_test_score']
            
            # Create pivot table
            df = pd.DataFrame({
                param1_name: param1_values,
                param2_name: param2_values,
                'score': scores
            })
            
            pivot_table = df.pivot_table(values='score', 
                                       index=param2_name, 
                                       columns=param1_name, 
                                       aggfunc='mean')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis')
            plt.title(f'Parameter Interaction: {param1_name} vs {param2_name}')
            plt.tight_layout()
            plt.show()

# Example models for demonstration
class SimpleLinearRegression:
    """Simple linear regression for demonstration"""
    
    def __init__(self, fit_intercept: bool = True, regularization: float = 0.0):
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.fit_intercept:
            X_with_bias = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_bias = X
        
        # Ridge regression solution
        I = np.eye(X_with_bias.shape[1])
        if self.fit_intercept:
            I[0, 0] = 0  # Don't regularize bias term
        
        try:
            params = np.linalg.solve(
                X_with_bias.T @ X_with_bias + self.regularization * I,
                X_with_bias.T @ y
            )
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            params = np.linalg.pinv(X_with_bias.T @ X_with_bias + self.regularization * I) @ X_with_bias.T @ y
        
        if self.fit_intercept:
            self.bias = params[0]
            self.weights = params[1:]
        else:
            self.bias = 0
            self.weights = params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X = np.asarray(X)
        return X @ self.weights + self.bias

class SimpleLogisticRegression:
    """Simple logistic regression for demonstration"""
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 regularization: float = 0.0, tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function"""
        z = np.clip(z, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            predictions = self._sigmoid(z)
            
            # Compute gradients
            dw = (1 / n_samples) * X.T @ (predictions - y) + self.regularization * self.weights
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if np.linalg.norm(dw) < self.tolerance:
                break
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        X = np.asarray(X)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return (self.predict_proba(X) > 0.5).astype(int)

# Comprehensive demonstration
def demonstrate_hyperparameter_tuning():
    """Comprehensive demonstration of hyperparameter tuning"""
    
    print("=== Hyperparameter Tuning Demonstration ===\n")
    
    # Generate sample data
    from sklearn.datasets import make_classification, make_regression
    
    # 1. Classification example
    print("1. Classification - Logistic Regression")
    print("-" * 50)
    
    X_class, y_class = make_classification(
        n_samples=1000, n_features=10, n_informative=8, 
        n_redundant=2, n_classes=2, random_state=42
    )
    
    # Define parameter grid
    param_grid_lr = {
        'learning_rate': [0.001, 0.01, 0.1, 1.0],
        'max_iter': [100, 500, 1000],
        'regularization': [0.0, 0.01, 0.1, 1.0]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=SimpleLogisticRegression(),
        param_grid=param_grid_lr,
        scoring='accuracy',
        cv=5,
        verbose=1
    )
    
    grid_search.fit(X_class, y_class)
    
    print(f"Best score: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    # 2. Regression example
    print("\n2. Regression - Linear Regression")
    print("-" * 50)
    
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=5, noise=0.1, random_state=42
    )
    
    # Define parameter grid
    param_grid_reg = {
        'fit_intercept': [True, False],
        'regularization': [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    }
    
    # Grid search for regression
    grid_search_reg = GridSearchCV(
        estimator=SimpleLinearRegression(),
        param_grid=param_grid_reg,
        scoring='r2',
        cv=5,
        verbose=1
    )
    
    grid_search_reg.fit(X_reg, y_reg)
    
    print(f"Best R² score: {grid_search_reg.best_score_:.4f}")
    print(f"Best parameters: {grid_search_reg.best_params_}")
    
    # 3. Randomized search
    print("\n3. Randomized Search")
    print("-" * 50)
    
    # Define parameter distributions
    param_distributions = {
        'learning_rate': [0.001, 0.01, 0.1, 1.0],
        'max_iter': [100, 200, 500, 1000, 2000],
        'regularization': [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    }
    
    random_search = RandomizedSearchCV(
        estimator=SimpleLogisticRegression(),
        param_distributions=param_distributions,
        n_iter=20,
        scoring='accuracy',
        cv=5,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_class, y_class)
    
    print(f"Random search best score: {random_search.best_score_:.4f}")
    print(f"Random search best parameters: {random_search.best_params_}")
    
    # 4. High-level optimizer
    print("\n4. High-level Optimizer")
    print("-" * 50)
    
    optimizer = HyperparameterOptimizer(search_type='grid', random_state=42)
    
    result = optimizer.optimize(
        estimator=SimpleLogisticRegression(),
        X=X_class,
        y=y_class,
        param_space={
            'learning_rate': [0.01, 0.1],
            'regularization': [0.0, 0.1, 1.0]
        },
        scoring='accuracy',
        cv=3,
        verbose=0
    )
    
    print(f"Optimizer best score: {result['best_score']:.4f}")
    print(f"Optimizer best parameters: {result['best_params']}")
    print(f"Total optimization time: {result['total_time']:.2f} seconds")
    
    # 5. Cross-validation strategies comparison
    print("\n5. Cross-validation Strategies Comparison")
    print("-" * 50)
    
    cv_strategies = {
        'KFold': KFoldValidator(n_splits=5, random_state=42),
        'StratifiedKFold': StratifiedKFoldValidator(n_splits=5, random_state=42)
    }
    
    for cv_name, cv_strategy in cv_strategies.items():
        grid_cv = GridSearchCV(
            estimator=SimpleLogisticRegression(),
            param_grid={'learning_rate': [0.01, 0.1], 'regularization': [0.0, 0.1]},
            scoring='accuracy',
            cv=cv_strategy,
            verbose=0
        )
        
        grid_cv.fit(X_class, y_class)
        print(f"{cv_name}: Best score = {grid_cv.best_score_:.4f}")
    
    # 6. Performance analysis
    print("\n6. Performance Analysis")
    print("-" * 50)
    
    # Compare grid vs random search
    search_times = []
    search_scores = []
    search_types = []
    
    # Grid search timing
    start_time = time.time()
    grid_search_small = GridSearchCV(
        estimator=SimpleLogisticRegression(),
        param_grid={'learning_rate': [0.01, 0.1], 'regularization': [0.0, 0.1]},
        scoring='accuracy',
        cv=3,
        verbose=0
    )
    grid_search_small.fit(X_class[:200], y_class[:200])
    grid_time = time.time() - start_time
    
    # Random search timing
    start_time = time.time()
    random_search_small = RandomizedSearchCV(
        estimator=SimpleLogisticRegression(),
        param_distributions={'learning_rate': [0.01, 0.1], 'regularization': [0.0, 0.1]},
        n_iter=4,
        scoring='accuracy',
        cv=3,
        verbose=0,
        random_state=42
    )
    random_search_small.fit(X_class[:200], y_class[:200])
    random_time = time.time() - start_time
    
    print(f"Grid search time: {grid_time:.3f} seconds, score: {grid_search_small.best_score_:.4f}")
    print(f"Random search time: {random_time:.3f} seconds, score: {random_search_small.best_score_:.4f}")
    
    return {
        'grid_search': grid_search,
        'random_search': random_search,
        'optimizer_result': result,
        'classification_data': (X_class, y_class),
        'regression_data': (X_reg, y_reg)
    }

# Advanced hyperparameter tuning utilities
class ParameterSpace:
    """Utilities for defining parameter spaces"""
    
    @staticmethod
    def logspace(start: float, stop: float, num: int) -> List[float]:
        """Generate logarithmically spaced values"""
        return [10**x for x in np.linspace(start, stop, num)]
    
    @staticmethod
    def choice(options: List[Any]) -> List[Any]:
        """Return list of choices"""
        return options
    
    @staticmethod
    def uniform(low: float, high: float, size: int = 10) -> List[float]:
        """Generate uniformly distributed values"""
        return np.linspace(low, high, size).tolist()

def save_search_results(results: Dict[str, Any], filename: str):
    """Save search results to file"""
    # Convert non-serializable objects
    serializable_results = {}
    for key, value in results.items():
        if key == 'best_estimator':
            serializable_results[key] = str(type(value).__name__)
        else:
            serializable_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

def load_search_results(filename: str) -> Dict[str, Any]:
    """Load search results from file"""
    with open(filename, 'r') as f:
        return json.load(f)

# Run demonstration
if __name__ == "__main__":
    results = demonstrate_hyperparameter_tuning()
    
    # Create visualizations
    if 'optimizer_result' in results:
        optimizer = HyperparameterOptimizer()
        try:
            optimizer.plot_results(results['optimizer_result'])
        except Exception as e:
            print(f"Could not create plots: {e}")
    
    print("\n=== Hyperparameter Tuning Demonstration Complete ===")
```

This comprehensive implementation provides:

1. **Complete GridSearchCV class** with cross-validation support
2. **RandomizedSearchCV** for efficient parameter space exploration
3. **Multiple validation strategies** (K-Fold, Stratified K-Fold, Time Series)
4. **Custom scoring functions** (accuracy, MSE, MAE, R², F1)
5. **High-level optimizer interface** for easy hyperparameter tuning
6. **Performance analysis tools** for comparing search strategies
7. **Visualization utilities** for understanding parameter importance
8. **Example estimators** (Linear and Logistic Regression) for demonstration
9. **Parameter space utilities** for defining search spaces
10. **Results persistence** for saving and loading optimization results
11. **Comprehensive demonstrations** showing real-world usage scenarios
12. **Production-ready features** with proper error handling and type hints

---

## Question 11

**Explain the concept of a neural network, and how you would implement one in Python.**

**Answer:**

Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that can learn complex patterns through training.

### Core Concepts:

1. **Neurons**: Basic processing units that receive inputs, apply weights and bias, and produce output
2. **Layers**: Collections of neurons (input, hidden, output layers)
3. **Weights & Biases**: Learnable parameters that determine network behavior
4. **Activation Functions**: Non-linear functions that introduce complexity
5. **Forward Propagation**: Computing output from inputs
6. **Backpropagation**: Learning algorithm that adjusts weights based on errors
7. **Loss Functions**: Measure difference between predicted and actual outputs

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable, Dict, Any
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from sklearn.datasets import make_classification, make_regression, load_digits
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

@dataclass
class TrainingHistory:
    """Container for training history"""
    loss: List[float]
    val_loss: List[float]
    accuracy: List[float]
    val_accuracy: List[float]
    epochs: List[int]

class ActivationFunction(ABC):
    """Abstract base class for activation functions"""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        pass
    
    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Backward pass (derivative)"""
        pass

class ReLU(ActivationFunction):
    """Rectified Linear Unit activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow
        x_clipped = np.clip(x, -250, 250)
        return 1 / (1 + np.exp(-x_clipped))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)

class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

class LeakyReLU(ActivationFunction):
    """Leaky ReLU activation function"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)

class Softmax(ActivationFunction):
    """Softmax activation function (for output layer)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        # Softmax derivative is handled in the loss function
        return np.ones_like(x)

class Linear(ActivationFunction):
    """Linear activation function (identity)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class LossFunction(ABC):
    """Abstract base class for loss functions"""
    
    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss"""
        pass
    
    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient"""
        pass

class MeanSquaredError(LossFunction):
    """Mean Squared Error loss function"""
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / len(y_true)

class BinaryCrossentropy(LossFunction):
    """Binary crossentropy loss function"""
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Clip predictions to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped) * len(y_true))

class CategoricalCrossentropy(LossFunction):
    """Categorical crossentropy loss function"""
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # For softmax + categorical crossentropy, the gradient simplifies
        return (y_pred - y_true) / len(y_true)

class Optimizer(ABC):
    """Abstract base class for optimizers"""
    
    @abstractmethod
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Update parameters"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset optimizer state"""
        pass

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity - self.learning_rate * grads
        return params + self.velocity
    
    def reset(self):
        self.velocity = None

class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # Compute bias-corrected estimates
        m_corrected = self.m / (1 - self.beta1 ** self.t)
        v_corrected = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        return params - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
    
    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

class Layer:
    """Dense (fully connected) neural network layer"""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: ActivationFunction, 
                 weight_init: str = 'xavier',
                 use_bias: bool = True):
        """
        Initialize layer
        
        Args:
            input_size: Number of input features
            output_size: Number of output neurons
            activation: Activation function
            weight_init: Weight initialization method
            use_bias: Whether to use bias terms
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        
        # Initialize weights
        if weight_init == 'xavier':
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == 'he':
            # He initialization (good for ReLU)
            std = np.sqrt(2 / input_size)
            self.weights = np.random.normal(0, std, (input_size, output_size))
        elif weight_init == 'normal':
            # Normal initialization
            self.weights = np.random.normal(0, 0.01, (input_size, output_size))
        else:
            # Random uniform
            self.weights = np.random.uniform(-0.1, 0.1, (input_size, output_size))
        
        # Initialize bias
        if self.use_bias:
            self.bias = np.zeros((1, output_size))
        else:
            self.bias = None
        
        # Store for backpropagation
        self.last_input = None
        self.last_z = None
        self.last_output = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.last_input = x.copy()
        
        # Linear transformation
        self.last_z = x @ self.weights
        if self.use_bias:
            self.last_z += self.bias
        
        # Apply activation
        self.last_output = self.activation.forward(self.last_z)
        return self.last_output
    
    def backward(self, d_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass
        
        Returns:
            d_input: Gradient w.r.t. input
            d_weights: Gradient w.r.t. weights
            d_bias: Gradient w.r.t. bias
        """
        # Gradient w.r.t. activation
        d_z = d_output * self.activation.backward(self.last_z)
        
        # Gradient w.r.t. weights
        d_weights = self.last_input.T @ d_z
        
        # Gradient w.r.t. bias
        d_bias = np.sum(d_z, axis=0, keepdims=True) if self.use_bias else None
        
        # Gradient w.r.t. input
        d_input = d_z @ self.weights.T
        
        return d_input, d_weights, d_bias

class NeuralNetwork:
    """
    Multi-layer perceptron neural network
    """
    
    def __init__(self, layers_config: List[Dict[str, Any]], 
                 loss_function: LossFunction,
                 optimizer: Optimizer):
        """
        Initialize neural network
        
        Args:
            layers_config: List of layer configurations
            loss_function: Loss function to use
            optimizer: Optimizer for training
        """
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.layers = []
        self.history = TrainingHistory([], [], [], [], [])
        
        # Build layers
        for i, config in enumerate(layers_config):
            if 'input_size' not in config:
                if i == 0:
                    raise ValueError("First layer must specify input_size")
                config['input_size'] = layers_config[i-1]['output_size']
            
            # Get activation function
            activation_name = config.get('activation', 'relu')
            activation = self._get_activation_function(activation_name)
            
            layer = Layer(
                input_size=config['input_size'],
                output_size=config['output_size'],
                activation=activation,
                weight_init=config.get('weight_init', 'xavier'),
                use_bias=config.get('use_bias', True)
            )
            
            self.layers.append(layer)
    
    def _get_activation_function(self, name: str) -> ActivationFunction:
        """Get activation function by name"""
        activations = {
            'relu': ReLU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'leaky_relu': LeakyReLU(),
            'softmax': Softmax(),
            'linear': Linear()
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        
        return activations[name]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """Backward pass through the network"""
        # Compute loss gradient
        d_output = self.loss_function.backward(y_true, y_pred)
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            d_input, d_weights, d_bias = layer.backward(d_output)
            
            # Update weights and bias
            layer.weights = self.optimizer.update(layer.weights, d_weights)
            if layer.use_bias and d_bias is not None:
                layer.bias = self.optimizer.update(layer.bias, d_bias)
            
            # Pass gradient to previous layer
            d_output = d_input
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.0,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            verbose: int = 1,
            patience: int = 10,
            min_delta: float = 1e-4) -> TrainingHistory:
        """
        Train the neural network
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            validation_split: Fraction of data to use for validation
            X_val: Validation features (overrides validation_split)
            y_val: Validation labels
            verbose: Verbosity level
            patience: Early stopping patience
            min_delta: Minimum change for early stopping
            
        Returns:
            Training history
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)
        elif validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            X = X_train
            y = y_train
        else:
            X_val = y_val = None
        
        # Reset optimizer
        self.optimizer.reset()
        
        # Initialize history
        self.history = TrainingHistory([], [], [], [], [])
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Training
            epoch_losses = []
            epoch_predictions = []
            epoch_targets = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.loss_function.forward(y_batch, y_pred)
                epoch_losses.append(loss)
                
                # Store predictions for accuracy calculation
                epoch_predictions.append(y_pred)
                epoch_targets.append(y_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, y_pred)
            
            # Calculate epoch metrics
            avg_loss = np.mean(epoch_losses)
            
            # Concatenate all predictions and targets
            all_predictions = np.vstack(epoch_predictions)
            all_targets = np.vstack(epoch_targets)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(all_targets, all_predictions)
            
            # Validation
            val_loss = val_accuracy = 0.0
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss = self.loss_function.forward(y_val, y_val_pred)
                val_accuracy = self._calculate_accuracy(y_val, y_val_pred)
            
            # Store history
            self.history.loss.append(avg_loss)
            self.history.accuracy.append(accuracy)
            self.history.val_loss.append(val_loss)
            self.history.val_accuracy.append(val_accuracy)
            self.history.epochs.append(epoch + 1)
            
            # Verbose output
            if verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                      f"loss: {avg_loss:.4f} - accuracy: {accuracy:.4f}", end="")
                
                if X_val is not None:
                    print(f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
                else:
                    print()
            
            # Early stopping
            if X_val is not None and patience > 0:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    if verbose > 0:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X = np.asarray(X)
        return self.forward(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        y_pred = self.predict(X)
        
        if y_pred.shape[1] == 1:
            # Binary classification
            return (y_pred > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            return np.argmax(y_pred, axis=1)
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        if y_true.shape[1] == 1:
            # Binary classification
            y_pred_classes = (y_pred > 0.5).astype(int)
            return np.mean(y_true == y_pred_classes)
        else:
            # Multi-class classification
            y_true_classes = np.argmax(y_true, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
            return np.mean(y_true_classes == y_pred_classes)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        y_pred = self.predict(X)
        loss = self.loss_function.forward(y, y_pred)
        accuracy = self._calculate_accuracy(y, y_pred)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        model_data = {
            'layers_config': [],
            'weights_and_biases': [],
            'loss_function': type(self.loss_function).__name__,
            'optimizer': type(self.optimizer).__name__
        }
        
        # Save layer configurations and parameters
        for layer in self.layers:
            config = {
                'input_size': layer.input_size,
                'output_size': layer.output_size,
                'activation': type(layer.activation).__name__,
                'use_bias': layer.use_bias
            }
            model_data['layers_config'].append(config)
            
            # Save weights and bias
            layer_params = {
                'weights': layer.weights,
                'bias': layer.bias
            }
            model_data['weights_and_biases'].append(layer_params)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Note: This is a simplified version
        # In practice, you'd need to reconstruct the full model
        print(f"Model loaded from {filepath}")
        print(f"Architecture: {len(model_data['layers_config'])} layers")
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot training history"""
        if not self.history.epochs:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        axes[0].plot(self.history.epochs, self.history.loss, 'b-', label='Training Loss')
        if self.history.val_loss and any(val > 0 for val in self.history.val_loss):
            axes[0].plot(self.history.epochs, self.history.val_loss, 'r-', label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history.epochs, self.history.accuracy, 'b-', label='Training Accuracy')
        if self.history.val_accuracy and any(val > 0 for val in self.history.val_accuracy):
            axes[1].plot(self.history.epochs, self.history.val_accuracy, 'r-', label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class MLPClassifier(NeuralNetwork):
    """Multi-layer perceptron for classification"""
    
    def __init__(self, hidden_layers: List[int] = [100], 
                 activation: str = 'relu',
                 output_activation: str = 'softmax',
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 random_state: Optional[int] = None):
        """
        Initialize MLP classifier
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            learning_rate: Learning rate
            optimizer: Optimizer type
            random_state: Random seed
        """
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # These will be set when fit is called
        self.n_classes_ = None
        self.classes_ = None
        self.label_encoder_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TrainingHistory:
        """Fit the classifier"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Convert to one-hot encoding
        y_onehot = np.eye(self.n_classes_)[y_encoded]
        
        # Build network architecture
        layers_config = []
        
        # Hidden layers
        prev_size = X.shape[1]
        for hidden_size in self.hidden_layers:
            layers_config.append({
                'input_size': prev_size,
                'output_size': hidden_size,
                'activation': self.activation_name
            })
            prev_size = hidden_size
        
        # Output layer
        output_activation = self.output_activation_name
        if self.n_classes_ == 2 and output_activation == 'softmax':
            # Binary classification can use sigmoid
            output_size = 1
            output_activation = 'sigmoid'
            y_onehot = y_encoded.reshape(-1, 1).astype(float)
        else:
            output_size = self.n_classes_
        
        layers_config.append({
            'input_size': prev_size,
            'output_size': output_size,
            'activation': output_activation
        })
        
        # Choose loss function
        if output_size == 1:
            loss_function = BinaryCrossentropy()
        else:
            loss_function = CategoricalCrossentropy()
        
        # Choose optimizer
        if self.optimizer_name == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=self.learning_rate)
        else:
            optimizer = Adam(learning_rate=self.learning_rate)
        
        # Initialize parent class
        super().__init__(layers_config, loss_function, optimizer)
        
        # Train the network
        return super().fit(X, y_onehot, **kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        X = np.asarray(X)
        y_pred_proba = super().predict(X)
        
        if y_pred_proba.shape[1] == 1:
            # Binary classification
            y_pred_encoded = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        
        return self.label_encoder_.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return super().predict(X)

class MLPRegressor(NeuralNetwork):
    """Multi-layer perceptron for regression"""
    
    def __init__(self, hidden_layers: List[int] = [100], 
                 activation: str = 'relu',
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 random_state: Optional[int] = None):
        """
        Initialize MLP regressor
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            learning_rate: Learning rate
            optimizer: Optimizer type
            random_state: Random seed
        """
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TrainingHistory:
        """Fit the regressor"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Build network architecture
        layers_config = []
        
        # Hidden layers
        prev_size = X.shape[1]
        for hidden_size in self.hidden_layers:
            layers_config.append({
                'input_size': prev_size,
                'output_size': hidden_size,
                'activation': self.activation_name
            })
            prev_size = hidden_size
        
        # Output layer (linear for regression)
        layers_config.append({
            'input_size': prev_size,
            'output_size': y.shape[1],
            'activation': 'linear'
        })
        
        # Loss function and optimizer
        loss_function = MeanSquaredError()
        
        if self.optimizer_name == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=self.learning_rate)
        else:
            optimizer = Adam(learning_rate=self.learning_rate)
        
        # Initialize parent class
        super().__init__(layers_config, loss_function, optimizer)
        
        # Train the network
        return super().fit(X, y, **kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        y_pred = super().predict(X)
        return y_pred.flatten() if y_pred.shape[1] == 1 else y_pred

# Comprehensive demonstration
def demonstrate_neural_networks():
    """Comprehensive demonstration of neural networks"""
    
    print("=== Neural Network Demonstration ===\n")
    
    # 1. Binary Classification
    print("1. Binary Classification Example")
    print("-" * 50)
    
    # Generate binary classification data
    X_binary, y_binary = make_classification(
        n_samples=1000, n_features=10, n_informative=8,
        n_redundant=2, n_classes=2, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train classifier
    clf = MLPClassifier(
        hidden_layers=[64, 32],
        activation='relu',
        learning_rate=0.001,
        optimizer='adam',
        random_state=42
    )
    
    print("Training binary classifier...")
    history = clf.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        patience=10
    )
    
    # Evaluate
    train_metrics = clf.evaluate(X_train_scaled, np.eye(2)[y_train])
    test_predictions = clf.predict(X_test_scaled)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"Training accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # 2. Multi-class Classification
    print("\n2. Multi-class Classification Example")
    print("-" * 50)
    
    # Load digits dataset (simplified)
    from sklearn.datasets import load_digits
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    
    # Take subset for faster training
    indices = np.random.choice(len(X_digits), 500, replace=False)
    X_digits = X_digits[indices]
    y_digits = y_digits[indices]
    
    # Split and scale
    X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(
        X_digits, y_digits, test_size=0.2, random_state=42
    )
    
    scaler_digits = StandardScaler()
    X_train_digits_scaled = scaler_digits.fit_transform(X_train_digits)
    X_test_digits_scaled = scaler_digits.transform(X_test_digits)
    
    # Create multi-class classifier
    clf_multi = MLPClassifier(
        hidden_layers=[128, 64],
        activation='relu',
        learning_rate=0.001,
        optimizer='adam',
        random_state=42
    )
    
    print("Training multi-class classifier...")
    history_multi = clf_multi.fit(
        X_train_digits_scaled, y_train_digits,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    test_predictions_multi = clf_multi.predict(X_test_digits_scaled)
    test_accuracy_multi = np.mean(test_predictions_multi == y_test_digits)
    
    print(f"Multi-class test accuracy: {test_accuracy_multi:.4f}")
    
    # 3. Regression Example
    print("\n3. Regression Example")
    print("-" * 50)
    
    # Generate regression data
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=8, noise=0.1, random_state=42
    )
    
    # Split and scale
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    # Create regressor
    reg = MLPRegressor(
        hidden_layers=[64, 32],
        activation='relu',
        learning_rate=0.001,
        optimizer='adam',
        random_state=42
    )
    
    print("Training regressor...")
    history_reg = reg.fit(
        X_train_reg_scaled, y_train_reg,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    test_predictions_reg = reg.predict(X_test_reg_scaled)
    mse = np.mean((test_predictions_reg - y_test_reg) ** 2)
    r2 = 1 - (np.sum((y_test_reg - test_predictions_reg) ** 2) / 
              np.sum((y_test_reg - np.mean(y_test_reg)) ** 2))
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # 4. Low-level Neural Network Example
    print("\n4. Low-level Neural Network Example")
    print("-" * 50)
    
    # Create a simple XOR problem
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    # Define network architecture
    layers_config = [
        {'input_size': 2, 'output_size': 4, 'activation': 'tanh'},
        {'input_size': 4, 'output_size': 1, 'activation': 'sigmoid'}
    ]
    
    # Create network
    nn = NeuralNetwork(
        layers_config=layers_config,
        loss_function=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=0.1)
    )
    
    print("Training XOR network...")
    # Train for more epochs since XOR is a difficult problem
    history_xor = nn.fit(X_xor, y_xor, epochs=1000, batch_size=4, verbose=0)
    
    # Test XOR network
    predictions = nn.predict(X_xor)
    print("XOR Results:")
    for i, (input_val, target, pred) in enumerate(zip(X_xor, y_xor, predictions)):
        print(f"  {input_val} -> Target: {target[0]}, Predicted: {pred[0]:.4f}")
    
    # 5. Different Activation Functions Comparison
    print("\n5. Activation Functions Comparison")
    print("-" * 50)
    
    activation_functions = ['relu', 'tanh', 'sigmoid']
    results = {}
    
    for activation in activation_functions:
        clf_act = MLPClassifier(
            hidden_layers=[32],
            activation=activation,
            learning_rate=0.01,
            optimizer='adam',
            random_state=42
        )
        
        # Use smaller dataset for quick comparison
        history_act = clf_act.fit(
            X_train_scaled[:200], y_train[:200],
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        
        test_pred = clf_act.predict(X_test_scaled)
        accuracy = np.mean(test_pred == y_test)
        results[activation] = accuracy
        
        print(f"  {activation.upper()}: {accuracy:.4f}")
    
    return {
        'binary_classifier': clf,
        'multi_class_classifier': clf_multi,
        'regressor': reg,
        'xor_network': nn,
        'activation_results': results,
        'histories': {
            'binary': history,
            'multi_class': history_multi,
            'regression': history_reg,
            'xor': history_xor
        }
    }

# Advanced neural network utilities
def visualize_network_architecture(layers_config: List[Dict[str, Any]], figsize: Tuple[int, int] = (12, 8)):
    """Visualize neural network architecture"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions
    max_neurons = max(config['output_size'] for config in layers_config)
    max_neurons = max(max_neurons, layers_config[0]['input_size'])
    
    layer_positions = []
    neuron_positions = []
    
    # Input layer
    input_size = layers_config[0]['input_size']
    x_pos = 0
    y_positions = np.linspace(0, max_neurons, input_size)
    layer_positions.append((x_pos, y_positions))
    
    # Hidden and output layers
    for i, config in enumerate(layers_config):
        x_pos = i + 1
        output_size = config['output_size']
        y_positions = np.linspace(0, max_neurons, output_size)
        layer_positions.append((x_pos, y_positions))
    
    # Draw neurons
    for layer_idx, (x_pos, y_positions) in enumerate(layer_positions):
        for y_pos in y_positions:
            circle = plt.Circle((x_pos, y_pos), 0.1, color='lightblue', ec='black')
            ax.add_patch(circle)
    
    # Draw connections
    for layer_idx in range(len(layer_positions) - 1):
        x1, y1_positions = layer_positions[layer_idx]
        x2, y2_positions = layer_positions[layer_idx + 1]
        
        for y1 in y1_positions:
            for y2 in y2_positions:
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
    
    # Labels
    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(layers_config)-1)] + ['Output']
    for i, name in enumerate(layer_names):
        ax.text(i, max_neurons + 0.5, name, ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(-0.5, len(layer_positions) - 0.5)
    ax.set_ylim(-0.5, max_neurons + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def compare_optimizers():
    """Compare different optimizers"""
    
    print("=== Optimizer Comparison ===\n")
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    optimizers = {
        'SGD': SGD(learning_rate=0.01),
        'SGD_Momentum': SGD(learning_rate=0.01, momentum=0.9),
        'Adam': Adam(learning_rate=0.001)
    }
    
    results = {}
    
    for opt_name, optimizer in optimizers.items():
        print(f"Training with {opt_name}...")
        
        # Create network
        layers_config = [
            {'input_size': X_train_scaled.shape[1], 'output_size': 32, 'activation': 'relu'},
            {'input_size': 32, 'output_size': 1, 'activation': 'sigmoid'}
        ]
        
        nn = NeuralNetwork(
            layers_config=layers_config,
            loss_function=BinaryCrossentropy(),
            optimizer=optimizer
        )
        
        # Train
        history = nn.fit(
            X_train_scaled, y_train.reshape(-1, 1).astype(float),
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        test_pred = nn.predict(X_test_scaled)
        test_accuracy = np.mean((test_pred > 0.5).astype(int).flatten() == y_test)
        
        results[opt_name] = {
            'accuracy': test_accuracy,
            'history': history
        }
        
        print(f"  Final accuracy: {test_accuracy:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for opt_name, result in results.items():
        plt.plot(result['history'].epochs, result['history'].loss, label=opt_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for opt_name, result in results.items():
        plt.plot(result['history'].epochs, result['history'].accuracy, label=opt_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demonstration
if __name__ == "__main__":
    # Run main demonstration
    results = demonstrate_neural_networks()
    
    # Show training plots for one of the models
    if 'binary_classifier' in results:
        print("\nPlotting training history...")
        try:
            results['binary_classifier'].plot_training_history()
        except Exception as e:
            print(f"Could not create training plots: {e}")
    
    # Visualize network architecture
    print("\nVisualizing network architecture...")
    try:
        example_config = [
            {'input_size': 10, 'output_size': 64, 'activation': 'relu'},
            {'input_size': 64, 'output_size': 32, 'activation': 'relu'},
            {'input_size': 32, 'output_size': 2, 'activation': 'softmax'}
        ]
        visualize_network_architecture(example_config)
    except Exception as e:
        print(f"Could not create architecture visualization: {e}")
    
    # Compare optimizers
    print("\nComparing optimizers...")
    try:
        optimizer_results = compare_optimizers()
    except Exception as e:
        print(f"Could not run optimizer comparison: {e}")
    
    print("\n=== Neural Network Demonstration Complete ===")
```

This comprehensive implementation covers:

1. **Complete Neural Network Architecture** with layers, activations, and forward/backward propagation
2. **Multiple Activation Functions** (ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax, Linear)
3. **Various Loss Functions** (MSE, Binary/Categorical Crossentropy)
4. **Multiple Optimizers** (SGD with momentum, Adam)
5. **High-level Interfaces** (MLPClassifier, MLPRegressor)
6. **Training Features** (batch processing, validation, early stopping)
7. **Visualization Tools** (training history, network architecture)
8. **Real-world Examples** (binary/multi-class classification, regression, XOR problem)
9. **Comprehensive Demonstrations** showing practical usage
10. **Production Features** (model saving/loading, proper error handling)

---

## Question 12

**Discuss reinforcement learning and its implementation challenges.**

**Answer:**

Reinforcement Learning (RL) is a machine learning paradigm where agents learn to make decisions through trial and error by interacting with an environment to maximize cumulative rewards. Unlike supervised learning, RL doesn't require labeled data; instead, agents learn from the consequences of their actions.

### Core Concepts:

1. **Agent**: The learner or decision maker
2. **Environment**: The world the agent interacts with
3. **State (S)**: Current situation of the agent
4. **Action (A)**: Choices available to the agent
5. **Reward (R)**: Feedback signal from the environment
6. **Policy (π)**: Strategy that defines agent's behavior
7. **Value Function (V)**: Expected cumulative reward from a state
8. **Q-Function (Q)**: Expected cumulative reward for state-action pairs

### Implementation Challenges:

1. **Exploration vs. Exploitation**: Balancing trying new actions vs. using known good actions
2. **Credit Assignment**: Determining which actions led to rewards
3. **Sample Efficiency**: Learning from limited interactions
4. **Stability**: Convergence issues in function approximation
5. **Scalability**: Handling large state/action spaces
6. **Generalization**: Transferring knowledge to unseen states

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
import random
from collections import defaultdict, deque
import pickle
from dataclasses import dataclass
import time

@dataclass
class Experience:
    """Container for experience tuples"""
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool

class Environment(ABC):
    """Abstract base class for RL environments"""
    
    @abstractmethod
    def reset(self) -> Any:
        """Reset environment and return initial state"""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take action and return (next_state, reward, done, info)"""
        pass
    
    @abstractmethod
    def get_valid_actions(self, state: Any) -> List[Any]:
        """Get valid actions for given state"""
        pass
    
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Size of action space"""
        pass

class GridWorld(Environment):
    """
    Simple grid world environment for RL demonstrations
    Agent navigates grid to reach goal while avoiding obstacles
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (5, 5), 
                 obstacles: List[Tuple[int, int]] = None,
                 goal: Tuple[int, int] = None,
                 start: Tuple[int, int] = None):
        """
        Initialize grid world
        
        Args:
            grid_size: (height, width) of grid
            obstacles: List of obstacle positions
            goal: Goal position
            start: Starting position
        """
        self.grid_size = grid_size
        self.height, self.width = grid_size
        
        self.obstacles = obstacles or [(2, 2), (3, 2)]
        self.goal = goal or (grid_size[0] - 1, grid_size[1] - 1)
        self.start = start or (0, 0)
        
        self.current_pos = self.start
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        self.action_names = ['right', 'left', 'down', 'up']
        
        # Rewards
        self.goal_reward = 10.0
        self.obstacle_penalty = -5.0
        self.step_penalty = -0.1
        self.out_of_bounds_penalty = -1.0
    
    def reset(self) -> Tuple[int, int]:
        """Reset to starting position"""
        self.current_pos = self.start
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """Take action and return next state, reward, done, info"""
        if action < 0 or action >= len(self.actions):
            raise ValueError(f"Invalid action: {action}")
        
        # Calculate new position
        dy, dx = self.actions[action]
        new_y, new_x = self.current_pos[0] + dy, self.current_pos[1] + dx
        
        # Check bounds
        if (new_y < 0 or new_y >= self.height or 
            new_x < 0 or new_x >= self.width):
            # Out of bounds - stay in place, get penalty
            reward = self.out_of_bounds_penalty
            done = False
        elif (new_y, new_x) in self.obstacles:
            # Hit obstacle - stay in place, get penalty
            reward = self.obstacle_penalty
            done = False
        else:
            # Valid move
            self.current_pos = (new_y, new_x)
            
            if self.current_pos == self.goal:
                reward = self.goal_reward
                done = True
            else:
                reward = self.step_penalty
                done = False
        
        info = {
            'action_name': self.action_names[action],
            'valid_move': self.current_pos == (new_y, new_x)
        }
        
        return self.current_pos, reward, done, info
    
    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        """Get valid actions from given state"""
        valid_actions = []
        y, x = state
        
        for action, (dy, dx) in enumerate(self.actions):
            new_y, new_x = y + dy, x + dx
            
            # Check if action leads to valid position
            if (0 <= new_y < self.height and 
                0 <= new_x < self.width and
                (new_y, new_x) not in self.obstacles):
                valid_actions.append(action)
        
        return valid_actions if valid_actions else list(range(len(self.actions)))
    
    @property
    def action_space_size(self) -> int:
        return len(self.actions)
    
    def render(self, policy: Optional[Dict] = None, values: Optional[Dict] = None):
        """Render the grid world"""
        
        # Create visualization
        fig, axes = plt.subplots(1, 2 if values else 1, figsize=(12, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Grid visualization
        grid = np.zeros(self.grid_size)
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1
        
        # Mark goal
        grid[self.goal] = 1
        
        # Mark current position
        grid[self.current_pos] = 0.5
        
        im1 = axes[0].imshow(grid, cmap='RdYlGn', alpha=0.8)
        
        # Add policy arrows if provided
        if policy:
            for y in range(self.height):
                for x in range(self.width):
                    if (y, x) not in self.obstacles and (y, x) != self.goal:
                        action = policy.get((y, x), 0)
                        dy, dx = self.actions[action]
                        axes[0].arrow(x, y, dx*0.3, dy*0.3, 
                                    head_width=0.1, head_length=0.1, 
                                    fc='blue', ec='blue')
        
        axes[0].set_title('Grid World')
        axes[0].set_xticks(range(self.width))
        axes[0].set_yticks(range(self.height))
        
        # Value function visualization
        if values:
            value_grid = np.zeros(self.grid_size)
            for (y, x), value in values.items():
                value_grid[y, x] = value
            
            im2 = axes[1].imshow(value_grid, cmap='viridis')
            axes[1].set_title('Value Function')
            axes[1].set_xticks(range(self.width))
            axes[1].set_yticks(range(self.height))
            
            # Add value text
            for y in range(self.height):
                for x in range(self.width):
                    value = value_grid[y, x]
                    axes[1].text(x, y, f'{value:.1f}', ha='center', va='center',
                               color='white' if abs(value) > np.max(np.abs(value_grid))/2 else 'black')
            
            plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()

class TabularQLearning:
    """
    Tabular Q-Learning implementation
    Solves the exploration-exploitation dilemma using epsilon-greedy policy
    """
    
    def __init__(self, 
                 environment: Environment,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent
        
        Args:
            environment: RL environment
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        self.env = environment
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: Q(state, action)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
    
    def get_action(self, state: Any, training: bool = True) -> int:
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        valid_actions = self.env.get_valid_actions(state)
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(valid_actions)
        else:
            # Exploitation: greedy action
            q_values = [self.q_table[state][action] for action in valid_actions]
            
            if not q_values:
                return random.choice(valid_actions)
            
            max_q = max(q_values)
            # Handle ties by random selection
            best_actions = [action for action, q_val in zip(valid_actions, q_values) 
                          if q_val == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state: Any, action: int, reward: float, 
                      next_state: Any, done: bool):
        """
        Update Q-value using Q-learning update rule
        
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        if done:
            # Terminal state
            max_next_q = 0
        else:
            # Find maximum Q-value for next state
            valid_next_actions = self.env.get_valid_actions(next_state)
            if valid_next_actions:
                max_next_q = max(self.q_table[next_state][a] for a in valid_next_actions)
            else:
                max_next_q = 0
        
        # Q-learning update
        target_q = reward + self.gamma * max_next_q
        self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)
    
    def train(self, episodes: int = 1000, max_steps: int = 100, verbose: int = 1) -> Dict[str, List]:
        """
        Train the Q-learning agent
        
        Args:
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Verbosity level
            
        Returns:
            Training statistics
        """
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'q_table_size': []
        }
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Get action
                action = self.get_action(state, training=True)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state, done)
                
                # Update state and statistics
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Store episode statistics
            stats['episode_rewards'].append(total_reward)
            stats['episode_lengths'].append(steps)
            stats['epsilon_values'].append(self.epsilon)
            stats['q_table_size'].append(len(self.q_table))
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Verbose output
            if verbose > 0 and (episode + 1) % (episodes // 10) == 0:
                avg_reward = np.mean(stats['episode_rewards'][-100:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Epsilon: {self.epsilon:.3f} - "
                      f"Q-table size: {len(self.q_table)}")
        
        return stats
    
    def get_policy(self) -> Dict[Any, int]:
        """Extract greedy policy from Q-table"""
        policy = {}
        
        for state in self.q_table:
            valid_actions = self.env.get_valid_actions(state)
            if valid_actions:
                q_values = [self.q_table[state][action] for action in valid_actions]
                best_action_idx = np.argmax(q_values)
                policy[state] = valid_actions[best_action_idx]
        
        return policy
    
    def get_value_function(self) -> Dict[Any, float]:
        """Extract value function from Q-table"""
        values = {}
        
        for state in self.q_table:
            valid_actions = self.env.get_valid_actions(state)
            if valid_actions:
                q_values = [self.q_table[state][action] for action in valid_actions]
                values[state] = max(q_values) if q_values else 0.0
            else:
                values[state] = 0.0
        
        return values

class SARSA:
    """
    SARSA (State-Action-Reward-State-Action) implementation
    On-policy temporal difference learning algorithm
    """
    
    def __init__(self, 
                 environment: Environment,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """Initialize SARSA agent"""
        self.env = environment
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def get_action(self, state: Any) -> int:
        """Epsilon-greedy action selection"""
        valid_actions = self.env.get_valid_actions(state)
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.q_table[state][action] for action in valid_actions]
            if not q_values:
                return random.choice(valid_actions)
            
            max_q = max(q_values)
            best_actions = [action for action, q_val in zip(valid_actions, q_values) 
                          if q_val == max_q]
            return random.choice(best_actions)
    
    def train(self, episodes: int = 1000, max_steps: int = 100, verbose: int = 1) -> Dict[str, List]:
        """
        Train SARSA agent
        
        SARSA update: Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
        """
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': []
        }
        
        for episode in range(episodes):
            state = self.env.reset()
            action = self.get_action(state)
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                if done:
                    # Terminal state update
                    current_q = self.q_table[state][action]
                    self.q_table[state][action] = current_q + self.alpha * (reward - current_q)
                    total_reward += reward
                    steps += 1
                    break
                else:
                    # Get next action
                    next_action = self.get_action(next_state)
                    
                    # SARSA update
                    current_q = self.q_table[state][action]
                    next_q = self.q_table[next_state][next_action]
                    target_q = reward + self.gamma * next_q
                    self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)
                    
                    # Move to next state-action pair
                    state = next_state
                    action = next_action
                    total_reward += reward
                    steps += 1
            
            # Store statistics
            stats['episode_rewards'].append(total_reward)
            stats['episode_lengths'].append(steps)
            stats['epsilon_values'].append(self.epsilon)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Verbose output
            if verbose > 0 and (episode + 1) % (episodes // 10) == 0:
                avg_reward = np.mean(stats['episode_rewards'][-100:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return stats

class DQN:
    """
    Deep Q-Network implementation (simplified version without neural networks)
    Uses experience replay and target network concepts with tabular representation
    """
    
    def __init__(self, 
                 environment: Environment,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 replay_buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100):
        """Initialize DQN agent"""
        self.env = environment
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q-networks (main and target)
        self.q_network = defaultdict(lambda: defaultdict(float))
        self.target_network = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Training counters
        self.training_steps = 0
    
    def get_action(self, state: Any, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        valid_actions = self.env.get_valid_actions(state)
        
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.q_network[state][action] for action in valid_actions]
            if not q_values:
                return random.choice(valid_actions)
            
            max_q = max(q_values)
            best_actions = [action for action, q_val in zip(valid_actions, q_values) 
                          if q_val == max_q]
            return random.choice(best_actions)
    
    def store_experience(self, state: Any, action: int, reward: float, 
                        next_state: Any, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
    
    def update_target_network(self):
        """Copy main network to target network"""
        self.target_network = defaultdict(lambda: defaultdict(float))
        for state in self.q_network:
            for action in self.q_network[state]:
                self.target_network[state][action] = self.q_network[state][action]
    
    def replay_experience(self):
        """Train on batch of experiences from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        for experience in batch:
            state = experience.state
            action = experience.action
            reward = experience.reward
            next_state = experience.next_state
            done = experience.done
            
            current_q = self.q_network[state][action]
            
            if done:
                target_q = reward
            else:
                # Use target network for stability
                valid_next_actions = self.env.get_valid_actions(next_state)
                if valid_next_actions:
                    max_next_q = max(self.target_network[next_state][a] for a in valid_next_actions)
                else:
                    max_next_q = 0
                target_q = reward + self.gamma * max_next_q
            
            # Update Q-value
            self.q_network[state][action] = current_q + self.alpha * (target_q - current_q)
    
    def train(self, episodes: int = 1000, max_steps: int = 100, verbose: int = 1) -> Dict[str, List]:
        """Train DQN agent"""
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'buffer_size': []
        }
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Get action
                action = self.get_action(state, training=True)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                # Train on replay buffer
                if len(self.replay_buffer) >= self.batch_size:
                    self.replay_experience()
                    self.training_steps += 1
                
                # Update target network
                if self.training_steps % self.target_update_freq == 0:
                    self.update_target_network()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Store statistics
            stats['episode_rewards'].append(total_reward)
            stats['episode_lengths'].append(steps)
            stats['epsilon_values'].append(self.epsilon)
            stats['buffer_size'].append(len(self.replay_buffer))
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Verbose output
            if verbose > 0 and (episode + 1) % (episodes // 10) == 0:
                avg_reward = np.mean(stats['episode_rewards'][-100:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Epsilon: {self.epsilon:.3f} - "
                      f"Buffer Size: {len(self.replay_buffer)}")
        
        return stats

class PolicyGradient:
    """
    Simple policy gradient implementation (REINFORCE)
    Direct policy optimization without value function
    """
    
    def __init__(self, 
                 environment: Environment,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.95):
        """Initialize policy gradient agent"""
        self.env = environment
        self.alpha = learning_rate
        self.gamma = discount_factor
        
        # Policy parameters (logits for each state-action pair)
        self.policy_params = defaultdict(lambda: defaultdict(float))
        
        # Store episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def get_action_probabilities(self, state: Any) -> Dict[int, float]:
        """Get action probabilities using softmax policy"""
        valid_actions = self.env.get_valid_actions(state)
        
        # Get logits for valid actions
        logits = [self.policy_params[state][action] for action in valid_actions]
        
        # Softmax
        exp_logits = np.exp(np.array(logits) - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        return dict(zip(valid_actions, probabilities))
    
    def get_action(self, state: Any) -> int:
        """Sample action from policy"""
        action_probs = self.get_action_probabilities(state)
        actions = list(action_probs.keys())
        probabilities = list(action_probs.values())
        
        return np.random.choice(actions, p=probabilities)
    
    def store_transition(self, state: Any, action: int, reward: float):
        """Store transition for episode"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        # Calculate returns (discounted cumulative rewards)
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Update policy parameters
        for t, (state, action, G) in enumerate(zip(self.episode_states, 
                                                   self.episode_actions, 
                                                   returns)):
            # Get action probabilities
            action_probs = self.get_action_probabilities(state)
            
            # Update parameters for all actions
            for a in action_probs:
                if a == action:
                    # Increase probability of taken action
                    grad = (1 - action_probs[a]) * G
                else:
                    # Decrease probability of other actions
                    grad = -action_probs[a] * G
                
                self.policy_params[state][a] += self.alpha * grad
        
        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
    
    def train(self, episodes: int = 1000, max_steps: int = 100, verbose: int = 1) -> Dict[str, List]:
        """Train policy gradient agent"""
        stats = {
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            # Collect episode
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.store_transition(state, action, reward)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Update policy
            self.update_policy()
            
            # Store statistics
            stats['episode_rewards'].append(total_reward)
            stats['episode_lengths'].append(steps)
            
            # Verbose output
            if verbose > 0 and (episode + 1) % (episodes // 10) == 0:
                avg_reward = np.mean(stats['episode_rewards'][-100:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Avg Reward: {avg_reward:.2f}")
        
        return stats

def compare_rl_algorithms():
    """Compare different RL algorithms on GridWorld"""
    
    print("=== Reinforcement Learning Algorithms Comparison ===\n")
    
    # Create environment
    env = GridWorld(grid_size=(5, 5), obstacles=[(2, 2), (3, 2)])
    
    # Initialize algorithms
    algorithms = {
        'Q-Learning': TabularQLearning(env, learning_rate=0.1, epsilon=0.1),
        'SARSA': SARSA(env, learning_rate=0.1, epsilon=0.1),
        'DQN': DQN(env, learning_rate=0.01, epsilon=1.0, batch_size=16),
        'Policy Gradient': PolicyGradient(env, learning_rate=0.01)
    }
    
    # Train algorithms
    results = {}
    training_episodes = 500
    
    for name, algorithm in algorithms.items():
        print(f"Training {name}...")
        
        start_time = time.time()
        stats = algorithm.train(episodes=training_episodes, verbose=0)
        training_time = time.time() - start_time
        
        results[name] = {
            'stats': stats,
            'algorithm': algorithm,
            'training_time': training_time
        }
        
        # Final performance
        final_reward = np.mean(stats['episode_rewards'][-50:])
        print(f"  Final avg reward (last 50 episodes): {final_reward:.2f}")
        print(f"  Training time: {training_time:.2f} seconds")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Learning curves
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        rewards = result['stats']['episode_rewards']
        # Smooth the curves
        window = 50
        if len(rewards) >= window:
            smoothed = [np.mean(rewards[i:i+window]) for i in range(len(rewards)-window+1)]
            plt.plot(range(window-1, len(rewards)), smoothed, label=name)
        else:
            plt.plot(rewards, label=name)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Episode lengths
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        lengths = result['stats']['episode_lengths']
        window = 50
        if len(lengths) >= window:
            smoothed = [np.mean(lengths[i:i+window]) for i in range(len(lengths)-window+1)]
            plt.plot(range(window-1, len(lengths)), smoothed, label=name)
        else:
            plt.plot(lengths, label=name)
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final performance comparison
    plt.subplot(2, 2, 3)
    final_rewards = [np.mean(result['stats']['episode_rewards'][-50:]) 
                    for result in results.values()]
    algorithm_names = list(results.keys())
    
    bars = plt.bar(algorithm_names, final_rewards, alpha=0.7)
    plt.ylabel('Final Average Reward')
    plt.title('Final Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, reward in zip(bars, final_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{reward:.2f}', ha='center', va='bottom')
    
    # Training time comparison
    plt.subplot(2, 2, 4)
    training_times = [result['training_time'] for result in results.values()]
    
    bars = plt.bar(algorithm_names, training_times, alpha=0.7, color='orange')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

def demonstrate_rl_challenges():
    """Demonstrate key RL implementation challenges"""
    
    print("=== RL Implementation Challenges Demonstration ===\n")
    
    # 1. Exploration vs Exploitation
    print("1. Exploration vs Exploitation Challenge")
    print("-" * 50)
    
    env = GridWorld(grid_size=(4, 4), obstacles=[(1, 1), (2, 2)])
    
    # Compare different epsilon values
    epsilon_values = [0.0, 0.1, 0.3, 0.5]
    exploration_results = {}
    
    for epsilon in epsilon_values:
        agent = TabularQLearning(env, epsilon=epsilon, epsilon_decay=1.0)  # No decay
        stats = agent.train(episodes=200, verbose=0)
        exploration_results[f'ε={epsilon}'] = np.mean(stats['episode_rewards'][-50:])
    
    print("Final performance with different exploration rates:")
    for setting, reward in exploration_results.items():
        print(f"  {setting}: {reward:.2f}")
    
    # 2. Sample Efficiency
    print("\n2. Sample Efficiency Challenge")
    print("-" * 50)
    
    # Compare learning speed with different learning rates
    learning_rates = [0.01, 0.1, 0.3, 0.5]
    efficiency_results = {}
    
    for lr in learning_rates:
        agent = TabularQLearning(env, learning_rate=lr)
        stats = agent.train(episodes=100, verbose=0)
        
        # Find episode where agent first achieves good performance (>5 reward)
        episodes_to_learn = len(stats['episode_rewards'])
        for i, reward in enumerate(stats['episode_rewards']):
            if reward > 5:
                episodes_to_learn = i
                break
        
        efficiency_results[f'α={lr}'] = episodes_to_learn
    
    print("Episodes needed to achieve good performance (>5 reward):")
    for setting, episodes in efficiency_results.items():
        print(f"  {setting}: {episodes} episodes")
    
    # 3. Credit Assignment Problem
    print("\n3. Credit Assignment Challenge")
    print("-" * 50)
    
    # Create environment with delayed reward
    class DelayedRewardGridWorld(GridWorld):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_penalty = -0.01  # Smaller step penalty
            self.goal_reward = 10.0    # Large reward only at goal
    
    delayed_env = DelayedRewardGridWorld(grid_size=(6, 6))
    
    # Compare Q-Learning vs SARSA on delayed reward problem
    agents = {
        'Q-Learning': TabularQLearning(delayed_env, learning_rate=0.1),
        'SARSA': SARSA(delayed_env, learning_rate=0.1)
    }
    
    credit_results = {}
    for name, agent in agents.items():
        stats = agent.train(episodes=300, verbose=0)
        credit_results[name] = np.mean(stats['episode_rewards'][-50:])
    
    print("Performance on delayed reward problem:")
    for algorithm, reward in credit_results.items():
        print(f"  {algorithm}: {reward:.2f}")
    
    # 4. Scalability Challenge
    print("\n4. Scalability Challenge")
    print("-" * 50)
    
    # Compare performance on different grid sizes
    grid_sizes = [(3, 3), (5, 5), (7, 7)]
    scalability_results = {}
    
    for size in grid_sizes:
        env_scale = GridWorld(grid_size=size)
        agent = TabularQLearning(env_scale, learning_rate=0.1)
        
        start_time = time.time()
        stats = agent.train(episodes=200, verbose=0)
        training_time = time.time() - start_time
        
        state_space_size = len(agent.q_table)
        final_reward = np.mean(stats['episode_rewards'][-50:])
        
        scalability_results[f'{size[0]}x{size[1]}'] = {
            'state_space': state_space_size,
            'training_time': training_time,
            'final_reward': final_reward
        }
    
    print("Scalability analysis:")
    for size, metrics in scalability_results.items():
        print(f"  {size} grid:")
        print(f"    State space size: {metrics['state_space']}")
        print(f"    Training time: {metrics['training_time']:.2f}s")
        print(f"    Final reward: {metrics['final_reward']:.2f}")
    
    return {
        'exploration_results': exploration_results,
        'efficiency_results': efficiency_results,
        'credit_results': credit_results,
        'scalability_results': scalability_results
    }

def visualize_learned_policies():
    """Visualize policies learned by different algorithms"""
    
    print("=== Policy Visualization ===\n")
    
    env = GridWorld(grid_size=(5, 5), obstacles=[(2, 2), (3, 2)])
    
    # Train Q-Learning agent
    q_agent = TabularQLearning(env, learning_rate=0.1, epsilon=0.1)
    q_stats = q_agent.train(episodes=500, verbose=0)
    
    # Get policy and value function
    policy = q_agent.get_policy()
    values = q_agent.get_value_function()
    
    print("Visualizing Q-Learning policy and value function...")
    env.render(policy=policy, values=values)
    
    # Train SARSA agent for comparison
    sarsa_agent = SARSA(env, learning_rate=0.1, epsilon=0.1)
    sarsa_stats = sarsa_agent.train(episodes=500, verbose=0)
    
    # Convert SARSA to same format
    sarsa_policy = {}
    sarsa_values = {}
    
    for state in sarsa_agent.q_table:
        valid_actions = env.get_valid_actions(state)
        if valid_actions:
            q_values = [sarsa_agent.q_table[state][action] for action in valid_actions]
            best_action_idx = np.argmax(q_values)
            sarsa_policy[state] = valid_actions[best_action_idx]
            sarsa_values[state] = max(q_values)
    
    print("Visualizing SARSA policy and value function...")
    env.render(policy=sarsa_policy, values=sarsa_values)

# Comprehensive demonstration function
def demonstrate_reinforcement_learning():
    """Comprehensive RL demonstration"""
    
    print("=== Comprehensive Reinforcement Learning Demonstration ===\n")
    
    # 1. Basic environment interaction
    print("1. Environment Interaction Example")
    print("-" * 50)
    
    env = GridWorld(grid_size=(4, 4))
    print(f"Grid size: {env.grid_size}")
    print(f"Start position: {env.start}")
    print(f"Goal position: {env.goal}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Action space size: {env.action_space_size}")
    
    # Sample episode
    state = env.reset()
    print(f"\nSample episode:")
    print(f"Initial state: {state}")
    
    for step in range(10):
        valid_actions = env.get_valid_actions(state)
        action = random.choice(valid_actions)
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step+1}: Action={env.action_names[action]}, "
              f"Next State={next_state}, Reward={reward:.1f}, Done={done}")
        
        if done:
            print("Episode completed!")
            break
        state = next_state
    
    # 2. Algorithm comparison
    print("\n2. Algorithm Comparison")
    print("-" * 50)
    comparison_results = compare_rl_algorithms()
    
    # 3. Challenge demonstration
    print("\n3. Implementation Challenges")
    print("-" * 50)
    challenge_results = demonstrate_rl_challenges()
    
    # 4. Policy visualization
    visualize_learned_policies()
    
    return {
        'comparison_results': comparison_results,
        'challenge_results': challenge_results
    }

# Additional utilities for RL
class RLAnalyzer:
    """Utility class for analyzing RL experiments"""
    
    @staticmethod
    def plot_learning_curve(episode_rewards: List[float], window: int = 50, title: str = "Learning Curve"):
        """Plot smoothed learning curve"""
        if len(episode_rewards) < window:
            window = len(episode_rewards)
        
        smoothed_rewards = [np.mean(episode_rewards[max(0, i-window):i+1]) 
                          for i in range(len(episode_rewards))]
        
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, alpha=0.3, label='Raw rewards')
        plt.plot(smoothed_rewards, label=f'Smoothed (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def analyze_convergence(episode_rewards: List[float], threshold: float = 0.1) -> Dict[str, Any]:
        """Analyze convergence properties of learning curve"""
        
        # Find convergence point (when variance becomes small)
        window = 50
        convergence_episode = len(episode_rewards)
        
        if len(episode_rewards) > window:
            for i in range(window, len(episode_rewards)):
                recent_rewards = episode_rewards[i-window:i]
                if np.std(recent_rewards) < threshold:
                    convergence_episode = i
                    break
        
        # Final performance
        final_performance = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        
        # Learning stability
        stability = 1.0 / (1.0 + np.std(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.std(episode_rewards))
        
        return {
            'convergence_episode': convergence_episode,
            'final_performance': final_performance,
            'learning_stability': stability,
            'total_episodes': len(episode_rewards)
        }

# Run demonstration if this script is executed
if __name__ == "__main__":
    results = demonstrate_reinforcement_learning()
    
    print("\n=== Summary of Implementation Challenges ===")
    print("1. Exploration vs Exploitation: Balance between trying new actions and using known good ones")
    print("2. Credit Assignment: Determining which actions led to rewards in multi-step episodes")
    print("3. Sample Efficiency: Learning effectively from limited environment interactions")
    print("4. Stability: Ensuring convergence in function approximation settings")
    print("5. Scalability: Handling large state and action spaces")
    print("6. Generalization: Transferring knowledge to unseen states")
    
    print("\n=== Reinforcement Learning Demonstration Complete ===")
```

This comprehensive implementation demonstrates:

### Core RL Concepts:
1. **Agent-Environment Interaction** with proper state-action-reward cycles
2. **Multiple Algorithms** (Q-Learning, SARSA, DQN, Policy Gradient)
3. **Grid World Environment** for practical demonstration

### Implementation Challenges Addressed:
1. **Exploration vs Exploitation** - Epsilon-greedy strategies with decay
2. **Credit Assignment** - Temporal difference learning methods
3. **Sample Efficiency** - Experience replay in DQN
4. **Stability** - Target networks and proper learning rates
5. **Scalability** - Demonstrated on different environment sizes

### Advanced Features:
1. **Experience Replay** for sample efficiency
2. **Target Networks** for training stability
3. **Policy Visualization** for interpretability
4. **Comprehensive Comparisons** between algorithms
5. **Performance Analysis** tools for convergence studies

---

## Question 13

**What is transfer learning, and how can you implement it using Python libraries?**

**Answer:**

Transfer learning is a machine learning technique where knowledge gained from training a model on one task is applied to a related task. Instead of training from scratch, we leverage pre-trained models and adapt them to new domains, significantly reducing training time and data requirements.

### Core Concepts:

1. **Source Domain**: Original task/dataset the model was trained on
2. **Target Domain**: New task/dataset we want to apply the model to
3. **Feature Extraction**: Using pre-trained features without modification
4. **Fine-tuning**: Adapting pre-trained weights to new task
5. **Domain Adaptation**: Adjusting model for different data distributions

### Types of Transfer Learning:

1. **Inductive Transfer**: Source and target tasks are different
2. **Transductive Transfer**: Same task, different domains
3. **Unsupervised Transfer**: No labeled data in target domain

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import pickle
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TransferResults:
    """Container for transfer learning results"""
    source_accuracy: float
    target_accuracy_scratch: float
    target_accuracy_transfer: float
    training_time_scratch: float
    training_time_transfer: float
    data_efficiency: Dict[str, float]

class PretrainedModel(ABC):
    """Abstract base class for pre-trained models"""
    
    @abstractmethod
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from input data"""
        pass
    
    @abstractmethod
    def fine_tune(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fine-tune the model on new data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

class SimpleNN:
    """Simple neural network for demonstration"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 output_size: int, learning_rate: float = 0.01):
        """Initialize neural network"""
        self.learning_rate = learning_rate
        self.layers = []
        
        # Build layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append({
                'weights': np.random.normal(0, 0.1, (prev_size, hidden_size)),
                'bias': np.zeros(hidden_size),
                'type': 'hidden'
            })
            prev_size = hidden_size
        
        # Output layer
        self.layers.append({
            'weights': np.random.normal(0, 0.1, (prev_size, output_size)),
            'bias': np.zeros(output_size),
            'type': 'output'
        })
        
        # Store activations for backpropagation
        self.activations = []
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.activations = [X]
        
        for i, layer in enumerate(self.layers):
            z = self.activations[-1] @ layer['weights'] + layer['bias']
            
            if layer['type'] == 'output':
                activation = self._softmax(z)
            else:
                activation = self._relu(z)
            
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """Backward pass"""
        m = X.shape[0]
        
        # Convert y to one-hot if needed
        if y.ndim == 1:
            y_onehot = np.eye(output.shape[1])[y]
        else:
            y_onehot = y
        
        # Output layer gradient
        d_output = output - y_onehot
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if i == len(self.layers) - 1:  # Output layer
                d_weights = self.activations[i].T @ d_output / m
                d_bias = np.mean(d_output, axis=0)
                d_prev = d_output @ layer['weights'].T
            else:  # Hidden layers
                d_relu = self._relu_derivative(self.activations[i+1])
                d_hidden = d_prev * d_relu
                d_weights = self.activations[i].T @ d_hidden / m
                d_bias = np.mean(d_hidden, axis=0)
                if i > 0:
                    d_prev = d_hidden @ layer['weights'].T
            
            # Update weights and biases
            layer['weights'] -= self.learning_rate * d_weights
            layer['bias'] -= self.learning_rate * d_bias
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            batch_size: int = 32, verbose: bool = False) -> List[float]:
        """Train the neural network"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Calculate loss (cross-entropy)
                y_onehot = np.eye(output.shape[1])[y_batch] if y_batch.ndim == 1 else y_batch
                batch_loss = -np.mean(np.sum(y_onehot * np.log(output + 1e-15), axis=1))
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass
                self.backward(X_batch, y_batch, output)
            
            losses.append(epoch_loss / n_batches)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        return self.forward(X)

class TransferLearningNN(PretrainedModel):
    """Neural network with transfer learning capabilities"""
    
    def __init__(self, base_model: SimpleNN, freeze_layers: int = 0):
        """
        Initialize transfer learning model
        
        Args:
            base_model: Pre-trained base model
            freeze_layers: Number of layers to freeze (from beginning)
        """
        self.base_model = base_model
        self.freeze_layers = freeze_layers
        self.feature_extractor = None
        self.classifier = None
        
        # Create feature extractor (frozen layers)
        if freeze_layers > 0:
            self.frozen_layers = self.base_model.layers[:freeze_layers]
        else:
            self.frozen_layers = []
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features using frozen layers"""
        if not self.frozen_layers:
            return X
        
        activations = X
        for layer in self.frozen_layers:
            z = activations @ layer['weights'] + layer['bias']
            activations = np.maximum(0, z)  # ReLU
        
        return activations
    
    def fine_tune(self, X: np.ndarray, y: np.ndarray, 
                 new_output_size: Optional[int] = None,
                 epochs: int = 50, learning_rate: float = 0.01,
                 freeze_base: bool = True) -> List[float]:
        """
        Fine-tune the model on new data
        
        Args:
            X: New training data
            y: New training labels
            new_output_size: Number of classes in new task
            epochs: Training epochs
            learning_rate: Learning rate
            freeze_base: Whether to freeze base layers
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Determine output size
        if new_output_size is None:
            new_output_size = len(np.unique(y))
        
        if freeze_base:
            # Feature extraction approach
            features = self.extract_features(X)
            feature_size = features.shape[1]
            
            # Create new classifier
            self.classifier = SimpleNN(
                input_size=feature_size,
                hidden_sizes=[64],  # Small classifier
                output_size=new_output_size,
                learning_rate=learning_rate
            )
            
            losses = self.classifier.fit(features, y, epochs=epochs, verbose=False)
        else:
            # Fine-tuning approach
            # Replace output layer
            if len(self.base_model.layers) > 0:
                last_hidden_size = self.base_model.layers[-2]['weights'].shape[1] if len(self.base_model.layers) > 1 else X.shape[1]
                self.base_model.layers[-1] = {
                    'weights': np.random.normal(0, 0.1, (last_hidden_size, new_output_size)),
                    'bias': np.zeros(new_output_size),
                    'type': 'output'
                }
            
            # Lower learning rate for fine-tuning
            original_lr = self.base_model.learning_rate
            self.base_model.learning_rate = learning_rate * 0.1
            
            losses = self.base_model.fit(X, y, epochs=epochs, verbose=False)
            
            # Restore original learning rate
            self.base_model.learning_rate = original_lr
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.classifier is not None:
            features = self.extract_features(X)
            return self.classifier.predict(features)
        else:
            return self.base_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if self.classifier is not None:
            features = self.extract_features(X)
            return self.classifier.predict_proba(features)
        else:
            return self.base_model.predict_proba(X)

class DomainAdaptationModel:
    """Model for domain adaptation scenarios"""
    
    def __init__(self, base_model: SimpleNN):
        """Initialize domain adaptation model"""
        self.base_model = base_model
        self.domain_classifier = None
        self.feature_size = None
    
    def add_domain_adversarial_training(self, 
                                      X_source: np.ndarray, 
                                      X_target: np.ndarray,
                                      y_source: np.ndarray,
                                      epochs: int = 50) -> float:
        """
        Add domain adversarial training to reduce domain gap
        
        This is a simplified version demonstrating the concept
        """
        # Extract features from both domains
        features_source = self.base_model.forward(X_source)
        features_target = self.base_model.forward(X_target)
        
        # Create domain labels (0 for source, 1 for target)
        domain_labels_source = np.zeros(len(X_source))
        domain_labels_target = np.ones(len(X_target))
        
        # Combine features and labels
        all_features = np.vstack([features_source, features_target])
        all_domain_labels = np.hstack([domain_labels_source, domain_labels_target])
        
        # Train domain classifier
        if self.feature_size is None:
            self.feature_size = all_features.shape[1]
        
        self.domain_classifier = SimpleNN(
            input_size=self.feature_size,
            hidden_sizes=[32],
            output_size=2,
            learning_rate=0.01
        )
        
        self.domain_classifier.fit(all_features, all_domain_labels, epochs=epochs, verbose=False)
        
        # Evaluate domain classification accuracy
        domain_predictions = self.domain_classifier.predict(all_features)
        domain_accuracy = np.mean(domain_predictions == all_domain_labels)
        
        return domain_accuracy

class TransferLearningFramework:
    """
    Comprehensive transfer learning framework
    """
    
    def __init__(self):
        """Initialize framework"""
        self.results = {}
    
    def create_source_task_data(self, n_samples: int = 1000, n_features: int = 20, 
                              n_classes: int = 5, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic source task data"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.8),
            n_redundant=int(n_features * 0.1),
            n_classes=n_classes,
            random_state=42
        )
        
        # Add noise
        X += np.random.normal(0, noise, X.shape)
        
        return X, y
    
    def create_target_task_data(self, source_X: np.ndarray, source_y: np.ndarray,
                              domain_shift: float = 0.5, 
                              task_similarity: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Create target task data with controlled domain shift"""
        n_samples = len(source_X) // 2  # Smaller target dataset
        n_features = source_X.shape[1]
        
        # Create target data with domain shift
        if task_similarity > 0.5:
            # Similar task: use source data as base with shift
            indices = np.random.choice(len(source_X), n_samples, replace=False)
            target_X = source_X[indices] + np.random.normal(0, domain_shift, (n_samples, n_features))
            target_y = source_y[indices]
            
            # Add some label noise for task difference
            if task_similarity < 1.0:
                noise_indices = np.random.choice(n_samples, int(n_samples * (1 - task_similarity)), replace=False)
                n_classes = len(np.unique(source_y))
                target_y[noise_indices] = np.random.choice(n_classes, len(noise_indices))
        else:
            # Different task: create new data
            target_X, target_y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=len(np.unique(source_y)),
                random_state=24
            )
            target_X += np.random.normal(0, domain_shift, target_X.shape)
        
        return target_X, target_y
    
    def train_source_model(self, X_source: np.ndarray, y_source: np.ndarray,
                          model_name: str = 'source_model') -> SimpleNN:
        """Train model on source task"""
        print(f"Training {model_name} on source task...")
        
        # Determine architecture
        n_features = X_source.shape[1]
        n_classes = len(np.unique(y_source))
        
        # Create and train source model
        source_model = SimpleNN(
            input_size=n_features,
            hidden_sizes=[128, 64],
            output_size=n_classes,
            learning_rate=0.01
        )
        
        source_model.fit(X_source, y_source, epochs=100, verbose=False)
        
        # Evaluate source performance
        source_predictions = source_model.predict(X_source)
        source_accuracy = np.mean(source_predictions == y_source)
        
        print(f"Source task accuracy: {source_accuracy:.4f}")
        
        return source_model
    
    def evaluate_transfer_learning(self, 
                                 source_model: SimpleNN,
                                 X_target: np.ndarray, 
                                 y_target: np.ndarray,
                                 test_size: float = 0.3) -> TransferResults:
        """Evaluate transfer learning performance"""
        
        # Split target data
        X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(
            X_target, y_target, test_size=test_size, random_state=42
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_target_scaled = scaler.fit_transform(X_train_target)
        X_test_target_scaled = scaler.transform(X_test_target)
        
        # 1. Evaluate source model performance on target (no adaptation)
        source_predictions = source_model.predict(X_test_target_scaled)
        source_accuracy = np.mean(source_predictions == y_test_target)
        
        # 2. Train from scratch on target data
        scratch_model = SimpleNN(
            input_size=X_target.shape[1],
            hidden_sizes=[128, 64],
            output_size=len(np.unique(y_target)),
            learning_rate=0.01
        )
        
        start_time = time.time()
        scratch_model.fit(X_train_target_scaled, y_train_target, epochs=100, verbose=False)
        scratch_time = time.time() - start_time
        
        scratch_predictions = scratch_model.predict(X_test_target_scaled)
        scratch_accuracy = np.mean(scratch_predictions == y_test_target)
        
        # 3. Transfer learning (feature extraction)
        transfer_model = TransferLearningNN(source_model, freeze_layers=1)
        
        start_time = time.time()
        transfer_model.fine_tune(
            X_train_target_scaled, y_train_target,
            epochs=50, freeze_base=True
        )
        transfer_time = time.time() - start_time
        
        transfer_predictions = transfer_model.predict(X_test_target_scaled)
        transfer_accuracy = np.mean(transfer_predictions == y_test_target)
        
        # 4. Data efficiency analysis
        data_efficiency = {}
        data_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        
        for fraction in data_fractions:
            n_samples = int(len(X_train_target_scaled) * fraction)
            X_subset = X_train_target_scaled[:n_samples]
            y_subset = y_train_target[:n_samples]
            
            # Transfer learning with limited data
            transfer_subset = TransferLearningNN(source_model, freeze_layers=1)
            transfer_subset.fine_tune(X_subset, y_subset, epochs=30, freeze_base=True)
            subset_predictions = transfer_subset.predict(X_test_target_scaled)
            subset_accuracy = np.mean(subset_predictions == y_test_target)
            
            data_efficiency[f'{int(fraction*100)}%'] = subset_accuracy
        
        return TransferResults(
            source_accuracy=source_accuracy,
            target_accuracy_scratch=scratch_accuracy,
            target_accuracy_transfer=transfer_accuracy,
            training_time_scratch=scratch_time,
            training_time_transfer=transfer_time,
            data_efficiency=data_efficiency
        )
    
    def compare_transfer_methods(self, 
                               source_model: SimpleNN,
                               X_target: np.ndarray, 
                               y_target: np.ndarray) -> Dict[str, float]:
        """Compare different transfer learning methods"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # 1. Feature Extraction (freeze base layers)
        feature_extractor = TransferLearningNN(source_model, freeze_layers=1)
        feature_extractor.fine_tune(X_train_scaled, y_train, freeze_base=True, epochs=50)
        fe_predictions = feature_extractor.predict(X_test_scaled)
        results['Feature Extraction'] = np.mean(fe_predictions == y_test)
        
        # 2. Fine-tuning (adapt all layers)
        fine_tuner = TransferLearningNN(source_model, freeze_layers=0)
        fine_tuner.fine_tune(X_train_scaled, y_train, freeze_base=False, epochs=30)
        ft_predictions = fine_tuner.predict(X_test_scaled)
        results['Fine-tuning'] = np.mean(ft_predictions == y_test)
        
        # 3. Domain Adaptation
        domain_adapter = DomainAdaptationModel(source_model)
        domain_accuracy = domain_adapter.add_domain_adversarial_training(
            X_train_scaled, X_test_scaled, y_train, epochs=30
        )
        results['Domain Adaptation'] = 1.0 - domain_accuracy  # Lower domain classification = better adaptation
        
        return results
    
    def visualize_transfer_results(self, results: Dict[str, TransferResults]):
        """Visualize transfer learning results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        methods = ['Source Only', 'From Scratch', 'Transfer Learning']
        
        for name, result in results.items():
            accuracies = [
                result.source_accuracy,
                result.target_accuracy_scratch,
                result.target_accuracy_transfer
            ]
            ax1.plot(methods, accuracies, marker='o', label=name)
        
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Transfer Learning Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training time comparison
        ax2 = axes[0, 1]
        for name, result in results.items():
            times = [result.training_time_scratch, result.training_time_transfer]
            ax2.bar([f'{name}\nScratch', f'{name}\nTransfer'], times, alpha=0.7)
        
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Data efficiency
        ax3 = axes[1, 0]
        for name, result in results.items():
            fractions = list(result.data_efficiency.keys())
            accuracies = list(result.data_efficiency.values())
            ax3.plot(fractions, accuracies, marker='s', label=name)
        
        ax3.set_xlabel('Training Data Fraction')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Data Efficiency Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Transfer learning benefit
        ax4 = axes[1, 1]
        for name, result in results.items():
            benefit = result.target_accuracy_transfer - result.target_accuracy_scratch
            speedup = result.training_time_scratch / result.training_time_transfer
            ax4.scatter(benefit, speedup, s=100, alpha=0.7, label=name)
        
        ax4.set_xlabel('Accuracy Improvement')
        ax4.set_ylabel('Training Speedup')
        ax4.set_title('Transfer Learning Benefits')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstration with real-world scenario
def demonstrate_transfer_learning():
    """Comprehensive transfer learning demonstration"""
    
    print("=== Transfer Learning Comprehensive Demonstration ===\n")
    
    # Initialize framework
    framework = TransferLearningFramework()
    
    # 1. Create source and target tasks
    print("1. Creating Source and Target Tasks")
    print("-" * 50)
    
    # Source task: General classification
    X_source, y_source = framework.create_source_task_data(
        n_samples=2000, n_features=20, n_classes=5, noise=0.1
    )
    
    print(f"Source task: {X_source.shape[0]} samples, {X_source.shape[1]} features, {len(np.unique(y_source))} classes")
    
    # Target task: Related but different classification
    X_target, y_target = framework.create_target_task_data(
        X_source, y_source, domain_shift=0.3, task_similarity=0.7
    )
    
    print(f"Target task: {X_target.shape[0]} samples, {X_target.shape[1]} features, {len(np.unique(y_target))} classes")
    
    # 2. Train source model
    print("\n2. Training Source Model")
    print("-" * 50)
    
    # Scale source data
    scaler_source = StandardScaler()
    X_source_scaled = scaler_source.fit_transform(X_source)
    
    source_model = framework.train_source_model(X_source_scaled, y_source)
    
    # 3. Evaluate transfer learning
    print("\n3. Evaluating Transfer Learning")
    print("-" * 50)
    
    results = framework.evaluate_transfer_learning(source_model, X_target, y_target)
    
    print(f"Source model on target: {results.source_accuracy:.4f}")
    print(f"Training from scratch: {results.target_accuracy_scratch:.4f}")
    print(f"Transfer learning: {results.target_accuracy_transfer:.4f}")
    print(f"Training time - Scratch: {results.training_time_scratch:.2f}s")
    print(f"Training time - Transfer: {results.training_time_transfer:.2f}s")
    print(f"Speedup: {results.training_time_scratch/results.training_time_transfer:.2f}x")
    
    # 4. Compare transfer methods
    print("\n4. Comparing Transfer Methods")
    print("-" * 50)
    
    method_results = framework.compare_transfer_methods(source_model, X_target, y_target)
    
    for method, accuracy in method_results.items():
        print(f"{method}: {accuracy:.4f}")
    
    # 5. Data efficiency analysis
    print("\n5. Data Efficiency Analysis")
    print("-" * 50)
    
    print("Transfer learning performance with limited data:")
    for fraction, accuracy in results.data_efficiency.items():
        print(f"  {fraction} of data: {accuracy:.4f}")
    
    # 6. Visualization
    print("\n6. Visualizing Results")
    print("-" * 50)
    
    scenario_results = {'Main Scenario': results}
    framework.visualize_transfer_results(scenario_results)
    
    return {
        'framework': framework,
        'source_model': source_model,
        'results': results,
        'method_comparison': method_results
    }

# Advanced transfer learning utilities
class TransferLearningAnalyzer:
    """Utility class for analyzing transfer learning experiments"""
    
    @staticmethod
    def analyze_feature_transferability(source_model: SimpleNN, 
                                      X_source: np.ndarray, 
                                      X_target: np.ndarray) -> Dict[str, float]:
        """Analyze how well features transfer between domains"""
        
        # Extract features from different layers
        transferability = {}
        
        # Feature similarity analysis
        source_features = source_model.forward(X_source)
        target_features = source_model.forward(X_target)
        
        # Calculate feature distribution similarity
        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)
        source_std = np.std(source_features, axis=0)
        target_std = np.std(target_features, axis=0)
        
        # Similarity metrics
        mean_similarity = np.corrcoef(source_mean, target_mean)[0, 1]
        std_similarity = np.corrcoef(source_std, target_std)[0, 1]
        
        transferability['mean_correlation'] = mean_similarity if not np.isnan(mean_similarity) else 0.0
        transferability['std_correlation'] = std_similarity if not np.isnan(std_similarity) else 0.0
        
        # Feature activation similarity
        activation_overlap = np.mean((source_features > 0) == (target_features > 0))
        transferability['activation_overlap'] = activation_overlap
        
        return transferability
    
    @staticmethod
    def plot_transfer_learning_timeline(training_losses: List[float], 
                                      validation_accuracies: List[float],
                                      title: str = "Transfer Learning Progress"):
        """Plot training progress for transfer learning"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training loss
        ax1.plot(training_losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Validation accuracy
        ax2.plot(validation_accuracies)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

# Real-world transfer learning example with digits
def digits_transfer_learning_example():
    """Transfer learning example with digit recognition"""
    
    print("=== Real-world Transfer Learning: Digit Recognition ===\n")
    
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Create source task: classify digits 0-4
    source_mask = y <= 4
    X_source = X[source_mask]
    y_source = y[source_mask]
    
    # Create target task: classify digits 5-9 (map to 0-4)
    target_mask = y > 4
    X_target = X[target_mask]
    y_target = y[target_mask] - 5  # Map 5-9 to 0-4
    
    print(f"Source task: Digits 0-4, {len(X_source)} samples")
    print(f"Target task: Digits 5-9, {len(X_target)} samples")
    
    # Scale data
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # Train source model
    print("\nTraining source model (digits 0-4)...")
    source_model = SimpleNN(
        input_size=64,  # 8x8 pixel images
        hidden_sizes=[128, 64],
        output_size=5,
        learning_rate=0.01
    )
    
    source_model.fit(X_source_scaled, y_source, epochs=100, verbose=False)
    source_accuracy = np.mean(source_model.predict(X_source_scaled) == y_source)
    print(f"Source model accuracy: {source_accuracy:.4f}")
    
    # Split target data
    X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(
        X_target_scaled, y_target, test_size=0.3, random_state=42
    )
    
    # Compare transfer methods
    print("\nComparing transfer learning approaches...")
    
    # 1. Direct application (no adaptation)
    direct_predictions = source_model.predict(X_test_target)
    direct_accuracy = np.mean(direct_predictions == y_test_target)
    print(f"Direct application: {direct_accuracy:.4f}")
    
    # 2. Training from scratch
    scratch_model = SimpleNN(64, [128, 64], 5, 0.01)
    scratch_model.fit(X_train_target, y_train_target, epochs=100, verbose=False)
    scratch_predictions = scratch_model.predict(X_test_target)
    scratch_accuracy = np.mean(scratch_predictions == y_test_target)
    print(f"Training from scratch: {scratch_accuracy:.4f}")
    
    # 3. Feature extraction
    feature_extractor = TransferLearningNN(source_model, freeze_layers=1)
    feature_extractor.fine_tune(X_train_target, y_train_target, freeze_base=True, epochs=50)
    fe_predictions = feature_extractor.predict(X_test_target)
    fe_accuracy = np.mean(fe_predictions == y_test_target)
    print(f"Feature extraction: {fe_accuracy:.4f}")
    
    # 4. Fine-tuning
    fine_tuner = TransferLearningNN(source_model, freeze_layers=0)
    fine_tuner.fine_tune(X_train_target, y_train_target, freeze_base=False, epochs=30)
    ft_predictions = fine_tuner.predict(X_test_target)
    ft_accuracy = np.mean(ft_predictions == y_test_target)
    print(f"Fine-tuning: {ft_accuracy:.4f}")
    
    # Analyze transferability
    analyzer = TransferLearningAnalyzer()
    transferability = analyzer.analyze_feature_transferability(
        source_model, X_source_scaled[:100], X_target_scaled[:100]
    )
    
    print(f"\nTransferability Analysis:")
    for metric, value in transferability.items():
        print(f"  {metric}: {value:.4f}")
    
    return {
        'source_accuracy': source_accuracy,
        'direct_accuracy': direct_accuracy,
        'scratch_accuracy': scratch_accuracy,
        'feature_extraction_accuracy': fe_accuracy,
        'fine_tuning_accuracy': ft_accuracy,
        'transferability': transferability
    }

# Run demonstrations
if __name__ == "__main__":
    # Main demonstration
    main_results = demonstrate_transfer_learning()
    
    # Real-world example
    print("\n" + "="*60)
    digits_results = digits_transfer_learning_example()
    
    print("\n=== Transfer Learning Summary ===")
    print("Transfer learning is effective when:")
    print("1. Source and target tasks are related")
    print("2. Source dataset is large and diverse")
    print("3. Target dataset is small")
    print("4. Computational resources are limited")
    
    print("\nKey benefits:")
    print("- Reduced training time")
    print("- Better performance with limited data")
    print("- Faster convergence")
    print("- Reduced overfitting")
    
    print("\n=== Transfer Learning Demonstration Complete ===")
```

This comprehensive implementation demonstrates:

### Core Transfer Learning Concepts:
1. **Feature Extraction** - Using pre-trained features without modification
2. **Fine-tuning** - Adapting pre-trained weights to new tasks
3. **Domain Adaptation** - Handling distribution shift between domains

### Advanced Features:
1. **Complete Neural Network** with transfer learning capabilities
2. **Multiple Transfer Strategies** (feature extraction, fine-tuning, domain adaptation)
3. **Real-world Example** with digit classification showing practical application
4. **Performance Comparison** tools for analyzing transfer effectiveness
5. **Data Efficiency Analysis** showing benefits with limited target data
6. **Transferability Metrics** for understanding feature reusability
7. **Comprehensive Visualizations** for results analysis

### Production-Ready Components:
1. **Modular Design** with abstract base classes for extensibility
2. **Proper Error Handling** and input validation
3. **Performance Monitoring** and analysis tools
4. **Scalable Architecture** supporting different model types
5. **Educational Features** with detailed explanations and demonstrations

---

## Question 14

**How do you implement recommendation systems using Python?**

**Answer:**

Recommendation systems are algorithms designed to suggest relevant items to users based on their preferences, behavior, or characteristics. They are essential for modern applications like e-commerce, streaming services, and social media platforms.

### Types of Recommendation Systems:

1. **Collaborative Filtering**: Recommends based on user-item interactions
   - User-based: Find similar users and recommend their preferred items
   - Item-based: Find similar items to those the user liked
2. **Content-Based Filtering**: Recommends based on item features and user preferences
3. **Matrix Factorization**: Decomposes user-item matrix to find latent factors
4. **Hybrid Systems**: Combine multiple approaches for better performance

### Common Challenges:

1. **Cold Start Problem**: New users/items with no historical data
2. **Sparsity**: Most users interact with few items
3. **Scalability**: Handling millions of users and items
4. **Diversity vs Accuracy**: Balancing recommendation relevance and variety

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import random

@dataclass
class RecommendationResult:
    """Container for recommendation results"""
    user_id: int
    item_recommendations: List[Tuple[int, float]]  # (item_id, score)
    explanation: str

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    rmse: float
    mae: float
    precision_at_k: float
    recall_at_k: float
    coverage: float
    diversity: float

class RecommenderSystem(ABC):
    """Abstract base class for recommendation systems"""
    
    @abstractmethod
    def fit(self, user_item_matrix: np.ndarray, **kwargs):
        """Train the recommender system"""
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate recommendations for a user"""
        pass

class CollaborativeFilteringUserBased(RecommenderSystem):
    """User-based Collaborative Filtering"""
    
    def __init__(self, n_neighbors: int = 50, similarity_metric: str = 'cosine'):
        """
        Initialize user-based collaborative filtering
        
        Args:
            n_neighbors: Number of similar users to consider
            similarity_metric: 'cosine' or 'euclidean'
        """
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_means = None
    
    def fit(self, user_item_matrix: np.ndarray, **kwargs):
        """Train the model by computing user similarities"""
        self.user_item_matrix = user_item_matrix.copy()
        
        # Handle missing values (0s) for similarity computation
        matrix_for_similarity = user_item_matrix.copy()
        matrix_for_similarity[matrix_for_similarity == 0] = np.nan
        
        # Compute user means (excluding zeros/NaN)
        self.user_means = np.nanmean(matrix_for_similarity, axis=1)
        
        # Center the ratings by subtracting user means
        centered_matrix = matrix_for_similarity - self.user_means.reshape(-1, 1)
        
        # Fill NaN with 0 for similarity computation
        centered_matrix = np.nan_to_num(centered_matrix, nan=0.0)
        
        # Compute user similarity matrix
        if self.similarity_metric == 'cosine':
            self.user_similarity = cosine_similarity(centered_matrix)
        elif self.similarity_metric == 'euclidean':
            distances = euclidean_distances(centered_matrix)
            # Convert distances to similarities
            self.user_similarity = 1 / (1 + distances)
        else:
            raise ValueError("similarity_metric must be 'cosine' or 'euclidean'")
        
        # Set self-similarity to 0 to avoid self-recommendation
        np.fill_diagonal(self.user_similarity, 0)
    
    def _get_similar_users(self, user_id: int) -> List[int]:
        """Get most similar users"""
        similarities = self.user_similarity[user_id]
        similar_users = np.argsort(similarities)[::-1][:self.n_neighbors]
        return similar_users[similarities[similar_users] > 0]  # Only positive similarities
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if self.user_item_matrix is None:
            raise ValueError("Model must be fitted first")
        
        # Get similar users who have rated this item
        similar_users = self._get_similar_users(user_id)
        relevant_users = [u for u in similar_users if self.user_item_matrix[u, item_id] > 0]
        
        if not relevant_users:
            return self.user_means[user_id]
        
        # Weighted average of similar users' ratings
        numerator = 0
        denominator = 0
        
        for similar_user in relevant_users:
            similarity = self.user_similarity[user_id, similar_user]
            rating = self.user_item_matrix[similar_user, item_id]
            user_mean = self.user_means[similar_user]
            
            numerator += similarity * (rating - user_mean)
            denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_means[user_id]
        
        predicted_rating = self.user_means[user_id] + (numerator / denominator)
        return max(1, min(5, predicted_rating))  # Clamp to rating scale
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate recommendations for a user"""
        if self.user_item_matrix is None:
            raise ValueError("Model must be fitted first")
        
        n_items = self.user_item_matrix.shape[1]
        item_scores = []
        
        for item_id in range(n_items):
            # Skip items already rated by user
            if exclude_seen and self.user_item_matrix[user_id, item_id] > 0:
                continue
            
            predicted_rating = self.predict(user_id, item_id)
            item_scores.append((item_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]

class CollaborativeFilteringItemBased(RecommenderSystem):
    """Item-based Collaborative Filtering"""
    
    def __init__(self, n_neighbors: int = 50, similarity_metric: str = 'cosine'):
        """Initialize item-based collaborative filtering"""
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_means = None
    
    def fit(self, user_item_matrix: np.ndarray, **kwargs):
        """Train the model by computing item similarities"""
        self.user_item_matrix = user_item_matrix.copy()
        
        # Transpose for item-based similarity
        item_user_matrix = user_item_matrix.T
        
        # Handle missing values
        matrix_for_similarity = item_user_matrix.copy()
        matrix_for_similarity[matrix_for_similarity == 0] = np.nan
        
        # Compute item means
        self.item_means = np.nanmean(matrix_for_similarity, axis=1)
        
        # Center the ratings
        centered_matrix = matrix_for_similarity - self.item_means.reshape(-1, 1)
        centered_matrix = np.nan_to_num(centered_matrix, nan=0.0)
        
        # Compute item similarity matrix
        if self.similarity_metric == 'cosine':
            self.item_similarity = cosine_similarity(centered_matrix)
        elif self.similarity_metric == 'euclidean':
            distances = euclidean_distances(centered_matrix)
            self.item_similarity = 1 / (1 + distances)
        
        np.fill_diagonal(self.item_similarity, 0)
    
    def _get_similar_items(self, item_id: int) -> List[int]:
        """Get most similar items"""
        similarities = self.item_similarity[item_id]
        similar_items = np.argsort(similarities)[::-1][:self.n_neighbors]
        return similar_items[similarities[similar_items] > 0]
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        # Get similar items that the user has rated
        similar_items = self._get_similar_items(item_id)
        relevant_items = [i for i in similar_items if self.user_item_matrix[user_id, i] > 0]
        
        if not relevant_items:
            return self.item_means[item_id]
        
        # Weighted average based on item similarities
        numerator = 0
        denominator = 0
        
        for similar_item in relevant_items:
            similarity = self.item_similarity[item_id, similar_item]
            rating = self.user_item_matrix[user_id, similar_item]
            
            numerator += similarity * rating
            denominator += abs(similarity)
        
        if denominator == 0:
            return self.item_means[item_id]
        
        predicted_rating = numerator / denominator
        return max(1, min(5, predicted_rating))
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate recommendations for a user"""
        n_items = self.user_item_matrix.shape[1]
        item_scores = []
        
        for item_id in range(n_items):
            if exclude_seen and self.user_item_matrix[user_id, item_id] > 0:
                continue
            
            predicted_rating = self.predict(user_id, item_id)
            item_scores.append((item_id, predicted_rating))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]

class MatrixFactorization(RecommenderSystem):
    """Matrix Factorization using Stochastic Gradient Descent"""
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01, 
                 regularization: float = 0.1, n_epochs: int = 100):
        """
        Initialize Matrix Factorization model
        
        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            regularization: L2 regularization parameter
            n_epochs: Number of training epochs
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.training_history = []
    
    def fit(self, user_item_matrix: np.ndarray, **kwargs):
        """Train the matrix factorization model"""
        self.user_item_matrix = user_item_matrix
        n_users, n_items = user_item_matrix.shape
        
        # Initialize factors and biases
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # Global mean of all ratings (excluding zeros)
        self.global_mean = np.mean(user_item_matrix[user_item_matrix > 0])
        
        # Get all rated user-item pairs
        rated_pairs = list(zip(*np.where(user_item_matrix > 0)))
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle the order of training pairs
            random.shuffle(rated_pairs)
            epoch_loss = 0
            
            for user_id, item_id in rated_pairs:
                # Predict rating
                prediction = self._predict_single(user_id, item_id)
                error = user_item_matrix[user_id, item_id] - prediction
                epoch_loss += error ** 2
                
                # Update biases
                user_bias_old = self.user_biases[user_id]
                item_bias_old = self.item_biases[item_id]
                
                self.user_biases[user_id] += self.learning_rate * (
                    error - self.regularization * self.user_biases[user_id]
                )
                self.item_biases[item_id] += self.learning_rate * (
                    error - self.regularization * self.item_biases[item_id]
                )
                
                # Update factors
                user_factors_old = self.user_factors[user_id, :].copy()
                
                self.user_factors[user_id, :] += self.learning_rate * (
                    error * self.item_factors[item_id, :] - 
                    self.regularization * self.user_factors[user_id, :]
                )
                
                self.item_factors[item_id, :] += self.learning_rate * (
                    error * user_factors_old - 
                    self.regularization * self.item_factors[item_id, :]
                )
            
            # Calculate RMSE for this epoch
            rmse = np.sqrt(epoch_loss / len(rated_pairs))
            self.training_history.append(rmse)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, RMSE: {rmse:.4f}")
    
    def _predict_single(self, user_id: int, item_id: int) -> float:
        """Predict rating for a single user-item pair"""
        prediction = (self.global_mean + 
                     self.user_biases[user_id] + 
                     self.item_biases[item_id] + 
                     np.dot(self.user_factors[user_id], self.item_factors[item_id]))
        return prediction
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if self.user_factors is None:
            raise ValueError("Model must be fitted first")
        
        prediction = self._predict_single(user_id, item_id)
        return max(1, min(5, prediction))  # Clamp to rating scale
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate recommendations for a user"""
        if self.user_factors is None:
            raise ValueError("Model must be fitted first")
        
        n_items = self.item_factors.shape[0]
        item_scores = []
        
        for item_id in range(n_items):
            if exclude_seen and self.user_item_matrix[user_id, item_id] > 0:
                continue
            
            predicted_rating = self.predict(user_id, item_id)
            item_scores.append((item_id, predicted_rating))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]
    
    def get_user_factors(self) -> np.ndarray:
        """Get user latent factors"""
        return self.user_factors
    
    def get_item_factors(self) -> np.ndarray:
        """Get item latent factors"""
        return self.item_factors

class ContentBasedRecommender:
    """Content-based recommendation system"""
    
    def __init__(self, similarity_metric: str = 'cosine'):
        """Initialize content-based recommender"""
        self.similarity_metric = similarity_metric
        self.item_features = None
        self.user_profiles = None
        self.feature_matrix = None
        self.tfidf_vectorizer = None
    
    def fit(self, user_item_matrix: np.ndarray, item_features: pd.DataFrame):
        """
        Train content-based recommender
        
        Args:
            user_item_matrix: User-item rating matrix
            item_features: DataFrame with item features (text descriptions, genres, etc.)
        """
        self.user_item_matrix = user_item_matrix
        self.item_features = item_features
        
        # Create TF-IDF features from item descriptions
        if 'description' in item_features.columns:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_features = self.tfidf_vectorizer.fit_transform(
                item_features['description'].fillna('')
            ).toarray()
            
            # Combine with other numerical features
            numerical_features = item_features.select_dtypes(include=[np.number]).values
            if numerical_features.size > 0:
                # Normalize numerical features
                numerical_features = (numerical_features - numerical_features.mean(axis=0)) / (
                    numerical_features.std(axis=0) + 1e-8
                )
                self.feature_matrix = np.hstack([tfidf_features, numerical_features])
            else:
                self.feature_matrix = tfidf_features
        else:
            # Use only numerical features
            numerical_features = item_features.select_dtypes(include=[np.number]).values
            self.feature_matrix = (numerical_features - numerical_features.mean(axis=0)) / (
                numerical_features.std(axis=0) + 1e-8
            )
        
        # Build user profiles as weighted average of liked items' features
        self._build_user_profiles()
    
    def _build_user_profiles(self):
        """Build user profiles based on their rating history"""
        n_users, n_items = self.user_item_matrix.shape
        n_features = self.feature_matrix.shape[1]
        self.user_profiles = np.zeros((n_users, n_features))
        
        for user_id in range(n_users):
            user_ratings = self.user_item_matrix[user_id]
            rated_items = np.where(user_ratings > 0)[0]
            
            if len(rated_items) > 0:
                # Weight features by ratings (prefer higher rated items)
                weights = user_ratings[rated_items]
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights)
                
                # Weighted average of item features
                self.user_profiles[user_id] = np.average(
                    self.feature_matrix[rated_items], 
                    weights=weights, 
                    axis=0
                )
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating based on content similarity"""
        if self.user_profiles is None:
            raise ValueError("Model must be fitted first")
        
        user_profile = self.user_profiles[user_id]
        item_features = self.feature_matrix[item_id]
        
        # Calculate similarity between user profile and item
        if self.similarity_metric == 'cosine':
            similarity = cosine_similarity([user_profile], [item_features])[0, 0]
        else:
            # Euclidean distance converted to similarity
            distance = euclidean_distances([user_profile], [item_features])[0, 0]
            similarity = 1 / (1 + distance)
        
        # Convert similarity to rating scale (1-5)
        predicted_rating = 1 + 4 * max(0, similarity)  # Map [0,1] to [1,5]
        return predicted_rating
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate content-based recommendations"""
        if self.user_profiles is None:
            raise ValueError("Model must be fitted first")
        
        n_items = self.feature_matrix.shape[0]
        item_scores = []
        
        for item_id in range(n_items):
            if exclude_seen and self.user_item_matrix[user_id, item_id] > 0:
                continue
            
            predicted_rating = self.predict(user_id, item_id)
            item_scores.append((item_id, predicted_rating))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]

class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches"""
    
    def __init__(self, recommenders: List[RecommenderSystem], weights: List[float]):
        """
        Initialize hybrid recommender
        
        Args:
            recommenders: List of trained recommender systems
            weights: Weights for each recommender (should sum to 1)
        """
        self.recommenders = recommenders
        self.weights = np.array(weights)
        
        if abs(sum(weights) - 1.0) > 1e-6:
            print("Warning: Weights don't sum to 1. Normalizing...")
            self.weights = self.weights / np.sum(self.weights)
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating as weighted combination of all recommenders"""
        predictions = []
        
        for recommender in self.recommenders:
            try:
                pred = recommender.predict(user_id, item_id)
                predictions.append(pred)
            except:
                # If a recommender fails, use global average
                predictions.append(3.0)
        
        # Weighted average
        final_prediction = np.average(predictions, weights=self.weights)
        return final_prediction
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate hybrid recommendations"""
        # Get recommendations from all systems
        all_recommendations = {}
        
        for i, recommender in enumerate(self.recommenders):
            try:
                recs = recommender.recommend(user_id, n_recommendations * 2, exclude_seen)
                weight = self.weights[i]
                
                for item_id, score in recs:
                    if item_id not in all_recommendations:
                        all_recommendations[item_id] = 0
                    all_recommendations[item_id] += weight * score
            except:
                continue
        
        # Sort by combined score
        sorted_recommendations = sorted(
            all_recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_recommendations[:n_recommendations]

class RecommendationEvaluator:
    """Utility class for evaluating recommendation systems"""
    
    @staticmethod
    def train_test_split_temporal(user_item_matrix: np.ndarray, 
                                test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Split data temporally (last ratings as test set)"""
        train_matrix = user_item_matrix.copy()
        test_matrix = np.zeros_like(user_item_matrix)
        
        for user_id in range(user_item_matrix.shape[0]):
            user_ratings = np.where(user_item_matrix[user_id] > 0)[0]
            
            if len(user_ratings) > 1:
                n_test = max(1, int(len(user_ratings) * test_ratio))
                test_items = np.random.choice(user_ratings, n_test, replace=False)
                
                for item_id in test_items:
                    test_matrix[user_id, item_id] = user_item_matrix[user_id, item_id]
                    train_matrix[user_id, item_id] = 0
        
        return train_matrix, test_matrix
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        mask = y_true > 0  # Only consider rated items
        if np.sum(mask) == 0:
            return float('inf')
        return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        mask = y_true > 0
        if np.sum(mask) == 0:
            return float('inf')
        return mean_absolute_error(y_true[mask], y_pred[mask])
    
    @staticmethod
    def calculate_precision_recall_at_k(recommendations: List[Tuple[int, float]], 
                                      relevant_items: List[int], 
                                      k: int = 10) -> Tuple[float, float]:
        """Calculate Precision@K and Recall@K"""
        if not recommendations or not relevant_items:
            return 0.0, 0.0
        
        recommended_items = [item_id for item_id, _ in recommendations[:k]]
        relevant_recommended = set(recommended_items) & set(relevant_items)
        
        precision = len(relevant_recommended) / len(recommended_items) if recommended_items else 0
        recall = len(relevant_recommended) / len(relevant_items) if relevant_items else 0
        
        return precision, recall
    
    @staticmethod
    def calculate_coverage(all_recommendations: List[List[Tuple[int, float]]], 
                         total_items: int) -> float:
        """Calculate catalog coverage"""
        recommended_items = set()
        for user_recommendations in all_recommendations:
            for item_id, _ in user_recommendations:
                recommended_items.add(item_id)
        
        return len(recommended_items) / total_items
    
    @staticmethod
    def calculate_diversity(recommendations: List[Tuple[int, float]], 
                          item_similarity_matrix: np.ndarray) -> float:
        """Calculate intra-list diversity"""
        if len(recommendations) < 2:
            return 0.0
        
        item_ids = [item_id for item_id, _ in recommendations]
        similarities = []
        
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                similarities.append(item_similarity_matrix[item_ids[i], item_ids[j]])
        
        # Diversity is 1 - average similarity
        return 1 - np.mean(similarities) if similarities else 0.0

def create_synthetic_movie_data(n_users: int = 500, n_movies: int = 100, 
                              sparsity: float = 0.1) -> Tuple[np.ndarray, pd.DataFrame]:
    """Create synthetic movie rating data for demonstration"""
    
    # Create user-item matrix
    user_item_matrix = np.zeros((n_users, n_movies))
    
    # Generate ratings based on user and movie preferences
    np.random.seed(42)
    
    # Create user preferences (latent factors)
    user_factors = np.random.normal(0, 1, (n_users, 5))
    movie_factors = np.random.normal(0, 1, (n_movies, 5))
    
    # Generate ratings
    for user_id in range(n_users):
        n_ratings = np.random.poisson(n_movies * sparsity)
        rated_movies = np.random.choice(n_movies, min(n_ratings, n_movies), replace=False)
        
        for movie_id in rated_movies:
            # Base rating from latent factors
            base_rating = np.dot(user_factors[user_id], movie_factors[movie_id])
            # Add noise and convert to 1-5 scale
            rating = max(1, min(5, int(3 + base_rating + np.random.normal(0, 0.5))))
            user_item_matrix[user_id, movie_id] = rating
    
    # Create movie features
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    movie_features = pd.DataFrame({
        'movie_id': range(n_movies),
        'year': np.random.randint(1990, 2024, n_movies),
        'runtime': np.random.randint(80, 180, n_movies),
        'budget': np.random.exponential(50, n_movies),  # Million dollars
        'genre': np.random.choice(genres, n_movies),
        'description': [f"A {np.random.choice(genres).lower()} movie about {np.random.choice(['love', 'adventure', 'mystery', 'friendship', 'betrayal'])}" 
                       for _ in range(n_movies)]
    })
    
    # Add genre dummy variables
    for genre in genres:
        movie_features[f'genre_{genre}'] = (movie_features['genre'] == genre).astype(int)
    
    return user_item_matrix, movie_features

def demonstrate_recommendation_systems():
    """Comprehensive demonstration of recommendation systems"""
    
    print("=== Recommendation Systems Comprehensive Demonstration ===\n")
    
    # 1. Create synthetic data
    print("1. Creating Synthetic Movie Dataset")
    print("-" * 50)
    
    user_item_matrix, movie_features = create_synthetic_movie_data(
        n_users=200, n_movies=50, sparsity=0.15
    )
    
    n_users, n_movies = user_item_matrix.shape
    n_ratings = np.sum(user_item_matrix > 0)
    sparsity = 1 - (n_ratings / (n_users * n_movies))
    
    print(f"Dataset created:")
    print(f"  Users: {n_users}")
    print(f"  Movies: {n_movies}")
    print(f"  Ratings: {n_ratings}")
    print(f"  Sparsity: {sparsity:.3f}")
    print(f"  Rating scale: {user_item_matrix[user_item_matrix > 0].min():.0f}-{user_item_matrix[user_item_matrix > 0].max():.0f}")
    
    # 2. Split data for evaluation
    print("\n2. Splitting Data for Evaluation")
    print("-" * 50)
    
    evaluator = RecommendationEvaluator()
    train_matrix, test_matrix = evaluator.train_test_split_temporal(user_item_matrix, test_ratio=0.2)
    
    print(f"Training ratings: {np.sum(train_matrix > 0)}")
    print(f"Test ratings: {np.sum(test_matrix > 0)}")
    
    # 3. Train and evaluate different recommendation algorithms
    print("\n3. Training Recommendation Systems")
    print("-" * 50)
    
    results = {}
    
    # User-based Collaborative Filtering
    print("Training User-based Collaborative Filtering...")
    user_cf = CollaborativeFilteringUserBased(n_neighbors=20)
    user_cf.fit(train_matrix)
    
    # Item-based Collaborative Filtering
    print("Training Item-based Collaborative Filtering...")
    item_cf = CollaborativeFilteringItemBased(n_neighbors=20)
    item_cf.fit(train_matrix)
    
    # Matrix Factorization
    print("Training Matrix Factorization...")
    mf = MatrixFactorization(n_factors=20, n_epochs=50, learning_rate=0.01)
    mf.fit(train_matrix)
    
    # Content-based Filtering
    print("Training Content-based Filtering...")
    content_cf = ContentBasedRecommender()
    content_cf.fit(train_matrix, movie_features)
    
    # Hybrid System
    print("Creating Hybrid System...")
    hybrid = HybridRecommender(
        recommenders=[user_cf, item_cf, mf, content_cf],
        weights=[0.3, 0.3, 0.3, 0.1]
    )
    
    # 4. Evaluate all systems
    print("\n4. Evaluating Recommendation Systems")
    print("-" * 50)
    
    systems = {
        'User-based CF': user_cf,
        'Item-based CF': item_cf,
        'Matrix Factorization': mf,
        'Content-based': content_cf,
        'Hybrid': hybrid
    }
    
    for name, system in systems.items():
        print(f"\nEvaluating {name}...")
        
        # Predict ratings for test set
        predictions = np.zeros_like(test_matrix)
        
        for user_id in range(n_users):
            for item_id in range(n_movies):
                if test_matrix[user_id, item_id] > 0:
                    try:
                        predictions[user_id, item_id] = system.predict(user_id, item_id)
                    except:
                        predictions[user_id, item_id] = 3.0  # Default prediction
        
        # Calculate metrics
        rmse = evaluator.calculate_rmse(test_matrix, predictions)
        mae = evaluator.calculate_mae(test_matrix, predictions)
        
        # Calculate recommendation metrics for a sample of users
        precision_scores = []
        recall_scores = []
        
        for user_id in range(min(50, n_users)):  # Sample 50 users
            if np.sum(train_matrix[user_id] > 0) > 0:  # User has training data
                recommendations = system.recommend(user_id, n_recommendations=10)
                relevant_items = list(np.where(test_matrix[user_id] >= 4)[0])  # High ratings as relevant
                
                precision, recall = evaluator.calculate_precision_recall_at_k(
                    recommendations, relevant_items, k=10
                )
                precision_scores.append(precision)
                recall_scores.append(recall)
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'Precision@10': avg_precision,
            'Recall@10': avg_recall
        }
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Precision@10: {avg_precision:.4f}")
        print(f"  Recall@10: {avg_recall:.4f}")
    
    # 5. Generate sample recommendations
    print("\n5. Sample Recommendations")
    print("-" * 50)
    
    sample_user = 0
    print(f"\nRecommendations for User {sample_user}:")
    print(f"User's previous ratings:")
    
    user_ratings = np.where(train_matrix[sample_user] > 0)[0]
    for item_id in user_ratings[:5]:  # Show first 5 ratings
        rating = train_matrix[sample_user, item_id]
        genre = movie_features.iloc[item_id]['genre']
        print(f"  Movie {item_id} ({genre}): {rating:.0f} stars")
    
    print(f"\nTop recommendations:")
    for system_name, system in systems.items():
        recommendations = system.recommend(sample_user, n_recommendations=5)
        print(f"\n{system_name}:")
        for i, (item_id, score) in enumerate(recommendations, 1):
            genre = movie_features.iloc[item_id]['genre']
            print(f"  {i}. Movie {item_id} ({genre}): {score:.2f}")
    
    # 6. Visualize results
    print("\n6. Visualizing Results")
    print("-" * 50)
    
    # Performance comparison
    plt.figure(figsize=(15, 10))
    
    # RMSE comparison
    plt.subplot(2, 2, 1)
    systems_names = list(results.keys())
    rmse_values = [results[name]['RMSE'] for name in systems_names]
    plt.bar(systems_names, rmse_values, alpha=0.7, color='skyblue')
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Precision@10 comparison
    plt.subplot(2, 2, 2)
    precision_values = [results[name]['Precision@10'] for name in systems_names]
    plt.bar(systems_names, precision_values, alpha=0.7, color='lightgreen')
    plt.title('Precision@10 Comparison')
    plt.ylabel('Precision@10')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Training history for Matrix Factorization
    plt.subplot(2, 2, 3)
    plt.plot(mf.training_history)
    plt.title('Matrix Factorization Training History')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    
    # User-Item matrix visualization (sample)
    plt.subplot(2, 2, 4)
    sample_matrix = train_matrix[:20, :20]  # 20x20 sample
    plt.imshow(sample_matrix, cmap='YlOrRd', aspect='auto')
    plt.title('User-Item Matrix Sample')
    plt.xlabel('Movies')
    plt.ylabel('Users')
    plt.colorbar(label='Rating')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'user_item_matrix': user_item_matrix,
        'movie_features': movie_features,
        'systems': systems,
        'results': results,
        'train_matrix': train_matrix,
        'test_matrix': test_matrix
    }

def demonstrate_cold_start_problem():
    """Demonstrate solutions to the cold start problem"""
    
    print("=== Cold Start Problem Demonstration ===\n")
    
    # Create data
    user_item_matrix, movie_features = create_synthetic_movie_data(n_users=100, n_movies=30)
    
    # Simulate new user (no ratings)
    new_user_id = user_item_matrix.shape[0]
    new_user_ratings = np.zeros(user_item_matrix.shape[1])
    
    # Solutions for cold start
    print("1. Content-based Recommendations for New User")
    print("-" * 50)
    
    # Solution 1: Use popular items
    item_popularity = np.sum(user_item_matrix > 0, axis=0)
    popular_items = np.argsort(item_popularity)[::-1][:10]
    
    print("Most popular items:")
    for i, item_id in enumerate(popular_items[:5], 1):
        genre = movie_features.iloc[item_id]['genre']
        popularity = item_popularity[item_id]
        print(f"  {i}. Movie {item_id} ({genre}): {popularity} ratings")
    
    # Solution 2: Use content-based filtering with user preferences
    print("\n2. Content-based with User Preferences")
    print("-" * 50)
    
    # Simulate user providing genre preferences
    preferred_genres = ['Action', 'Sci-Fi']
    print(f"User prefers: {', '.join(preferred_genres)}")
    
    # Recommend based on content
    content_scores = []
    for item_id in range(len(movie_features)):
        score = 0
        movie_genre = movie_features.iloc[item_id]['genre']
        if movie_genre in preferred_genres:
            score += 1
        
        # Add other features
        year = movie_features.iloc[item_id]['year']
        if year >= 2010:  # Recent movies
            score += 0.5
        
        content_scores.append((item_id, score))
    
    content_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Content-based recommendations:")
    for i, (item_id, score) in enumerate(content_scores[:5], 1):
        genre = movie_features.iloc[item_id]['genre']
        year = movie_features.iloc[item_id]['year']
        print(f"  {i}. Movie {item_id} ({genre}, {year}): {score:.1f}")
    
    # Solution 3: Hybrid approach
    print("\n3. Hybrid Approach (Popularity + Content)")
    print("-" * 50)
    
    hybrid_scores = []
    for item_id in range(len(movie_features)):
        # Combine popularity and content scores
        popularity_score = item_popularity[item_id] / np.max(item_popularity)
        content_score = dict(content_scores)[item_id] / max(1, max(score for _, score in content_scores))
        
        hybrid_score = 0.6 * popularity_score + 0.4 * content_score
        hybrid_scores.append((item_id, hybrid_score))
    
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Hybrid recommendations:")
    for i, (item_id, score) in enumerate(hybrid_scores[:5], 1):
        genre = movie_features.iloc[item_id]['genre']
        popularity = item_popularity[item_id]
        print(f"  {i}. Movie {item_id} ({genre}): {score:.3f} (pop: {popularity})")

def analyze_recommendation_diversity():
    """Analyze and improve recommendation diversity"""
    
    print("=== Recommendation Diversity Analysis ===\n")
    
    # Create data and train systems
    user_item_matrix, movie_features = create_synthetic_movie_data(n_users=100, n_movies=40)
    
    # Train a simple collaborative filtering system
    cf_system = CollaborativeFilteringUserBased(n_neighbors=10)
    cf_system.fit(user_item_matrix)
    
    # Sample user
    user_id = 0
    
    # Get standard recommendations
    standard_recs = cf_system.recommend(user_id, n_recommendations=10)
    
    print("1. Standard Recommendations")
    print("-" * 50)
    genre_count = {}
    for i, (item_id, score) in enumerate(standard_recs, 1):
        genre = movie_features.iloc[item_id]['genre']
        genre_count[genre] = genre_count.get(genre, 0) + 1
        print(f"  {i}. Movie {item_id} ({genre}): {score:.3f}")
    
    print(f"\nGenre distribution: {genre_count}")
    
    # Implement diversity-aware recommendations
    print("\n2. Diversity-aware Recommendations")
    print("-" * 50)
    
    def diversified_recommendations(cf_system, user_id, n_recommendations=10, lambda_div=0.3):
        """Generate diversified recommendations using MMR-like approach"""
        
        # Get initial larger set of recommendations
        candidate_recs = cf_system.recommend(user_id, n_recommendations * 3)
        
        if not candidate_recs:
            return []
        
        selected_items = []
        remaining_items = list(candidate_recs)
        
        # Select first item (highest score)
        first_item = remaining_items.pop(0)
        selected_items.append(first_item)
        
        # Iteratively select diverse items
        while len(selected_items) < n_recommendations and remaining_items:
            best_item = None
            best_score = -float('inf')
            best_idx = -1
            
            for idx, (item_id, relevance_score) in enumerate(remaining_items):
                # Calculate diversity score (genre diversity)
                item_genre = movie_features.iloc[item_id]['genre']
                selected_genres = [movie_features.iloc[sel_id]['genre'] for sel_id, _ in selected_items]
                
                genre_diversity = 1.0 if item_genre not in selected_genres else 0.5
                
                # Combined score: relevance + diversity
                combined_score = (1 - lambda_div) * relevance_score + lambda_div * genre_diversity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_item = (item_id, relevance_score)
                    best_idx = idx
            
            if best_item:
                selected_items.append(best_item)
                remaining_items.pop(best_idx)
        
        return selected_items
    
    diverse_recs = diversified_recommendations(cf_system, user_id, n_recommendations=10)
    
    diverse_genre_count = {}
    for i, (item_id, score) in enumerate(diverse_recs, 1):
        genre = movie_features.iloc[item_id]['genre']
        diverse_genre_count[genre] = diverse_genre_count.get(genre, 0) + 1
        print(f"  {i}. Movie {item_id} ({genre}): {score:.3f}")
    
    print(f"\nDiverse genre distribution: {diverse_genre_count}")
    
    # Calculate diversity metrics
    standard_unique_genres = len(genre_count)
    diverse_unique_genres = len(diverse_genre_count)
    
    print(f"\n3. Diversity Metrics")
    print("-" * 50)
    print(f"Standard recommendations unique genres: {standard_unique_genres}")
    print(f"Diverse recommendations unique genres: {diverse_unique_genres}")
    print(f"Diversity improvement: {(diverse_unique_genres - standard_unique_genres) / standard_unique_genres * 100:.1f}%")

# Run comprehensive demonstration
if __name__ == "__main__":
    # Main demonstration
    main_results = demonstrate_recommendation_systems()
    
    # Cold start problem
    print("\n" + "="*60)
    demonstrate_cold_start_problem()
    
    # Diversity analysis
    print("\n" + "="*60)
    analyze_recommendation_diversity()
    
    print("\n=== Recommendation Systems Summary ===")
    print("Key Algorithms:")
    print("1. Collaborative Filtering - Uses user-item interactions")
    print("2. Content-based Filtering - Uses item features")
    print("3. Matrix Factorization - Finds latent factors")
    print("4. Hybrid Systems - Combines multiple approaches")
    
    print("\nKey Challenges:")
    print("1. Cold Start - New users/items with no data")
    print("2. Sparsity - Limited user-item interactions")
    print("3. Scalability - Handling large datasets")
    print("4. Diversity - Avoiding filter bubbles")
    
    print("\nEvaluation Metrics:")
    print("1. RMSE/MAE - Prediction accuracy")
    print("2. Precision/Recall@K - Recommendation quality")
    print("3. Coverage - Catalog diversity")
    print("4. Novelty/Serendipity - Discovery potential")
    
    print("\n=== Recommendation Systems Demonstration Complete ===")
```

This comprehensive implementation demonstrates:

### Core Recommendation Algorithms:
1. **User-based Collaborative Filtering** - Finds similar users and recommends their preferences
2. **Item-based Collaborative Filtering** - Recommends similar items to user's preferences  
3. **Matrix Factorization** - Uses latent factors to predict ratings
4. **Content-based Filtering** - Recommends based on item features
5. **Hybrid Systems** - Combines multiple approaches for better performance

### Advanced Features:
1. **Cold Start Solutions** - Handles new users/items with no historical data
2. **Diversity-aware Recommendations** - Prevents filter bubbles and improves discovery
3. **Comprehensive Evaluation** - RMSE, MAE, Precision@K, Recall@K, Coverage
4. **Real-world Challenges** - Sparsity handling, scalability considerations
5. **Visualization Tools** - Performance comparisons and data analysis

### Production-Ready Components:
1. **Modular Architecture** - Easy to extend and modify
2. **Proper Error Handling** - Robust to edge cases
3. **Evaluation Framework** - Comprehensive testing and validation
4. **Synthetic Data Generation** - For testing and demonstration
5. **Performance Monitoring** - Training history and convergence analysis

---

## Question 15

**Explain the concept of clustering and implement K-Means clustering.**

**Answer:**

Clustering is an unsupervised machine learning technique that groups similar data points together while keeping dissimilar points in different clusters. It's used for pattern discovery, data exploration, and customer segmentation.

### Key Concepts:

1. **Unsupervised Learning**: No target labels are provided
2. **Similarity/Distance**: Points are grouped based on similarity measures
3. **Cluster Centers**: Representative points for each cluster
4. **Inertia**: Within-cluster sum of squared distances (WCSS)
5. **Silhouette Score**: Measure of clustering quality

### Types of Clustering:

1. **Partitional**: K-Means, K-Medoids
2. **Hierarchical**: Agglomerative, Divisive
3. **Density-based**: DBSCAN, OPTICS
4. **Model-based**: Gaussian Mixture Models

### K-Means Algorithm:
1. Choose number of clusters (k)
2. Initialize cluster centers randomly
3. Assign each point to nearest center
4. Update centers to cluster centroids
5. Repeat steps 3-4 until convergence

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from sklearn.datasets import make_blobs, make_circles, make_moons, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.cluster import KMeans as SklearnKMeans, DBSCAN, AgglomerativeClustering
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ClusteringResults:
    """Container for clustering results"""
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    silhouette_score: float
    n_iterations: int

class KMeansClusterer:
    """
    Implementation of K-Means clustering algorithm from scratch
    """
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 300, 
                 tol: float = 1e-4, random_state: Optional[int] = None):
        """
        Initialize K-Means clusterer
        
        Args:
            n_clusters: Number of clusters
            max_iters: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
        # Results
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.history_ = []
    
    def _initialize_centers(self, X: np.ndarray, method: str = 'random') -> np.ndarray:
        """Initialize cluster centers"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        if method == 'random':
            # Random initialization
            centers = np.random.uniform(
                X.min(axis=0), X.max(axis=0), 
                size=(self.n_clusters, n_features)
            )
        elif method == 'kmeans++':
            # K-Means++ initialization
            centers = self._kmeans_plus_plus_init(X)
        else:
            raise ValueError("method must be 'random' or 'kmeans++'")
        
        return centers
    
    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ initialization for better center selection"""
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features))
        
        # Choose first center randomly
        centers[0] = X[np.random.randint(n_samples)]
        
        # Choose remaining centers
        for c_id in range(1, self.n_clusters):
            # Calculate distances to nearest center
            distances = np.array([
                min([np.sum((x - c)**2) for c in centers[:c_id]]) 
                for x in X
            ])
            
            # Choose next center with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centers[c_id] = X[j]
                    break
        
        return centers
    
    def _assign_clusters(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign each point to nearest cluster center"""
        distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update cluster centers to centroids of assigned points"""
        centers = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centers[k] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centers[k] = X[np.random.randint(len(X))]
        
        return centers
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
        """Calculate within-cluster sum of squared distances"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centers[k])**2)
        return inertia
    
    def fit(self, X: np.ndarray, init_method: str = 'kmeans++') -> 'KMeansClusterer':
        """
        Fit K-Means clustering to data
        
        Args:
            X: Data array of shape (n_samples, n_features)
            init_method: Initialization method ('random' or 'kmeans++')
        """
        X = np.array(X)
        self.history_ = []
        
        # Initialize centers
        centers = self._initialize_centers(X, method=init_method)
        
        # Main K-Means loop
        for iteration in range(self.max_iters):
            # Assign points to clusters
            labels = self._assign_clusters(X, centers)
            
            # Calculate inertia
            inertia = self._calculate_inertia(X, labels, centers)
            
            # Store history
            self.history_.append({
                'iteration': iteration,
                'inertia': inertia,
                'centers': centers.copy()
            })
            
            # Update centers
            new_centers = self._update_centers(X, labels)
            
            # Check for convergence
            center_shift = np.sqrt(np.sum((centers - new_centers)**2))
            if center_shift < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            centers = new_centers
        
        # Store final results
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted first")
        
        return self._assign_clusters(np.array(X), self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray, init_method: str = 'kmeans++') -> np.ndarray:
        """Fit model and return cluster labels"""
        return self.fit(X, init_method).labels_

class AdvancedKMeans:
    """Advanced K-Means with additional features"""
    
    def __init__(self, max_clusters: int = 10, random_state: Optional[int] = None):
        """Initialize advanced K-Means"""
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.results_ = {}
    
    def find_optimal_clusters(self, X: np.ndarray, 
                            methods: List[str] = ['elbow', 'silhouette']) -> Dict[str, int]:
        """
        Find optimal number of clusters using multiple methods
        
        Args:
            X: Data array
            methods: List of methods to use ['elbow', 'silhouette', 'gap']
        
        Returns:
            Dictionary with optimal k for each method
        """
        X = np.array(X)
        optimal_k = {}
        
        # Test different values of k
        k_range = range(2, min(self.max_clusters + 1, len(X)))
        
        if 'elbow' in methods:
            optimal_k['elbow'] = self._elbow_method(X, k_range)
        
        if 'silhouette' in methods:
            optimal_k['silhouette'] = self._silhouette_method(X, k_range)
        
        if 'gap' in methods:
            optimal_k['gap'] = self._gap_statistic(X, k_range)
        
        return optimal_k
    
    def _elbow_method(self, X: np.ndarray, k_range: range) -> int:
        """Find optimal k using elbow method"""
        inertias = []
        
        for k in k_range:
            kmeans = KMeansClusterer(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point using second derivative
        if len(inertias) < 3:
            return k_range[0]
        
        # Calculate second differences
        second_diffs = []
        for i in range(1, len(inertias) - 1):
            second_diff = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_diffs.append(second_diff)
        
        # Find the point with maximum second difference
        elbow_idx = np.argmax(second_diffs) + 1  # +1 because we start from index 1
        optimal_k = list(k_range)[elbow_idx]
        
        self.results_['elbow'] = {
            'k_range': list(k_range),
            'inertias': inertias,
            'optimal_k': optimal_k
        }
        
        return optimal_k
    
    def _silhouette_method(self, X: np.ndarray, k_range: range) -> int:
        """Find optimal k using silhouette analysis"""
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeansClusterer(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[optimal_idx]
        
        self.results_['silhouette'] = {
            'k_range': list(k_range),
            'scores': silhouette_scores,
            'optimal_k': optimal_k
        }
        
        return optimal_k
    
    def _gap_statistic(self, X: np.ndarray, k_range: range, n_refs: int = 10) -> int:
        """Find optimal k using gap statistic"""
        gaps = []
        
        for k in k_range:
            # Compute inertia for actual data
            kmeans = KMeansClusterer(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X)
            actual_inertia = kmeans.inertia_
            
            # Compute expected inertia from reference datasets
            ref_inertias = []
            for _ in range(n_refs):
                # Generate random reference data
                ref_data = np.random.uniform(
                    X.min(axis=0), X.max(axis=0), size=X.shape
                )
                ref_kmeans = KMeansClusterer(n_clusters=k, random_state=None)
                ref_kmeans.fit(ref_data)
                ref_inertias.append(ref_kmeans.inertia_)
            
            expected_inertia = np.mean(ref_inertias)
            
            # Calculate gap
            gap = np.log(expected_inertia) - np.log(actual_inertia)
            gaps.append(gap)
        
        # Find optimal k (first k where gap starts decreasing)
        optimal_k = list(k_range)[0]  # Default
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1]:
                optimal_k = list(k_range)[i]
                break
        
        self.results_['gap'] = {
            'k_range': list(k_range),
            'gaps': gaps,
            'optimal_k': optimal_k
        }
        
        return optimal_k
    
    def plot_optimization_results(self, figsize: Tuple[int, int] = (15, 5)):
        """Plot results of cluster optimization methods"""
        n_methods = len(self.results_)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method, results) in enumerate(self.results_.items()):
            ax = axes[idx]
            
            if method == 'elbow':
                ax.plot(results['k_range'], results['inertias'], 'bo-')
                ax.axvline(x=results['optimal_k'], color='red', linestyle='--', 
                          label=f'Optimal k={results["optimal_k"]}')
                ax.set_ylabel('Inertia')
                ax.set_title('Elbow Method')
                
            elif method == 'silhouette':
                ax.plot(results['k_range'], results['scores'], 'go-')
                ax.axvline(x=results['optimal_k'], color='red', linestyle='--',
                          label=f'Optimal k={results["optimal_k"]}')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Silhouette Analysis')
                
            elif method == 'gap':
                ax.plot(results['k_range'], results['gaps'], 'mo-')
                ax.axvline(x=results['optimal_k'], color='red', linestyle='--',
                          label=f'Optimal k={results["optimal_k"]}')
                ax.set_ylabel('Gap Statistic')
                ax.set_title('Gap Statistic')
            
            ax.set_xlabel('Number of Clusters (k)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def create_clustering_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create various datasets for clustering demonstration"""
    
    datasets = {}
    
    # 1. Well-separated blobs
    X_blobs, y_blobs = make_blobs(
        n_samples=300, centers=4, n_features=2,
        random_state=42, cluster_std=1.0
    )
    datasets['blobs'] = (X_blobs, y_blobs)
    
    # 2. Circles
    X_circles, y_circles = make_circles(
        n_samples=300, noise=0.1, factor=0.3, random_state=42
    )
    datasets['circles'] = (X_circles, y_circles)
    
    # 3. Moons
    X_moons, y_moons = make_moons(
        n_samples=300, noise=0.1, random_state=42
    )
    datasets['moons'] = (X_moons, y_moons)
    
    # 4. Random data
    np.random.seed(42)
    X_random = np.random.randn(300, 2)
    y_random = np.zeros(300)  # No true clusters
    datasets['random'] = (X_random, y_random)
    
    # 5. Iris dataset (real data)
    iris = load_iris()
    # Use only first 2 features for visualization
    X_iris = iris.data[:, :2]
    y_iris = iris.target
    datasets['iris'] = (X_iris, y_iris)
    
    return datasets

def demonstrate_kmeans_clustering():
    """Comprehensive demonstration of K-Means clustering"""
    
    print("=== K-Means Clustering Comprehensive Demonstration ===\n")
    
    # 1. Create demonstration datasets
    print("1. Creating Demonstration Datasets")
    print("-" * 50)
    
    datasets = create_clustering_datasets()
    
    for name, (X, y) in datasets.items():
        print(f"{name.capitalize()}: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} true clusters")
    
    # 2. Demonstrate basic K-Means
    print("\n2. Basic K-Means Implementation")
    print("-" * 50)
    
    # Use blobs dataset for clear demonstration
    X_demo, y_true = datasets['blobs']
    
    # Fit K-Means
    kmeans = KMeansClusterer(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_demo)
    
    print(f"Converged in {kmeans.n_iter_} iterations")
    print(f"Final inertia: {kmeans.inertia_:.2f}")
    print(f"Silhouette score: {silhouette_score(X_demo, labels):.3f}")
    
    # 3. Compare initialization methods
    print("\n3. Comparing Initialization Methods")
    print("-" * 50)
    
    init_methods = ['random', 'kmeans++']
    init_results = {}
    
    for method in init_methods:
        kmeans_init = KMeansClusterer(n_clusters=4, random_state=42)
        labels_init = kmeans_init.fit_predict(X_demo, init_method=method)
        
        init_results[method] = {
            'inertia': kmeans_init.inertia_,
            'iterations': kmeans_init.n_iter_,
            'silhouette': silhouette_score(X_demo, labels_init)
        }
        
        print(f"{method.capitalize()}:")
        print(f"  Inertia: {init_results[method]['inertia']:.2f}")
        print(f"  Iterations: {init_results[method]['iterations']}")
        print(f"  Silhouette: {init_results[method]['silhouette']:.3f}")
    
    # 4. Find optimal number of clusters
    print("\n4. Finding Optimal Number of Clusters")
    print("-" * 50)
    
    advanced_kmeans = AdvancedKMeans(max_clusters=8, random_state=42)
    optimal_clusters = advanced_kmeans.find_optimal_clusters(
        X_demo, methods=['elbow', 'silhouette']
    )
    
    print("Optimal number of clusters:")
    for method, k in optimal_clusters.items():
        print(f"  {method.capitalize()}: {k}")
    
    # Plot optimization results
    advanced_kmeans.plot_optimization_results()
    
    # 5. Compare with other clustering algorithms
    print("\n5. Comparing with Other Clustering Algorithms")
    print("-" * 50)
    
    # Scale data for fair comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_demo)
    
    clustering_algorithms = {
        'K-Means (Custom)': KMeansClusterer(n_clusters=4, random_state=42),
        'K-Means (Sklearn)': SklearnKMeans(n_clusters=4, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Agglomerative': AgglomerativeClustering(n_clusters=4)
    }
    
    comparison_results = {}
    
    for name, algorithm in clustering_algorithms.items():
        if hasattr(algorithm, 'fit_predict'):
            labels_comp = algorithm.fit_predict(X_scaled)
        else:
            labels_comp = algorithm.fit(X_scaled).labels_
        
        # Handle noise points in DBSCAN (labeled as -1)
        if -1 in labels_comp:
            n_clusters = len(np.unique(labels_comp[labels_comp != -1]))
            n_noise = np.sum(labels_comp == -1)
            print(f"{name}: {n_clusters} clusters, {n_noise} noise points")
        else:
            n_clusters = len(np.unique(labels_comp))
            print(f"{name}: {n_clusters} clusters")
        
        # Calculate metrics (only if we have multiple clusters)
        if len(np.unique(labels_comp)) > 1:
            silhouette = silhouette_score(X_scaled, labels_comp)
            ari = adjusted_rand_score(y_true, labels_comp)
            
            comparison_results[name] = {
                'silhouette': silhouette,
                'ari': ari,
                'labels': labels_comp
            }
            
            print(f"  Silhouette: {silhouette:.3f}")
            print(f"  ARI: {ari:.3f}")
        else:
            print(f"  Cannot calculate metrics (only 1 cluster)")
    
    # 6. Visualize clustering results
    print("\n6. Visualizing Clustering Results")
    print("-" * 50)
    
    # Plot all datasets with clustering results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, (dataset_name, (X, y_true)) in enumerate(datasets.items()):
        if idx >= 6:  # Limit to 6 plots
            break
        
        ax = axes[idx]
        
        # Apply K-Means
        if X.shape[1] > 2:
            # Use PCA for dimensionality reduction if needed
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
        else:
            X_plot = X
        
        # Determine optimal k for this dataset
        if len(np.unique(y_true)) > 1:
            optimal_k = len(np.unique(y_true))
        else:
            optimal_k = 3  # Default for random data
        
        kmeans_vis = KMeansClusterer(n_clusters=optimal_k, random_state=42)
        labels_vis = kmeans_vis.fit_predict(X_plot)
        
        # Plot results
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels_vis, 
                           cmap='viridis', alpha=0.7, s=50)
        
        # Plot centroids
        centers = kmeans_vis.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', 
                  s=200, linewidths=3, label='Centroids')
        
        ax.set_title(f'{dataset_name.capitalize()} Dataset\n'
                    f'K={optimal_k}, Silhouette={silhouette_score(X_plot, labels_vis):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 7. Demonstrate convergence behavior
    print("\n7. K-Means Convergence Analysis")
    print("-" * 50)
    
    # Show convergence history
    convergence_kmeans = KMeansClusterer(n_clusters=4, random_state=42)
    convergence_kmeans.fit(X_demo)
    
    # Plot convergence
    iterations = [h['iteration'] for h in convergence_kmeans.history_]
    inertias = [h['inertia'] for h in convergence_kmeans.history_]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, inertias, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Iteration')
    plt.ylabel('Inertia (WCSS)')
    plt.title('K-Means Convergence')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Algorithm converged after {len(iterations)} iterations")
    print(f"Final inertia: {inertias[-1]:.2f}")
    
    return {
        'datasets': datasets,
        'kmeans_model': kmeans,
        'comparison_results': comparison_results,
        'optimal_clusters': optimal_clusters
    }

def analyze_clustering_challenges():
    """Analyze common challenges in K-Means clustering"""
    
    print("=== K-Means Clustering Challenges Analysis ===\n")
    
    # 1. Sensitivity to initialization
    print("1. Sensitivity to Initialization")
    print("-" * 50)
    
    X_blob, _ = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=2.0)
    
    # Run K-Means multiple times with different random states
    initialization_results = []
    
    for i in range(5):
        kmeans_init = KMeansClusterer(n_clusters=3, random_state=i)
        kmeans_init.fit(X_blob, init_method='random')
        initialization_results.append({
            'random_state': i,
            'inertia': kmeans_init.inertia_,
            'iterations': kmeans_init.n_iter_
        })
    
    print("Results with different random initializations:")
    for result in initialization_results:
        print(f"  Random state {result['random_state']}: "
              f"Inertia={result['inertia']:.2f}, "
              f"Iterations={result['iterations']}")
    
    # Compare with K-Means++
    kmeans_plus = KMeansClusterer(n_clusters=3, random_state=42)
    kmeans_plus.fit(X_blob, init_method='kmeans++')
    print(f"  K-Means++: Inertia={kmeans_plus.inertia_:.2f}, "
          f"Iterations={kmeans_plus.n_iter_}")
    
    # 2. Choosing optimal K
    print("\n2. Choosing Optimal Number of Clusters")
    print("-" * 50)
    
    # Create dataset with unclear number of clusters
    X_unclear, _ = make_blobs(n_samples=300, centers=5, random_state=42, 
                             cluster_std=3.0)
    
    # Test different k values
    k_values = range(2, 8)
    metrics = {'inertia': [], 'silhouette': [], 'calinski_harabasz': []}
    
    for k in k_values:
        kmeans_k = KMeansClusterer(n_clusters=k, random_state=42)
        labels_k = kmeans_k.fit_predict(X_unclear)
        
        metrics['inertia'].append(kmeans_k.inertia_)
        metrics['silhouette'].append(silhouette_score(X_unclear, labels_k))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(X_unclear, labels_k))
    
    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Inertia (Elbow method)
    axes[0].plot(k_values, metrics['inertia'], 'b-o')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette score
    axes[1].plot(k_values, metrics['silhouette'], 'g-o')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].grid(True, alpha=0.3)
    
    # Calinski-Harabasz score
    axes[2].plot(k_values, metrics['calinski_harabasz'], 'r-o')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[2].set_title('Calinski-Harabasz Index')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Handling different cluster shapes
    print("\n3. Limitations with Non-spherical Clusters")
    print("-" * 50)
    
    # Create datasets with different shapes
    shape_datasets = {
        'circles': make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42),
        'moons': make_moons(n_samples=300, noise=0.1, random_state=42)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, (name, (X_shape, y_true)) in enumerate(shape_datasets.items()):
        # Original data
        ax1 = axes[idx, 0]
        ax1.scatter(X_shape[:, 0], X_shape[:, 1], c=y_true, cmap='viridis')
        ax1.set_title(f'{name.capitalize()} - True Clusters')
        ax1.grid(True, alpha=0.3)
        
        # K-Means results
        ax2 = axes[idx, 1]
        kmeans_shape = KMeansClusterer(n_clusters=2, random_state=42)
        labels_shape = kmeans_shape.fit_predict(X_shape)
        
        ax2.scatter(X_shape[:, 0], X_shape[:, 1], c=labels_shape, cmap='viridis')
        ax2.scatter(kmeans_shape.cluster_centers_[:, 0], 
                   kmeans_shape.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3)
        ax2.set_title(f'{name.capitalize()} - K-Means Results')
        ax2.grid(True, alpha=0.3)
        
        # Calculate ARI
        ari = adjusted_rand_score(y_true, labels_shape)
        ax2.text(0.02, 0.98, f'ARI: {ari:.3f}', transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 4. Scalability analysis
    print("\n4. Scalability Analysis")
    print("-" * 50)
    
    # Test performance with different dataset sizes
    sizes = [100, 500, 1000, 2000, 5000]
    times = []
    
    for size in sizes:
        X_scale, _ = make_blobs(n_samples=size, centers=5, random_state=42)
        
        import time as time_module
        start_time = time_module.time()
        
        kmeans_scale = KMeansClusterer(n_clusters=5, random_state=42)
        kmeans_scale.fit(X_scale)
        
        end_time = time_module.time()
        times.append(end_time - start_time)
        
        print(f"Dataset size {size}: {times[-1]:.4f} seconds")
    
    # Plot scalability
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Dataset Size')
    plt.ylabel('Training Time (seconds)')
    plt.title('K-Means Scalability')
    plt.grid(True, alpha=0.3)
    plt.show()

# Run comprehensive demonstration
if __name__ == "__main__":
    # Main demonstration
    main_results = demonstrate_kmeans_clustering()
    
    # Challenges analysis
    print("\n" + "="*60)
    analyze_clustering_challenges()
    
    print("\n=== K-Means Clustering Summary ===")
    print("Algorithm Steps:")
    print("1. Initialize k cluster centers")
    print("2. Assign points to nearest center")
    print("3. Update centers to cluster centroids")
    print("4. Repeat until convergence")
    
    print("\nAdvantages:")
    print("- Simple and fast")
    print("- Works well with spherical clusters")
    print("- Scales well to large datasets")
    print("- Guaranteed convergence")
    
    print("\nLimitations:")
    print("- Requires pre-specifying k")
    print("- Sensitive to initialization")
    print("- Assumes spherical clusters")
    print("- Sensitive to outliers")
    
    print("\nBest Practices:")
    print("- Use K-Means++ initialization")
    print("- Scale/normalize features")
    print("- Use multiple metrics to choose k")
    print("- Consider other algorithms for non-spherical data")
    
    print("\n=== K-Means Clustering Demonstration Complete ===")
```

This comprehensive implementation demonstrates:

### Core K-Means Concepts:
1. **Complete Algorithm Implementation** - K-Means from scratch with proper convergence
2. **Multiple Initialization Methods** - Random and K-Means++ for better results
3. **Cluster Optimization** - Elbow method, Silhouette analysis, Gap statistic
4. **Convergence Analysis** - Tracking algorithm progress and stopping criteria

### Advanced Features:
1. **Optimal Cluster Selection** - Multiple methods to find best k value
2. **Comparison with Other Algorithms** - DBSCAN, Agglomerative clustering
3. **Real-world Datasets** - Various data shapes and patterns
4. **Performance Analysis** - Scalability and timing analysis

### Practical Considerations:
1. **Initialization Sensitivity** - Demonstrating impact of random vs smart initialization
2. **Non-spherical Clusters** - Showing K-Means limitations with complex shapes
3. **Evaluation Metrics** - Silhouette score, ARI, Calinski-Harabasz index
4. **Visualization Tools** - Comprehensive plotting for understanding results

### Production-Ready Features:
1. **Robust Implementation** - Proper error handling and edge cases
2. **Comprehensive Evaluation** - Multiple metrics and validation approaches
3. **Educational Examples** - Clear demonstrations of concepts and limitations
4. **Scalability Testing** - Performance analysis with different dataset sizes

---

## Question 16

**Implement cross-validation techniques for model evaluation.**

**Answer:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Generator
from sklearn.datasets import make_classification, make_regression, load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import warnings
from abc import ABC, abstractmethod
from itertools import combinations
from dataclasses import dataclass

@dataclass
class CVResults:
    """Container for cross-validation results"""
    scores: List[float]
    mean_score: float
    std_score: float
    fold_details: List[Dict[str, Any]]
    training_times: List[float]
    validation_times: List[float]
    
    def summary(self) -> str:
        """Return summary statistics"""
        return f"Mean: {self.mean_score:.4f} (±{self.std_score:.4f})"

class BaseCrossValidator(ABC):
    """Abstract base class for cross-validation strategies"""
    
    def __init__(self, n_splits: int = 5, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/validation indices for each fold"""
        pass
    
    def get_n_splits(self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None) -> int:
        """Return number of splits"""
        return self.n_splits

class KFoldCV(BaseCrossValidator):
    """K-Fold Cross-Validation implementation"""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        super().__init__(n_splits, random_state)
        self.shuffle = shuffle
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate K-Fold splits"""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            yield train_indices, test_indices

class StratifiedKFoldCV(BaseCrossValidator):
    """Stratified K-Fold Cross-Validation implementation"""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        super().__init__(n_splits, random_state)
        self.shuffle = shuffle
    
    def split(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate stratified K-Fold splits"""
        n_samples = X.shape[0]
        unique_classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(unique_classes)
        
        # Create folds for each class
        class_folds = []
        for class_idx in range(n_classes):
            class_indices = np.where(y_indices == class_idx)[0]
            if self.shuffle:
                np.random.shuffle(class_indices)
            
            # Split class indices into folds
            fold_size = len(class_indices) // self.n_splits
            class_fold = []
            
            for i in range(self.n_splits):
                start = i * fold_size
                end = start + fold_size if i < self.n_splits - 1 else len(class_indices)
                class_fold.append(class_indices[start:end])
            
            class_folds.append(class_fold)
        
        # Combine folds from all classes
        for fold_idx in range(self.n_splits):
            test_indices = []
            for class_fold in class_folds:
                test_indices.extend(class_fold[fold_idx])
            
            test_indices = np.array(test_indices)
            train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
            
            yield train_indices, test_indices

class TimeSeriesCV(BaseCrossValidator):
    """Time Series Cross-Validation (Forward Chaining)"""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, gap: int = 0):
        super().__init__(n_splits)
        self.test_size = test_size
        self.gap = gap  # Gap between train and test to avoid data leakage
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate time series splits"""
        n_samples = X.shape[0]
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        for i in range(self.n_splits):
            # Test set
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            # Train set (all data before test set, minus gap)
            train_end = test_start - self.gap
            train_indices = np.arange(0, max(0, train_end))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

class LeaveOneOutCV(BaseCrossValidator):
    """Leave-One-Out Cross-Validation"""
    
    def __init__(self):
        pass
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate leave-one-out splits"""
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            test_indices = np.array([i])
            train_indices = np.concatenate([np.arange(0, i), np.arange(i + 1, n_samples)])
            yield train_indices, test_indices
    
    def get_n_splits(self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None) -> int:
        """Return number of splits"""
        return X.shape[0] if X is not None else 0

class GroupKFoldCV(BaseCrossValidator):
    """Group K-Fold Cross-Validation"""
    
    def __init__(self, n_splits: int = 5):
        super().__init__(n_splits)
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate group-based splits"""
        if groups is None:
            raise ValueError("Groups must be provided for GroupKFold")
        
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_splits:
            raise ValueError(f"Number of groups ({n_groups}) is less than n_splits ({self.n_splits})")
        
        # Shuffle groups
        np.random.shuffle(unique_groups)
        
        group_fold_size = n_groups // self.n_splits
        
        for i in range(self.n_splits):
            start = i * group_fold_size
            end = start + group_fold_size if i < self.n_splits - 1 else n_groups
            
            test_groups = unique_groups[start:end]
            train_groups = np.setdiff1d(unique_groups, test_groups)
            
            test_indices = np.where(np.isin(groups, test_groups))[0]
            train_indices = np.where(np.isin(groups, train_groups))[0]
            
            yield train_indices, test_indices

class RepeatedKFoldCV(BaseCrossValidator):
    """Repeated K-Fold Cross-Validation"""
    
    def __init__(self, n_splits: int = 5, n_repeats: int = 10, random_state: Optional[int] = None):
        super().__init__(n_splits, random_state)
        self.n_repeats = n_repeats
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate repeated K-Fold splits"""
        for repeat in range(self.n_repeats):
            # Use different random state for each repeat
            kfold = KFoldCV(n_splits=self.n_splits, shuffle=True, 
                           random_state=None if self.random_state is None else self.random_state + repeat)
            
            for train_idx, test_idx in kfold.split(X, y, groups):
                yield train_idx, test_idx
    
    def get_n_splits(self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None) -> int:
        """Return total number of splits"""
        return self.n_splits * self.n_repeats

class CrossValidator:
    """Main cross-validation framework"""
    
    def __init__(self, cv_strategy: BaseCrossValidator, scoring: Union[str, Callable] = 'accuracy'):
        """
        Initialize cross-validator
        
        Args:
            cv_strategy: Cross-validation strategy
            scoring: Scoring function ('accuracy', 'mse', 'r2', or custom function)
        """
        self.cv_strategy = cv_strategy
        self.scoring = scoring
        
        # Built-in scoring functions
        self.scoring_functions = {
            'accuracy': self._accuracy_score,
            'mse': self._mse_score,
            'rmse': self._rmse_score,
            'r2': self._r2_score,
            'mae': self._mae_score
        }
    
    def _accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score"""
        return np.mean(y_true == y_pred)
    
    def _mse_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean squared error (negative for maximization)"""
        return -np.mean((y_true - y_pred) ** 2)
    
    def _rmse_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate root mean squared error (negative for maximization)"""
        return -np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _mae_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean absolute error (negative for maximization)"""
        return -np.mean(np.abs(y_true - y_pred))
    
    def _get_scoring_function(self) -> Callable:
        """Get the scoring function"""
        if isinstance(self.scoring, str):
            if self.scoring in self.scoring_functions:
                return self.scoring_functions[self.scoring]
            else:
                raise ValueError(f"Unknown scoring function: {self.scoring}")
        else:
            return self.scoring
    
    def cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray, 
                      groups: np.ndarray = None, 
                      return_estimator: bool = False,
                      return_train_score: bool = False,
                      fit_params: Dict = None,
                      verbose: bool = False) -> CVResults:
        """
        Perform cross-validation
        
        Args:
            model: Machine learning model with fit/predict methods
            X: Feature matrix
            y: Target vector
            groups: Group labels for GroupKFold
            return_estimator: Whether to return fitted estimators
            return_train_score: Whether to return training scores
            fit_params: Parameters to pass to model.fit()
            verbose: Whether to print progress
            
        Returns:
            CVResults object with detailed results
        """
        if fit_params is None:
            fit_params = {}
        
        scoring_func = self._get_scoring_function()
        
        scores = []
        train_scores = []
        fold_details = []
        training_times = []
        validation_times = []
        fitted_estimators = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv_strategy.split(X, y, groups)):
            if verbose:
                print(f"Fold {fold_idx + 1}/{self.cv_strategy.get_n_splits(X, y, groups)}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone model (create fresh instance)
            from sklearn.base import clone
            try:
                fold_model = clone(model)
            except:
                # Fallback for non-sklearn models
                import copy
                fold_model = copy.deepcopy(model)
            
            # Train model
            start_time = time.time()
            fold_model.fit(X_train, y_train, **fit_params)
            train_time = time.time() - start_time
            
            # Validate model
            start_time = time.time()
            y_pred = fold_model.predict(X_test)
            val_time = time.time() - start_time
            
            # Calculate scores
            test_score = scoring_func(y_test, y_pred)
            scores.append(test_score)
            
            if return_train_score:
                y_train_pred = fold_model.predict(X_train)
                train_score = scoring_func(y_train, y_train_pred)
                train_scores.append(train_score)
            
            # Store details
            fold_details.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'test_score': test_score,
                'train_score': train_scores[-1] if return_train_score else None,
                'train_indices': train_idx,
                'test_indices': test_idx
            })
            
            training_times.append(train_time)
            validation_times.append(val_time)
            
            if return_estimator:
                fitted_estimators.append(fold_model)
        
        # Calculate statistics
        scores = np.array(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        results = CVResults(
            scores=scores.tolist(),
            mean_score=mean_score,
            std_score=std_score,
            fold_details=fold_details,
            training_times=training_times,
            validation_times=validation_times
        )
        
        # Add optional attributes
        if return_train_score:
            results.train_scores = train_scores
            results.mean_train_score = np.mean(train_scores)
            results.std_train_score = np.std(train_scores)
        
        if return_estimator:
            results.fitted_estimators = fitted_estimators
        
        return results
    
    def learning_curve(self, model: Any, X: np.ndarray, y: np.ndarray,
                      train_sizes: np.ndarray = None,
                      cv_folds: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate learning curves
        
        Args:
            model: Machine learning model
            X: Feature matrix
            y: Target vector
            train_sizes: Training set sizes to evaluate
            cv_folds: Number of CV folds for each size
            
        Returns:
            Tuple of (train_sizes, train_scores, validation_scores)
        """
        n_samples = X.shape[0]
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Convert fractions to absolute sizes
        if np.max(train_sizes) <= 1.0:
            train_sizes = (train_sizes * n_samples).astype(int)
        
        train_scores_mean = []
        train_scores_std = []
        val_scores_mean = []
        val_scores_std = []
        
        for train_size in train_sizes:
            # Limit training size
            train_size = min(train_size, n_samples - cv_folds)
            
            # Subsample data
            indices = np.random.choice(n_samples, train_size + n_samples//cv_folds, replace=False)
            X_subset = X[indices]
            y_subset = y[indices]
            
            # Perform cross-validation on subset
            cv = KFoldCV(n_splits=cv_folds, random_state=42)
            cv_validator = CrossValidator(cv, self.scoring)
            results = cv_validator.cross_validate(model, X_subset, y_subset, return_train_score=True)
            
            val_scores_mean.append(results.mean_score)
            val_scores_std.append(results.std_score)
            train_scores_mean.append(results.mean_train_score)
            train_scores_std.append(results.std_train_score)
        
        return (train_sizes, 
                (np.array(train_scores_mean), np.array(train_scores_std)),
                (np.array(val_scores_mean), np.array(val_scores_std)))

class NestedCrossValidation:
    """Nested Cross-Validation for hyperparameter tuning and model selection"""
    
    def __init__(self, 
                 outer_cv: BaseCrossValidator, 
                 inner_cv: BaseCrossValidator,
                 scoring: Union[str, Callable] = 'accuracy'):
        """
        Initialize nested cross-validation
        
        Args:
            outer_cv: Outer CV strategy for performance estimation
            inner_cv: Inner CV strategy for hyperparameter tuning
            scoring: Scoring function
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.scoring = scoring
    
    def nested_cross_validate(self, 
                            models_params: Dict[str, Dict[str, List]],
                            X: np.ndarray, 
                            y: np.ndarray,
                            verbose: bool = False) -> Dict[str, Any]:
        """
        Perform nested cross-validation with model selection
        
        Args:
            models_params: Dictionary of {model_name: {param_name: [values]}}
            X: Feature matrix
            y: Target vector
            verbose: Whether to print progress
            
        Returns:
            Dictionary with nested CV results
        """
        outer_scores = []
        best_models = []
        best_params_list = []
        
        for outer_fold, (train_idx, test_idx) in enumerate(self.outer_cv.split(X, y)):
            if verbose:
                print(f"Outer fold {outer_fold + 1}/{self.outer_cv.get_n_splits(X, y)}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            best_score = -np.inf
            best_model = None
            best_params = None
            
            for model_name, param_grid in models_params.items():
                if verbose:
                    print(f"  Testing {model_name}")
                
                # Grid search with inner CV
                param_combinations = self._generate_param_combinations(param_grid)
                
                for params in param_combinations:
                    # Create model with current parameters
                    model = self._create_model(model_name, params)
                    
                    # Inner CV evaluation
                    inner_cv_validator = CrossValidator(self.inner_cv, self.scoring)
                    inner_results = inner_cv_validator.cross_validate(
                        model, X_train_outer, y_train_outer, verbose=False
                    )
                    
                    if inner_results.mean_score > best_score:
                        best_score = inner_results.mean_score
                        best_model = model_name
                        best_params = params
            
            # Train best model on full outer training set
            final_model = self._create_model(best_model, best_params)
            final_model.fit(X_train_outer, y_train_outer)
            
            # Evaluate on outer test set
            y_pred = final_model.predict(X_test_outer)
            scorer = CrossValidator(self.outer_cv, self.scoring)
            scoring_func = scorer._get_scoring_function()
            outer_score = scoring_func(y_test_outer, y_pred)
            
            outer_scores.append(outer_score)
            best_models.append(best_model)
            best_params_list.append(best_params)
            
            if verbose:
                print(f"  Best model: {best_model}")
                print(f"  Best params: {best_params}")
                print(f"  Outer score: {outer_score:.4f}")
        
        return {
            'outer_scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_models': best_models,
            'best_params': best_params_list
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations"""
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = []
        from itertools import product
        
        for combo in product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def _create_model(self, model_name: str, params: Dict) -> Any:
        """Create model instance with given parameters"""
        if model_name == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params, random_state=42, max_iter=1000)
        elif model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params, random_state=42)
        elif model_name == 'svm':
            from sklearn.svm import SVC
            return SVC(**params, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")

class BiasVarianceAnalyzer:
    """Analyze bias-variance tradeoff using cross-validation"""
    
    def __init__(self, cv_strategy: BaseCrossValidator, n_bootstrap: int = 100):
        self.cv_strategy = cv_strategy
        self.n_bootstrap = n_bootstrap
    
    def analyze_bias_variance(self, 
                            model: Any, 
                            X: np.ndarray, 
                            y: np.ndarray,
                            X_test: np.ndarray,
                            y_test: np.ndarray) -> Dict[str, float]:
        """
        Analyze bias-variance decomposition
        
        Args:
            model: Machine learning model
            X: Training features
            y: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with bias, variance, and noise estimates
        """
        predictions = []
        
        # Bootstrap sampling and model training
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            n_samples = X.shape[0]
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Train model
            from sklearn.base import clone
            bootstrap_model = clone(model)
            bootstrap_model.fit(X_bootstrap, y_bootstrap)
            
            # Predict on test set
            y_pred = bootstrap_model.predict(X_test)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        # Calculate bias and variance
        mean_prediction = np.mean(predictions, axis=0)
        bias_squared = np.mean((mean_prediction - y_test) ** 2)
        variance = np.mean(np.var(predictions, axis=0))
        noise = 0  # Assuming no noise in test targets
        
        total_error = bias_squared + variance + noise
        
        return {
            'bias_squared': bias_squared,
            'variance': variance,
            'noise': noise,
            'total_error': total_error,
            'bias_variance_ratio': bias_squared / (variance + 1e-8)
        }

# Visualization utilities
class CVVisualizer:
    """Visualization utilities for cross-validation results"""
    
    @staticmethod
    def plot_cv_scores(results: CVResults, title: str = "Cross-Validation Scores"):
        """Plot cross-validation scores"""
        plt.figure(figsize=(10, 6))
        
        folds = range(1, len(results.scores) + 1)
        
        plt.subplot(1, 2, 1)
        plt.bar(folds, results.scores, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.axhline(y=results.mean_score, color='red', linestyle='--', 
                   label=f'Mean: {results.mean_score:.3f}')
        plt.fill_between(folds, 
                        results.mean_score - results.std_score,
                        results.mean_score + results.std_score,
                        alpha=0.2, color='red')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title('Scores by Fold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(results.scores, bins=max(3, len(results.scores)//2), 
                alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        plt.axvline(x=results.mean_score, color='red', linestyle='--',
                   label=f'Mean: {results.mean_score:.3f}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_learning_curves(train_sizes: np.ndarray, 
                           train_scores: Tuple[np.ndarray, np.ndarray],
                           val_scores: Tuple[np.ndarray, np.ndarray],
                           title: str = "Learning Curves"):
        """Plot learning curves"""
        plt.figure(figsize=(10, 6))
        
        train_mean, train_std = train_scores
        val_mean, val_std = val_scores
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def compare_cv_strategies(cv_results: Dict[str, CVResults]):
        """Compare different CV strategies"""
        strategies = list(cv_results.keys())
        means = [results.mean_score for results in cv_results.values()]
        stds = [results.std_score for results in cv_results.values()]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        x_pos = np.arange(len(strategies))
        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                      color=['skyblue', 'lightgreen', 'salmon', 'gold'][:len(strategies)])
        plt.xlabel('CV Strategy')
        plt.ylabel('Mean Score')
        plt.title('CV Strategy Comparison')
        plt.xticks(x_pos, strategies, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std/2,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.subplot(1, 2, 2)
        for i, (strategy, results) in enumerate(cv_results.items()):
            plt.scatter([i] * len(results.scores), results.scores, 
                       alpha=0.6, s=50, label=strategy)
        
        plt.xlabel('CV Strategy')
        plt.ylabel('Individual Scores')
        plt.title('Score Distributions')
        plt.xticks(range(len(strategies)), strategies, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstration functions
def demonstrate_basic_cv():
    """Demonstrate basic cross-validation techniques"""
    print("=== Basic Cross-Validation Demonstration ===\n")
    
    # Load dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    # Create models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    # CV strategies
    cv_strategies = {
        'K-Fold': KFoldCV(n_splits=5, random_state=42),
        'Stratified K-Fold': StratifiedKFoldCV(n_splits=5, random_state=42),
        'Repeated K-Fold': RepeatedKFoldCV(n_splits=5, n_repeats=3, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        model_results = {}
        
        for cv_name, cv_strategy in cv_strategies.items():
            validator = CrossValidator(cv_strategy, scoring='accuracy')
            cv_results = validator.cross_validate(model, X, y, return_train_score=True, verbose=False)
            
            model_results[cv_name] = cv_results
            
            print(f"  {cv_name}: {cv_results.summary()}")
            if hasattr(cv_results, 'mean_train_score'):
                print(f"    Train: {cv_results.mean_train_score:.4f} (±{cv_results.std_train_score:.4f})")
        
        results[model_name] = model_results
    
    return results

def demonstrate_time_series_cv():
    """Demonstrate time series cross-validation"""
    print("\n=== Time Series Cross-Validation Demonstration ===\n")
    
    # Generate time series data
    np.random.seed(42)
    n_samples = 200
    time = np.linspace(0, 4*np.pi, n_samples)
    trend = 0.1 * time
    seasonal = 2 * np.sin(time) + np.sin(2*time)
    noise = np.random.normal(0, 0.5, n_samples)
    y = trend + seasonal + noise
    
    # Create features (lagged values)
    X = np.column_stack([
        np.roll(y, 1),  # lag 1
        np.roll(y, 2),  # lag 2
        np.roll(y, 3),  # lag 3
        np.roll(y, 7),  # lag 7
    ])[7:]  # Remove first 7 rows due to lags
    y = y[7:]
    
    print(f"Time series data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Compare CV strategies
    cv_strategies = {
        'Standard K-Fold (Wrong!)': KFoldCV(n_splits=5, random_state=42),
        'Time Series CV': TimeSeriesCV(n_splits=5),
        'Time Series CV with Gap': TimeSeriesCV(n_splits=5, gap=5)
    }
    
    model = LinearRegression()
    results = {}
    
    for cv_name, cv_strategy in cv_strategies.items():
        validator = CrossValidator(cv_strategy, scoring='r2')
        cv_results = validator.cross_validate(model, X, y, verbose=False)
        results[cv_name] = cv_results
        
        print(f"{cv_name}: {cv_results.summary()}")
        if 'Wrong' in cv_name:
            print("  ^ This is wrong for time series data!")
    
    return results

def demonstrate_nested_cv():
    """Demonstrate nested cross-validation"""
    print("\n=== Nested Cross-Validation Demonstration ===\n")
    
    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Use subset for faster demo
    n_samples = 500
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y = y[indices]
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Define models and parameter grids
    models_params = {
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        },
        'random_forest': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7]
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear']
        }
    }
    
    # Nested CV
    outer_cv = KFoldCV(n_splits=3, random_state=42)  # Smaller for demo
    inner_cv = KFoldCV(n_splits=3, random_state=42)
    
    nested_cv = NestedCrossValidation(outer_cv, inner_cv, scoring='accuracy')
    
    print("Running nested cross-validation...")
    nested_results = nested_cv.nested_cross_validate(models_params, X, y, verbose=True)
    
    print(f"\nNested CV Results:")
    print(f"Mean Score: {nested_results['mean_score']:.4f} (±{nested_results['std_score']:.4f})")
    print(f"Best Models: {nested_results['best_models']}")
    
    return nested_results

def demonstrate_bias_variance():
    """Demonstrate bias-variance analysis"""
    print("\n=== Bias-Variance Analysis Demonstration ===\n")
    
    # Generate dataset with known noise
    np.random.seed(42)
    n_train = 500
    n_test = 200
    
    X_train, y_train = make_regression(n_samples=n_train, n_features=20, 
                                      noise=10, random_state=42)
    X_test, y_test = make_regression(n_samples=n_test, n_features=20, 
                                    noise=10, random_state=43)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Models with different complexity
    models = {
        'Linear (Low Variance, High Bias)': LinearRegression(),
        'Random Forest (High Variance, Low Bias)': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    cv = KFoldCV(n_splits=5, random_state=42)
    analyzer = BiasVarianceAnalyzer(cv, n_bootstrap=50)
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 50)
        
        analysis = analyzer.analyze_bias_variance(model, X_train, y_train, X_test, y_test)
        
        print(f"Bias²: {analysis['bias_squared']:.4f}")
        print(f"Variance: {analysis['variance']:.4f}")
        print(f"Total Error: {analysis['total_error']:.4f}")
        print(f"Bias/Variance Ratio: {analysis['bias_variance_ratio']:.4f}")
    
    return models

def demonstrate_learning_curves():
    """Demonstrate learning curves"""
    print("\n=== Learning Curves Demonstration ===\n")
    
    # Load data
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    cv = KFoldCV(n_splits=5, random_state=42)
    validator = CrossValidator(cv, scoring='accuracy')
    
    for model_name, model in models.items():
        print(f"\nGenerating learning curves for {model_name}...")
        
        train_sizes, train_scores, val_scores = validator.learning_curve(
            model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv_folds=3
        )
        
        print(f"Final training score: {train_scores[0][-1]:.4f} (±{train_scores[1][-1]:.4f})")
        print(f"Final validation score: {val_scores[0][-1]:.4f} (±{val_scores[1][-1]:.4f})")
        
        # Plot learning curves
        CVVisualizer.plot_learning_curves(train_sizes, train_scores, val_scores, 
                                        f"Learning Curves - {model_name}")

def demonstrate_group_cv():
    """Demonstrate group-based cross-validation"""
    print("\n=== Group Cross-Validation Demonstration ===\n")
    
    # Generate data with groups (e.g., different patients, experiments, etc.)
    np.random.seed(42)
    n_groups = 20
    samples_per_group = np.random.randint(10, 50, n_groups)
    
    X_list = []
    y_list = []
    groups_list = []
    
    for group_id in range(n_groups):
        n_samples = samples_per_group[group_id]
        
        # Each group has slightly different characteristics
        group_X, group_y = make_classification(
            n_samples=n_samples, n_features=10, n_informative=8,
            n_classes=2, random_state=group_id
        )
        
        # Add group-specific bias
        group_X += np.random.normal(0, 0.1 * group_id, group_X.shape)
        
        X_list.append(group_X)
        y_list.append(group_y)
        groups_list.extend([group_id] * n_samples)
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    groups = np.array(groups_list)
    
    print(f"Dataset: {X.shape[0]} samples, {n_groups} groups")
    print(f"Samples per group: min={min(samples_per_group)}, max={max(samples_per_group)}")
    
    # Compare CV strategies
    cv_strategies = {
        'Standard K-Fold (Wrong!)': KFoldCV(n_splits=5, random_state=42),
        'Group K-Fold': GroupKFoldCV(n_splits=5)
    }
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    for cv_name, cv_strategy in cv_strategies.items():
        print(f"\n{cv_name}:")
        validator = CrossValidator(cv_strategy, scoring='accuracy')
        
        if 'Group' in cv_name:
            cv_results = validator.cross_validate(model, X, y, groups=groups, verbose=False)
        else:
            cv_results = validator.cross_validate(model, X, y, verbose=False)
        
        print(f"  Score: {cv_results.summary()}")
        if 'Wrong' in cv_name:
            print("  ^ This may overestimate performance due to data leakage!")

# Main demonstration function
def run_all_demonstrations():
    """Run all cross-validation demonstrations"""
    print("=" * 80)
    print("COMPREHENSIVE CROSS-VALIDATION DEMONSTRATION")
    print("=" * 80)
    
    # 1. Basic CV
    basic_results = demonstrate_basic_cv()
    
    # 2. Time Series CV
    ts_results = demonstrate_time_series_cv()
    
    # 3. Nested CV
    nested_results = demonstrate_nested_cv()
    
    # 4. Bias-Variance Analysis
    bv_models = demonstrate_bias_variance()
    
    # 5. Learning Curves
    demonstrate_learning_curves()
    
    # 6. Group CV
    demonstrate_group_cv()
    
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION BEST PRACTICES")
    print("=" * 80)
    
    print("""
1. CHOOSE THE RIGHT CV STRATEGY:
   - K-Fold: General purpose, balanced datasets
   - Stratified K-Fold: Imbalanced classification
   - Time Series CV: Temporal data (avoid data leakage)
   - Group K-Fold: Grouped data (patients, experiments)
   - Leave-One-Out: Small datasets (computationally expensive)

2. NESTED CV FOR MODEL SELECTION:
   - Use nested CV when tuning hyperparameters
   - Outer loop: Performance estimation
   - Inner loop: Hyperparameter tuning
   - Avoids optimistic bias from parameter tuning

3. CONSIDERATIONS:
   - Computational cost vs. accuracy trade-off
   - Dataset size and CV strategy choice
   - Stratification for imbalanced data
   - Data leakage prevention (especially time series)
   - Consistent preprocessing within CV folds

4. METRICS AND INTERPRETATION:
   - Report mean ± std dev
   - Consider confidence intervals
   - Visualize score distributions
   - Analyze bias-variance trade-off
   - Use learning curves for model complexity analysis
    """)
    
    return {
        'basic_results': basic_results,
        'ts_results': ts_results,
        'nested_results': nested_results,
        'bv_models': bv_models
    }

# Example usage and testing
if __name__ == "__main__":
    # Run comprehensive demonstration
    all_results = run_all_demonstrations()
    
    print("\n=== Cross-Validation Implementation Complete ===")
    
    # Quick example
    print("\n=== Quick Example ===")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Simple k-fold CV
    cv = KFoldCV(n_splits=5, random_state=42)
    validator = CrossValidator(cv, scoring='accuracy')
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    results = validator.cross_validate(model, X, y, verbose=False)
    print(f"Iris classification with Logistic Regression: {results.summary()}")
    
    # Visualize results
    CVVisualizer.plot_cv_scores(results, "Iris Classification CV Results")
```

This comprehensive cross-validation implementation provides:

### Core CV Strategies:
1. **K-Fold Cross-Validation** - Standard approach for most datasets
2. **Stratified K-Fold** - Maintains class proportions in each fold
3. **Time Series CV** - Respects temporal order, prevents data leakage
4. **Leave-One-Out CV** - Uses each sample as validation once
5. **Group K-Fold** - Ensures related samples stay together
6. **Repeated K-Fold** - Multiple repetitions for more robust estimates

### Advanced Features:
1. **Nested Cross-Validation** - For unbiased hyperparameter tuning
2. **Bias-Variance Analysis** - Understanding model complexity trade-offs
3. **Learning Curves** - Analyzing performance vs. training set size
4. **Multiple Scoring Metrics** - Accuracy, MSE, R², MAE, custom functions

### Practical Tools:
1. **Comprehensive Evaluation Framework** - Detailed results with statistics
2. **Visualization Tools** - Score distributions, learning curves, comparisons
3. **Real-world Examples** - Time series, grouped data, model selection
4. **Best Practices Guide** - When to use which CV strategy

The implementation handles edge cases, provides detailed diagnostics, and includes extensive demonstrations showing proper usage patterns.
    source_accuracy: float
    target_accuracy_scratch: float
    target_accuracy_transfer: float
    training_time_scratch: float
    training_time_transfer: float
    data_efficiency: Dict[str, float]

class PretrainedModel(ABC):
    """Abstract base class for pre-trained models"""
    
    @abstractmethod
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from input data"""
        pass
    
    @abstractmethod
    def fine_tune(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fine-tune the model on new data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

class SimpleNN:
    """Simple neural network for demonstration"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 output_size: int, learning_rate: float = 0.01):
        """Initialize neural network"""
        self.learning_rate = learning_rate
        self.layers = []
        
        # Build layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append({
                'weights': np.random.normal(0, 0.1, (prev_size, hidden_size)),
                'bias': np.zeros((1, hidden_size)),
                'type': 'hidden'
            })
            prev_size = hidden_size
        
        # Output layer
        self.layers.append({
            'weights': np.random.normal(0, 0.1, (prev_size, output_size)),
            'bias': np.zeros((1, output_size)),
            'type': 'output'
        })
        
        # Store activations for backpropagation
        self.activations = []
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.activations = [X]
        
        for i, layer in enumerate(self.layers):
            z = self.activations[-1] @ layer['weights'] + layer['bias']
            
            if layer['type'] == 'hidden':
                activation = self._relu(z)
            else:  # output layer
                activation = self._softmax(z)
            
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """Backward pass"""
        m = X.shape[0]
        
        # Convert y to one-hot if needed
        if y.ndim == 1:
            y_onehot = np.eye(output.shape[1])[y]
        else:
            y_onehot = y
        
        # Output layer gradient
        d_output = output - y_onehot
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            # Compute gradients
            d_weights = self.activations[i].T @ d_output / m
            d_bias = np.sum(d_output, axis=0, keepdims=True) / m
            
            # Update weights
            layer['weights'] -= self.learning_rate * d_weights
            layer['bias'] -= self.learning_rate * d_bias
            
            # Compute gradient for previous layer
            if i > 0:
                d_output = (d_output @ layer['weights'].T) * self._relu_derivative(self.activations[i])
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            batch_size: int = 32, verbose: bool = False) -> List[float]:
        """Train the neural network"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Compute loss (cross-entropy)
                if y_batch.ndim == 1:
                    y_onehot = np.eye(output.shape[1])[y_batch]
                else:
                    y_onehot = y_batch
                
                loss = -np.mean(np.sum(y_onehot * np.log(output + 1e-15), axis=1))
                epoch_losses.append(loss)
                
                # Backward pass
                self.backward(X_batch, y_batch, output)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        return self.forward(X)

class TransferLearningNN(PretrainedModel):
    """Neural network with transfer learning capabilities"""
    
    def __init__(self, base_model: SimpleNN, freeze_layers: int = 0):
        """
        Initialize transfer learning model
        
        Args:
            base_model: Pre-trained base model
            freeze_layers: Number of layers to freeze (from beginning)
        """
        self.base_model = base_model
        self.freeze_layers = freeze_layers
        self.feature_extractor = None
        self.classifier = None
        
        # Create feature extractor (frozen layers)
        if freeze_layers > 0:
            self.frozen_layers = self.base_model.layers[:freeze_layers]
        else:
            self.frozen_layers = []
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features using frozen layers"""
        if not self.frozen_layers:
            return X
        
        activations = X
        for layer in self.frozen_layers:
            z = activations @ layer['weights'] + layer['bias']
            activations = np.maximum(0, z)  # ReLU
        
        return activations
    
    def fine_tune(self, X: np.ndarray, y: np.ndarray, 
                 new_output_size: Optional[int] = None,
                 epochs: int = 50, learning_rate: float = 0.01,
                 freeze_base: bool = True) -> List[float]:
        """
        Fine-tune the model on new data
        
        Args:
            X: New training data
            y: New training labels
            new_output_size: Size of new output layer
            epochs: Training epochs
            learning_rate: Learning rate for fine-tuning
            freeze_base: Whether to freeze base layers
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Determine output size
        if new_output_size is None:
            new_output_size = len(np.unique(y))
        
        if freeze_base:
            # Feature extraction approach
            features = self.extract_features(X)
            
            # Create new classifier
            self.classifier = SimpleNN(
                input_size=features.shape[1],
                hidden_sizes=[64],
                output_size=new_output_size,
                learning_rate=learning_rate
            )
            
            # Train classifier
            losses = self.classifier.fit(features, y, epochs=epochs, verbose=False)
        else:
            # Fine-tuning approach
            # Replace output layer
            if len(self.base_model.layers) > 0:
                last_hidden_size = self.base_model.layers[-2]['weights'].shape[1] if len(self.base_model.layers) > 1 else X.shape[1]
                self.base_model.layers[-1] = {
                    'weights': np.random.normal(0, 0.1, (last_hidden_size, new_output_size)),
                    'bias': np.zeros((1, new_output_size)),
                    'type': 'output'
                }
            
            # Reduce learning rate for fine-tuning
            original_lr = self.base_model.learning_rate
            self.base_model.learning_rate = learning_rate * 0.1
            
            # Train the entire model
            losses = self.base_model.fit(X, y, epochs=epochs, verbose=False)
            
            # Restore original learning rate
            self.base_model.learning_rate = original_lr
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.classifier is not None:
            # Feature extraction + classification
            features = self.extract_features(X)
            return self.classifier.predict(features)
        else:
            # End-to-end prediction
            return self.base_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if self.classifier is not None:
            features = self.extract_features(X)
            return self.classifier.predict_proba(features)
        else:
            return self.base_model.predict_proba(X)

class DomainAdaptationModel:
    """Model for domain adaptation scenarios"""
    
    def __init__(self, base_model: SimpleNN):
        """Initialize domain adaptation model"""
        self.base_model = base_model
        self.domain_classifier = None
        self.feature_size = None
    
    def add_domain_adversarial_training(self, 
                                      X_source: np.ndarray, 
                                      X_target: np.ndarray,
                                      y_source: np.ndarray,
                                      epochs: int = 50):
        """
        Implement simplified domain adversarial training
        
        This is a simplified version that demonstrates the concept
        """
        print("Training domain adaptation model...")
        
        # Extract features from both domains
        source_features = self.base_model.extract_features(X_source) if hasattr(self.base_model, 'extract_features') else X_source
        target_features = self.base_model.extract_features(X_target) if hasattr(self.base_model, 'extract_features') else X_target
        
        # Create domain labels (0 for source, 1 for target)
        source_domain_labels = np.zeros(len(source_features))
        target_domain_labels = np.ones(len(target_features))
        
        # Combine features and domain labels
        all_features = np.vstack([source_features, target_features])
        domain_labels = np.concatenate([source_domain_labels, target_domain_labels])
        
        # Train domain classifier (to measure domain discrepancy)
        self.domain_classifier = SimpleNN(
            input_size=all_features.shape[1],
            hidden_sizes=[32],
            output_size=2,
            learning_rate=0.01
        )
        
        self.domain_classifier.fit(all_features, domain_labels.astype(int), 
                                 epochs=epochs//2, verbose=False)
        
        # Measure domain adaptation success
        domain_predictions = self.domain_classifier.predict(all_features)
        domain_accuracy = accuracy_score(domain_labels, domain_predictions)
        
        print(f"Domain classifier accuracy: {domain_accuracy:.4f}")
        print("Lower domain accuracy indicates better domain adaptation")
        
        return domain_accuracy

class TransferLearningFramework:
    """
    Comprehensive transfer learning framework
    """
    
    def __init__(self):
        """Initialize framework"""
        self.models = {}
        self.results = {}
    
    def create_source_task_data(self, n_samples: int = 1000, n_features: int = 20, 
                              n_classes: int = 5, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic source task data"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Add noise
        X += np.random.normal(0, noise, X.shape)
        
        return X, y
    
    def create_target_task_data(self, source_X: np.ndarray, source_y: np.ndarray,
                              domain_shift: float = 0.5, 
                              label_shift: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Create target task data with domain shift"""
        
        # Apply domain shift (feature distribution change)
        shift_matrix = np.random.normal(1.0, domain_shift, source_X.shape[1])
        target_X = source_X * shift_matrix + np.random.normal(0, domain_shift/2, source_X.shape)
        
        # Apply label shift if requested
        if label_shift:
            # Change class distribution
            unique_classes = np.unique(source_y)
            new_class_probs = np.random.dirichlet(np.ones(len(unique_classes)) * 0.5)
            
            # Resample according to new distribution
            n_samples = len(source_y)
            new_class_counts = (new_class_probs * n_samples).astype(int)
            
            target_indices = []
            for class_idx, count in enumerate(new_class_counts):
                class_indices = np.where(source_y == unique_classes[class_idx])[0]
                if len(class_indices) >= count:
                    selected = np.random.choice(class_indices, count, replace=False)
                else:
                    selected = np.random.choice(class_indices, count, replace=True)
                target_indices.extend(selected)
            
            target_X = target_X[target_indices]
            target_y = source_y[target_indices]
        else:
            target_y = source_y.copy()
        
        return target_X, target_y
    
    def train_source_model(self, X_source: np.ndarray, y_source: np.ndarray,
                          model_name: str = 'source_model') -> SimpleNN:
        """Train model on source task"""
        
        print(f"Training source model: {model_name}")
        
        # Create and train source model
        source_model = SimpleNN(
            input_size=X_source.shape[1],
            hidden_sizes=[128, 64, 32],
            output_size=len(np.unique(y_source)),
            learning_rate=0.01
        )
        
        # Train source model
        start_time = time.time()
        losses = source_model.fit(X_source, y_source, epochs=100, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate source model
        source_pred = source_model.predict(X_source)
        source_accuracy = accuracy_score(y_source, source_pred)
        
        print(f"Source model accuracy: {source_accuracy:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        self.models[model_name] = source_model
        
        return source_model
    
    def evaluate_transfer_learning(self, 
                                 source_model: SimpleNN,
                                 X_target: np.ndarray, 
                                 y_target: np.ndarray,
                                 data_fractions: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
                                 method: str = 'fine_tuning') -> TransferResults:
        """
        Evaluate transfer learning effectiveness
        
        Args:
            source_model: Pre-trained source model
            X_target: Target task features
            y_target: Target task labels
            data_fractions: Fractions of target data to use
            method: 'fine_tuning' or 'feature_extraction'
        """
        
        print(f"\nEvaluating transfer learning with {method}")
        
        # Split target data
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target, test_size=0.3, random_state=42, stratify=y_target
        )
        
        # Train model from scratch for comparison
        print("Training baseline model from scratch...")
        start_time = time.time()
        scratch_model = SimpleNN(
            input_size=X_target.shape[1],
            hidden_sizes=[128, 64, 32],
            output_size=len(np.unique(y_target)),
            learning_rate=0.01
        )
        scratch_model.fit(X_train, y_train, epochs=100, verbose=False)
        scratch_time = time.time() - start_time
        
        scratch_pred = scratch_model.predict(X_test)
        scratch_accuracy = accuracy_score(y_test, scratch_pred)
        
        # Transfer learning
        print(f"Applying transfer learning ({method})...")
        start_time = time.time()
        
        if method == 'fine_tuning':
            transfer_model = TransferLearningNN(source_model, freeze_layers=0)
            transfer_model.fine_tune(
                X_train, y_train, 
                new_output_size=len(np.unique(y_target)),
                epochs=50, 
                freeze_base=False
            )
        else:  # feature_extraction
            transfer_model = TransferLearningNN(source_model, freeze_layers=2)
            transfer_model.fine_tune(
                X_train, y_train,
                new_output_size=len(np.unique(y_target)),
                epochs=50,
                freeze_base=True
            )
        
        transfer_time = time.time() - start_time
        
        transfer_pred = transfer_model.predict(X_test)
        transfer_accuracy = accuracy_score(y_test, transfer_pred)
        
        # Evaluate data efficiency
        data_efficiency = {}
        
        for fraction in data_fractions:
            n_samples = int(len(X_train) * fraction)
            X_subset = X_train[:n_samples]
            y_subset = y_train[:n_samples]
            
            # Transfer learning with subset
            if method == 'fine_tuning':
                subset_transfer = TransferLearningNN(source_model, freeze_layers=0)
                subset_transfer.fine_tune(
                    X_subset, y_subset,
                    new_output_size=len(np.unique(y_target)),
                    epochs=30,
                    freeze_base=False
                )
            else:
                subset_transfer = TransferLearningNN(source_model, freeze_layers=2)
                subset_transfer.fine_tune(
                    X_subset, y_subset,
                    new_output_size=len(np.unique(y_target)),
                    epochs=30,
                    freeze_base=True
                )
            
            subset_pred = subset_transfer.predict(X_test)
            subset_accuracy = accuracy_score(y_test, subset_pred)
            data_efficiency[f'{fraction*100:.0f}%'] = subset_accuracy
        
        # Calculate source model accuracy on target (for comparison)
        try:
            source_pred = source_model.predict(X_test)
            source_accuracy = accuracy_score(y_test, source_pred)
        except:
            source_accuracy = 0.0  # If output dimensions don't match
        
        results = TransferResults(
            source_accuracy=source_accuracy,
            target_accuracy_scratch=scratch_accuracy,
            target_accuracy_transfer=transfer_accuracy,
            training_time_scratch=scratch_time,
            training_time_transfer=transfer_time,
            data_efficiency=data_efficiency
        )
        
        # Print results
        print(f"\nResults:")
        print(f"Source model on target: {source_accuracy:.4f}")
        print(f"Scratch model accuracy: {scratch_accuracy:.4f}")
        print(f"Transfer model accuracy: {transfer_accuracy:.4f}")
        print(f"Improvement: {transfer_accuracy - scratch_accuracy:.4f}")
        print(f"Training time reduction: {(scratch_time - transfer_time)/scratch_time*100:.1f}%")
        
        return results
    
    def compare_transfer_methods(self, 
                               source_model: SimpleNN,
                               X_target: np.ndarray,
                               y_target: np.ndarray) -> Dict[str, TransferResults]:
        """Compare different transfer learning methods"""
        
        methods = ['feature_extraction', 'fine_tuning']
        results = {}
        
        for method in methods:
            results[method] = self.evaluate_transfer_learning(
                source_model, X_target, y_target, method=method
            )
        
        return results
    
    def visualize_transfer_results(self, results: Dict[str, TransferResults]):
        """Visualize transfer learning results"""
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Accuracy comparison
        methods = list(results.keys())
        scratch_accuracies = [results[method].target_accuracy_scratch for method in methods]
        transfer_accuracies = [results[method].target_accuracy_transfer for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, scratch_accuracies, width, label='From Scratch', alpha=0.7)
        axes[0, 0].bar(x + width/2, transfer_accuracies, width, label='Transfer Learning', alpha=0.7)
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training time comparison
        scratch_times = [results[method].training_time_scratch for method in methods]
        transfer_times = [results[method].training_time_transfer for method in methods]
        
        axes[0, 1].bar(x - width/2, scratch_times, width, label='From Scratch', alpha=0.7)
        axes[0, 1].bar(x + width/2, transfer_times, width, label='Transfer Learning', alpha=0.7)
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].set_title('Training Time Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(methods)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Data efficiency for feature extraction
        if 'feature_extraction' in results:
            data_fractions = list(results['feature_extraction'].data_efficiency.keys())
            fe_accuracies = list(results['feature_extraction'].data_efficiency.values())
            
            axes[1, 0].plot(data_fractions, fe_accuracies, 'o-', linewidth=2, markersize=8, label='Feature Extraction')
            
            if 'fine_tuning' in results:
                ft_accuracies = list(results['fine_tuning'].data_efficiency.values())
                axes[1, 0].plot(data_fractions, ft_accuracies, 's-', linewidth=2, markersize=8, label='Fine Tuning')
            
            axes[1, 0].set_xlabel('Training Data Percentage')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Data Efficiency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Improvement over baseline
        improvements = [results[method].target_accuracy_transfer - results[method].target_accuracy_scratch 
                       for method in methods]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = axes[1, 1].bar(methods, improvements, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('Accuracy Improvement')
        axes[1, 1].set_title('Transfer Learning Improvement')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.005),
                           f'{imp:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.show()

# Demonstration with real-world scenario
def demonstrate_transfer_learning():
    """Comprehensive transfer learning demonstration"""
    
    print("=== Transfer Learning Demonstration ===\n")
    
    # Initialize framework
    framework = TransferLearningFramework()
    
    # 1. Create source task (general image classification)
    print("1. Creating Source Task Data (General Classification)")
    print("-" * 60)
    
    X_source, y_source = framework.create_source_task_data(
        n_samples=2000, n_features=50, n_classes=10, noise=0.1
    )
    
    print(f"Source data shape: {X_source.shape}")
    print(f"Source classes: {len(np.unique(y_source))}")
    
    # Scale source data
    scaler_source = StandardScaler()
    X_source_scaled = scaler_source.fit_transform(X_source)
    
    # 2. Train source model
    print("\n2. Training Source Model")
    print("-" * 60)
    
    source_model = framework.train_source_model(X_source_scaled, y_source)
    
    # 3. Create target task with domain shift
    print("\n3. Creating Target Task Data (Domain Adaptation)")
    print("-" * 60)
    
    X_target, y_target = framework.create_target_task_data(
        X_source_scaled, y_source, 
        domain_shift=0.3, 
        label_shift=True
    )
    
    # Reduce target data size (common in transfer learning scenarios)
    n_target = 800
    indices = np.random.choice(len(X_target), n_target, replace=False)
    X_target = X_target[indices]
    y_target = y_target[indices]
    
    print(f"Target data shape: {X_target.shape}")
    print(f"Target classes: {len(np.unique(y_target))}")
    
    # 4. Compare transfer learning methods
    print("\n4. Comparing Transfer Learning Methods")
    print("-" * 60)
    
    transfer_results = framework.compare_transfer_methods(
        source_model, X_target, y_target
    )
    
    # 5. Domain adaptation demonstration
    print("\n5. Domain Adaptation Demonstration")
    print("-" * 60)
    
    # Create more significant domain shift
    X_target_shifted, _ = framework.create_target_task_data(
        X_source_scaled, y_source, domain_shift=0.8
    )
    
    domain_adapter = DomainAdaptationModel(source_model)
    domain_accuracy = domain_adapter.add_domain_adversarial_training(
        X_source_scaled[:500], X_target_shifted[:500], y_source[:500]
    )
    
    # 6. Visualize results
    print("\n6. Visualizing Results")
    print("-" * 60)
    
    framework.visualize_transfer_results(transfer_results)
    
    # 7. Feature visualization
    print("\n7. Feature Analysis")
    print("-" * 60)
    
    # Extract features from different layers
    transfer_model = TransferLearningNN(source_model, freeze_layers=2)
    
    # Source features
    source_features = transfer_model.extract_features(X_source_scaled[:200])
    target_features = transfer_model.extract_features(X_target[:200])
    
    # Simple feature comparison
    print(f"Source feature statistics:")
    print(f"  Mean: {np.mean(source_features):.4f}")
    print(f"  Std: {np.std(source_features):.4f}")
    
    print(f"Target feature statistics:")
    print(f"  Mean: {np.mean(target_features):.4f}")
    print(f"  Std: {np.std(target_features):.4f}")
    
    # Feature distribution comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(source_features.flatten(), bins=50, alpha=0.7, label='Source', density=True)
    plt.hist(target_features.flatten(), bins=50, alpha=0.7, label='Target', density=True)
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.title('Feature Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature correlation
    plt.subplot(1, 2, 2)
    feature_correlation = np.corrcoef(
        np.mean(source_features, axis=0), 
        np.mean(target_features, axis=0)
    )[0, 1]
    
    plt.scatter(np.mean(source_features, axis=0), np.mean(target_features, axis=0), alpha=0.6)
    plt.xlabel('Source Feature Means')
    plt.ylabel('Target Feature Means')
    plt.title(f'Feature Correlation (r={feature_correlation:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line
    min_val = min(np.min(np.mean(source_features, axis=0)), np.min(np.mean(target_features, axis=0)))
    max_val = max(np.max(np.mean(source_features, axis=0)), np.max(np.mean(target_features, axis=0)))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'framework': framework,
        'source_model': source_model,
        'transfer_results': transfer_results,
        'domain_accuracy': domain_accuracy
    }

# Advanced transfer learning utilities
class TransferLearningAnalyzer:
    """Utility class for analyzing transfer learning experiments"""
    
    @staticmethod
    def compute_transferability_score(source_accuracy: float, 
                                    target_accuracy_scratch: float,
                                    target_accuracy_transfer: float) -> float:
        """
        Compute transferability score
        
        Score > 1: Positive transfer
        Score = 1: No transfer
        Score < 1: Negative transfer
        """
        if target_accuracy_scratch == 0:
            return float('inf') if target_accuracy_transfer > 0 else 1.0
        
        return target_accuracy_transfer / target_accuracy_scratch
    
    @staticmethod
    def compute_data_efficiency_gain(transfer_results: TransferResults, 
                                   target_accuracy: float = 0.8) -> Optional[float]:
        """
        Compute data efficiency gain (how much less data needed to reach target accuracy)
        """
        data_efficiency = transfer_results.data_efficiency
        
        # Find minimum data fraction that achieves target accuracy
        for fraction_str, accuracy in data_efficiency.items():
            if accuracy >= target_accuracy:
                fraction = float(fraction_str.replace('%', '')) / 100
                return 1.0 / fraction  # e.g., if 25% data needed, gain is 4x
        
        return None
    
    @staticmethod
    def analyze_feature_similarity(source_features: np.ndarray, 
                                 target_features: np.ndarray) -> Dict[str, float]:
        """Analyze similarity between source and target features"""
        
        # Statistical similarity
        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)
        source_std = np.std(source_features, axis=0)
        target_std = np.std(target_features, axis=0)
        
        # Correlation between mean features
        mean_correlation = np.corrcoef(source_mean, target_mean)[0, 1]
        
        # KL divergence approximation (assuming normal distributions)
        kl_div = np.mean(np.log(target_std / source_std) + 
                        (source_std**2 + (source_mean - target_mean)**2) / (2 * target_std**2) - 0.5)
        
        # Cosine similarity
        cosine_sim = np.dot(source_mean, target_mean) / (np.linalg.norm(source_mean) * np.linalg.norm(target_mean))
        
        return {
            'mean_correlation': mean_correlation,
            'kl_divergence': kl_div,
            'cosine_similarity': cosine_sim
        }

# Real-world transfer learning example with digits
def digits_transfer_learning_example():
    """Transfer learning example with digit recognition"""
    
    print("=== Digits Transfer Learning Example ===\n")
    
    # Load digit data
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Scenario: Pretrain on digits 0-7, transfer to classify 8-9
    source_mask = y < 8
    target_mask = y >= 8
    
    X_source = X[source_mask]
    y_source = y[source_mask]
    X_target = X[target_mask]
    y_target = y[target_mask] - 8  # Relabel as 0, 1
    
    print(f"Source task: Classify digits 0-7")
    print(f"Source data: {X_source.shape[0]} samples, {len(np.unique(y_source))} classes")
    print(f"Target task: Classify digits 8-9")
    print(f"Target data: {X_target.shape[0]} samples, {len(np.unique(y_target))} classes")
    
    # Scale data
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # Train source model
    print("\nTraining source model...")
    source_model = SimpleNN(
        input_size=X_source.shape[1],
        hidden_sizes=[64, 32],
        output_size=len(np.unique(y_source)),
        learning_rate=0.01
    )
    
    source_model.fit(X_source_scaled, y_source, epochs=100, verbose=False)
    source_accuracy = accuracy_score(y_source, source_model.predict(X_source_scaled))
    print(f"Source model accuracy: {source_accuracy:.4f}")
    
    # Split target data
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target_scaled, y_target, test_size=0.3, random_state=42
    )
    
    # Compare methods
    print("\nComparing transfer learning approaches...")
    
    # 1. From scratch
    scratch_model = SimpleNN(
        input_size=X_target.shape[1],
        hidden_sizes=[64, 32],
        output_size=2,
        learning_rate=0.01
    )
    scratch_model.fit(X_target_train, y_target_train, epochs=50, verbose=False)
    scratch_accuracy = accuracy_score(y_target_test, scratch_model.predict(X_target_test))
    
    # 2. Feature extraction
    fe_model = TransferLearningNN(source_model, freeze_layers=1)
    fe_model.fine_tune(X_target_train, y_target_train, epochs=30, freeze_base=True)
    fe_accuracy = accuracy_score(y_target_test, fe_model.predict(X_target_test))
    
    # 3. Fine-tuning
    ft_model = TransferLearningNN(source_model, freeze_layers=0)
    ft_model.fine_tune(X_target_train, y_target_train, epochs=30, freeze_base=False)
    ft_accuracy = accuracy_score(y_target_test, ft_model.predict(X_target_test))
    
    print(f"\nResults on digit 8-9 classification:")
    print(f"From scratch: {scratch_accuracy:.4f}")
    print(f"Feature extraction: {fe_accuracy:.4f}")
    print(f"Fine-tuning: {ft_accuracy:.4f}")
    
    # Data efficiency analysis
    print(f"\nImprovement over baseline:")
    print(f"Feature extraction: {fe_accuracy - scratch_accuracy:+.4f}")
    print(f"Fine-tuning: {ft_accuracy - scratch_accuracy:+.4f}")
    
    return {
        'scratch_accuracy': scratch_accuracy,
        'fe_accuracy': fe_accuracy,
        'ft_accuracy': ft_accuracy
    }

# Run demonstrations
if __name__ == "__main__":
    # Main demonstration
    main_results = demonstrate_transfer_learning()
    
    # Digits example
    print("\n" + "="*80)
    digits_results = digits_transfer_learning_example()
    
    print("\n=== Transfer Learning Summary ===")
    print("Key Benefits:")
    print("1. Reduced training time")
    print("2. Improved performance with limited data")
    print("3. Better feature representations")
    print("4. Faster convergence")
    
    print("\nKey Challenges:")
    print("1. Negative transfer when domains are too different")
    print("2. Choosing appropriate layers to freeze/fine-tune")
    print("3. Learning rate scheduling for fine-tuning")
    print("4. Domain adaptation for distribution shift")
    
    print("\n=== Transfer Learning Demonstration Complete ===")
```

This comprehensive implementation demonstrates:

### Core Transfer Learning Concepts:
1. **Feature Extraction** - Using pre-trained features without modification
2. **Fine-tuning** - Adapting pre-trained weights to new tasks
3. **Domain Adaptation** - Handling distribution shift between domains

### Practical Implementation:
1. **Complete Transfer Learning Framework** with multiple strategies
2. **Real-world Examples** (digit classification, synthetic data)
3. **Performance Comparisons** between different approaches
4. **Data Efficiency Analysis** showing reduced data requirements

### Advanced Features:
1. **Domain Adversarial Training** for domain adaptation
2. **Feature Similarity Analysis** for transferability assessment
3. **Comprehensive Evaluation Metrics** for transfer success
4. **Visualization Tools** for understanding transfer effectiveness

The implementation shows how transfer learning can significantly improve performance and reduce training time, especially when target data is limited.

---

## Question 14

**How do you implement a recommendation system using Python?**

**Answer:** _[To be filled]_

---

## Question 15

**How would you develop a spam detection system using Python?**

**Answer:** _[To be filled]_

---

## Question 16

**Describe the steps to design a Python system that predicts house prices based on multiple features.**

**Answer:** _[To be filled]_

---

## Question 17

**Explain how you would create a sentiment analysis model with Python.**

**Answer:** _[To be filled]_

---

## Question 18

**How would you build and deploy a machine-learning model for predicting customer churn?**

**Answer:** _[To be filled]_

---

## Question 19

**Discuss the development of a system to    classify images using Python.**

**Answer:** _[To be filled]_

---

## Question 20

**Propose a method for detecting fraudulent transactions with Python-based machine learning.**

**Answer:** _[To be filled]_

---

## Question 21

**Create a Python generator that yields batches of data from a large dataset.**

**Answer:** _[To be filled]_

---

## Question 22

**Implement a convolutional neural network using PyTorch or TensorFlow in Python.**

**Answer:** _[To be filled]_

---

## Question 23

**Develop a Python function that uses genetic algorithms to optimize a simple problem.**

**Answer:** _[To be filled]_

---

## Question 24

**Code a Python simulation that compares different optimization techniques on a fixed dataset.**

**Answer:** _[To be filled]_

---

## Question 25

**Write a Python script that visualizes decision boundaries for a classification model.**

**Answer:** _[To be filled]_

---

## Question 26

**Create a Python implementation of the A* search algorithm for pathfinding on a grid.**

**Answer:** _[To be filled]_

---

## Question 27

**Implement a simple reinforcement learning agent that learns to play a basic game.**

**Answer:** _[To be filled]_

---

## Question 28

**Use a Python library to perform time-series forecasting on stock market data.**

**Answer:** _[To be filled]_

---

## Question 29

**What is federated learning, and how can Python be used to implement it?**

**Answer:** _[To be filled]_

---

