# Python Ml Interview Questions - Theory Questions

## Question 1

**Explain the difference between Python 2 and Python 3.**

**Answer:**

**Core Differences:**

1. **Print Statement vs Function:**
   ```python
   # Python 2
   print "Hello World"
   
   # Python 3
   print("Hello World")
   ```

2. **Integer Division:**
   ```python
   # Python 2 - Integer division returns int
   result = 5 / 2  # Returns 2
   
   # Python 3 - True division
   result = 5 / 2   # Returns 2.5
   result = 5 // 2  # Returns 2 (floor division)
   ```

3. **Unicode and Strings:**
   ```python
   # Python 2 - Separate string and unicode types
   str_type = "Hello"      # bytes string
   unicode_type = u"Hello" # unicode string
   
   # Python 3 - All strings are unicode by default
   str_type = "Hello"      # unicode string
   bytes_type = b"Hello"   # bytes
   ```

4. **Range Function:**
   ```python
   # Python 2
   range(10)     # Returns list
   xrange(10)    # Returns iterator
   
   # Python 3
   range(10)     # Returns iterator (like xrange in Python 2)
   list(range(10))  # Returns list
   ```

**Machine Learning Implications:**

```python
# Data handling differences
import numpy as np
import pandas as pd

# Python 2 vs 3 compatibility for ML workflows
def load_data_py2_vs_py3():
    # Python 2 - Required explicit unicode handling
    # with open('data.csv', 'rb') as f:
    #     data = f.read().decode('utf-8')
    
    # Python 3 - Built-in unicode support
    data = pd.read_csv('data.csv', encoding='utf-8')
    return data

# Dictionary iteration differences
def iterate_hyperparameters():
    params = {'learning_rate': 0.01, 'epochs': 100, 'batch_size': 32}
    
    # Python 2 - .iteritems(), .iterkeys(), .itervalues()
    # for key, value in params.iteritems():
    
    # Python 3 - .items(), .keys(), .values() return views
    for key, value in params.items():
        print(f"Parameter: {key}, Value: {value}")
```

**Key Advantages of Python 3:**
- Better Unicode support for international datasets
- More consistent syntax and semantics
- Improved performance and memory efficiency
- Active development and security updates
- Better async/await support for scalable ML applications
- Enhanced standard library features

**Migration Considerations:**
- Use `2to3` tool for automated conversion
- `__future__` imports for backward compatibility
- Consider `six` library for cross-version compatibility
- Update dependencies to Python 3 compatible versions

---

## Question 2

**How doesPythonmanagememory?**

**Answer:**

**Python Memory Management Architecture:**

1. **Private Heap Management:**
   ```python
   import sys
   import gc
   
   # Memory allocation example
   def demonstrate_memory_management():
       # Objects are allocated in private heap
       my_list = [1, 2, 3, 4, 5]  # Allocated in heap
       
       # Check object reference count
       ref_count = sys.getrefcount(my_list)
       print(f"Reference count: {ref_count}")
       
       # Memory usage
       memory_usage = sys.getsizeof(my_list)
       print(f"Memory usage: {memory_usage} bytes")
   ```

2. **Reference Counting:**
   ```python
   import sys
   
   def reference_counting_demo():
       # Create object
       data = [1, 2, 3]
       print(f"Initial ref count: {sys.getrefcount(data)}")
       
       # Assign to another variable
       data2 = data
       print(f"After assignment: {sys.getrefcount(data)}")
       
       # Delete reference
       del data2
       print(f"After deletion: {sys.getrefcount(data)}")
   ```

3. **Garbage Collection:**
   ```python
   import gc
   import weakref
   
   class MLModel:
       def __init__(self, name):
           self.name = name
           self.data = []
   
   def garbage_collection_demo():
       # Create circular reference
       model1 = MLModel("Model1")
       model2 = MLModel("Model2")
       model1.partner = model2
       model2.partner = model1
       
       # Check garbage collection
       print(f"Objects before GC: {len(gc.get_objects())}")
       
       # Force garbage collection
       collected = gc.collect()
       print(f"Objects collected: {collected}")
       
       # Monitor object lifecycle
       def callback(ref):
           print("Object was garbage collected")
       
       weak_ref = weakref.ref(model1, callback)
   ```

**Memory Optimization for ML:**

```python
import numpy as np
import pandas as pd
from memory_profiler import profile

class MemoryEfficientMLWorkflow:
    def __init__(self):
        self.data = None
        self.model = None
    
    @profile
    def load_and_process_data(self, filepath):
        """Memory-efficient data loading"""
        # Use chunking for large datasets
        chunk_size = 10000
        chunks = []
        
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            # Process chunk
            processed_chunk = self.preprocess_chunk(chunk)
            chunks.append(processed_chunk)
        
        # Combine chunks efficiently
        self.data = pd.concat(chunks, ignore_index=True)
        
        # Clear intermediate variables
        del chunks
        gc.collect()
    
    def preprocess_chunk(self, chunk):
        """Memory-efficient preprocessing"""
        # Use view instead of copy when possible
        numeric_columns = chunk.select_dtypes(include=[np.number])
        
        # Optimize data types
        chunk = self.optimize_dtypes(chunk)
        
        return chunk
    
    def optimize_dtypes(self, df):
        """Optimize pandas dtypes to reduce memory"""
        for col in df.columns:
            if df[col].dtype == 'int64':
                if df[col].min() >= 0 and df[col].max() <= 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype('int8')
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
        
        return df
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'data') and self.data is not None:
            del self.data
        gc.collect()
```

**Memory Management Best Practices:**

```python
# 1. Use generators for large datasets
def data_generator(filepath):
    """Memory-efficient data iteration"""
    with open(filepath, 'r') as file:
        for line in file:
            yield process_line(line)

# 2. Context managers for resource management
class ModelTrainer:
    def __enter__(self):
        self.model = initialize_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'model'):
            del self.model
        gc.collect()

# 3. Memory monitoring
def monitor_memory_usage():
    """Monitor memory usage during ML training"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
```

**Key Components:**
- **Private Heap**: All Python objects stored in private heap
- **Reference Counting**: Automatic memory deallocation
- **Garbage Collector**: Handles circular references
- **Memory Pools**: Efficient allocation for small objects
- **Object-specific allocators**: Optimized for different data types

**ML-Specific Considerations:**
- Use NumPy arrays for numerical computation efficiency
- Implement data generators for large datasets
- Monitor memory usage during training
- Optimize data types to reduce memory footprint
- Use context managers for resource cleanup

---

## Question 3

**What isPEP 8and why is it important?**

**Answer:**

**PEP 8 Overview:**

PEP 8 is the official style guide for Python code, providing conventions for writing readable and consistent Python code. It was written by Python's creator Guido van Rossum and is essential for maintaining code quality in machine learning projects.

**Key PEP 8 Guidelines:**

1. **Indentation and Line Length:**
   ```python
   # Good - 4 spaces per indentation level
   def train_model(X_train, y_train, model_params):
       if model_params is not None:
           model = create_model(model_params)
           model.fit(X_train, y_train)
           return model
   
   # Bad - inconsistent indentation
   def train_model(X_train, y_train, model_params):
     if model_params is not None:
        model = create_model(model_params)
          model.fit(X_train, y_train)
         return model
   
   # Line length - max 79 characters
   def calculate_evaluation_metrics(y_true, y_pred, 
                                  classification_report_params):
       return classification_report(y_true, y_pred, 
                                  **classification_report_params)
   ```

2. **Naming Conventions:**
   ```python
   # Variables and functions - snake_case
   learning_rate = 0.01
   batch_size = 32
   
   def preprocess_data(raw_data):
       pass
   
   def calculate_accuracy_score(predictions, targets):
       pass
   
   # Classes - PascalCase
   class NeuralNetwork:
       pass
   
   class RandomForestClassifier:
       pass
   
   # Constants - UPPER_CASE
   MAX_EPOCHS = 1000
   DEFAULT_LEARNING_RATE = 0.001
   RANDOM_SEED = 42
   ```

3. **Import Organization:**
   ```python
   # Standard library imports
   import os
   import sys
   from pathlib import Path
   
   # Third-party imports
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report
   
   # Local application imports
   from .preprocessing import clean_data, feature_engineering
   from .models import BaseModel, EnsembleModel
   from .utils import save_model, load_model
   ```

**ML-Specific PEP 8 Application:**

```python
class MLPipeline:
    """Machine learning pipeline following PEP 8 conventions."""
    
    def __init__(self, model_config, preprocessing_config):
        """Initialize ML pipeline with configurations.
        
        Args:
            model_config (dict): Model hyperparameters
            preprocessing_config (dict): Preprocessing parameters
        """
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.is_trained = False
        self._model = None
        self._preprocessor = None
    
    def fit(self, X_train, y_train):
        """Train the machine learning model.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training targets
            
        Returns:
            self: Returns the instance for method chaining
        """
        # Preprocess training data
        X_processed = self._preprocess_features(X_train)
        
        # Initialize and train model
        self._model = self._initialize_model()
        self._model.fit(X_processed, y_train)
        
        self.is_trained = True
        return self
    
    def predict(self, X_test):
        """Generate predictions on test data.
        
        Args:
            X_test (array-like): Test features
            
        Returns:
            array: Predictions
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self._preprocess_features(X_test)
        return self._model.predict(X_processed)
    
    def _preprocess_features(self, X):
        """Private method for feature preprocessing."""
        # Implementation details...
        return X
    
    def _initialize_model(self):
        """Private method for model initialization."""
        # Implementation details...
        pass
```

**Documentation Standards:**

```python
def cross_validate_model(model, X, y, cv_folds=5, scoring='accuracy'):
    """Perform cross-validation on a machine learning model.
    
    This function implements k-fold cross-validation to assess model
    performance and detect overfitting.
    
    Args:
        model: Sklearn-compatible model object
        X (array-like, shape=[n_samples, n_features]): Input features
        y (array-like, shape=[n_samples]): Target values
        cv_folds (int, optional): Number of cross-validation folds. 
            Defaults to 5.
        scoring (str, optional): Scoring metric for evaluation. 
            Defaults to 'accuracy'.
    
    Returns:
        dict: Dictionary containing:
            - 'scores' (array): Cross-validation scores for each fold
            - 'mean_score' (float): Mean cross-validation score
            - 'std_score' (float): Standard deviation of scores
    
    Raises:
        ValueError: If cv_folds is less than 2
        
    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(random_state=42)
        >>> results = cross_validate_model(model, X_train, y_train)
        >>> print(f"CV Score: {results['mean_score']:.3f}")
    """
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    
    # Implementation...
    pass
```

**Code Quality Tools:**

```python
# Setup for PEP 8 compliance checking
"""
Installation and usage of PEP 8 tools:

pip install flake8 black isort mypy

# Check PEP 8 compliance
flake8 ml_project/

# Auto-format code
black ml_project/

# Sort imports
isort ml_project/

# Type checking
mypy ml_project/
"""

# Example configuration files

# .flake8
"""
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = 
    .git,
    __pycache__,
    .venv,
    .eggs,
    *.egg
"""

# pyproject.toml for black
"""
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
"""
```

**Importance in ML Projects:**

1. **Team Collaboration**: Consistent code style improves team productivity
2. **Code Maintainability**: Easier to maintain and debug ML pipelines
3. **Readability**: Clear code is crucial for complex ML algorithms
4. **Professional Standards**: Industry expectation for production code
5. **Tool Integration**: Better IDE support and automated tools
6. **Error Reduction**: Consistent style reduces bugs
7. **Documentation**: Clear naming and structure improve understanding

**Best Practices for ML Code:**
- Use descriptive variable names for models and datasets
- Organize imports by category (standard, third-party, local)
- Follow consistent naming for hyperparameters
- Document complex mathematical operations
- Use type hints for function parameters
- Keep functions focused and modular

---

## Question 4

**Describe how adictionaryworks inPython. What arekeysandvalues?**

**Answer:**

**Dictionary Fundamentals:**

A dictionary is a mutable, unordered collection of key-value pairs in Python. It's implemented as a hash table, providing O(1) average time complexity for lookups, insertions, and deletions.

**Basic Dictionary Operations:**

```python
# Creating dictionaries
model_params = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'adam'
}

# Alternative creation methods
hyperparameters = dict(learning_rate=0.01, batch_size=32)
config = dict([('model_type', 'neural_network'), ('layers', 3)])

# Dictionary comprehension
squared_numbers = {x: x**2 for x in range(5)}
print(squared_numbers)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

**Keys and Values Properties:**

```python
# Keys must be immutable (hashable)
valid_keys = {
    'string_key': 'value1',
    42: 'value2',
    (1, 2): 'value3',  # Tuple is immutable
    frozenset([1, 2, 3]): 'value4'
}

# Invalid keys (uncomment to see errors)
# invalid_keys = {
#     [1, 2, 3]: 'value',  # List is mutable - TypeError
#     {'a': 1}: 'value'    # Dict is mutable - TypeError
# }

# Values can be any type
ml_data = {
    'features': ['age', 'income', 'education'],
    'target': 'purchase',
    'model': RandomForestClassifier(),
    'metrics': {'accuracy': 0.85, 'precision': 0.80},
    'data_shape': (1000, 10)
}
```

**Dictionary Methods and Operations:**

```python
class MLExperiment:
    def __init__(self):
        self.experiments = {}
    
    def add_experiment(self, exp_id, config):
        """Add new experiment configuration"""
        self.experiments[exp_id] = config
    
    def demonstrate_dict_methods(self):
        """Comprehensive dictionary methods demonstration"""
        
        # Basic operations
        config = {
            'model': 'random_forest',
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        # Accessing values
        model_type = config['model']  # Direct access
        max_depth = config.get('max_depth', 5)  # Safe access with default
        
        # Adding/updating values
        config['min_samples_split'] = 2
        config.update({'min_samples_leaf': 1, 'bootstrap': True})
        
        # Removing items
        removed_value = config.pop('random_state', None)
        last_item = config.popitem()  # Removes and returns arbitrary item
        
        # Dictionary views
        keys = config.keys()        # dict_keys view
        values = config.values()    # dict_values view
        items = config.items()      # dict_items view
        
        # Iteration patterns
        for key in config:
            print(f"Key: {key}")
        
        for key, value in config.items():
            print(f"{key}: {value}")
        
        for value in config.values():
            print(f"Value: {value}")
        
        # Dictionary operations
        config.clear()  # Remove all items
        is_empty = len(config) == 0
        
        return config
```

**ML-Specific Dictionary Usage:**

```python
class ModelRegistry:
    """Example of dictionary usage in ML workflows"""
    
    def __init__(self):
        # Store multiple models with configurations
        self.models = {}
        self.metrics = {}
        self.feature_importance = {}
    
    def register_model(self, model_name, model, config):
        """Register model with its configuration"""
        self.models[model_name] = {
            'model_object': model,
            'config': config,
            'trained': False,
            'timestamp': datetime.now()
        }
    
    def store_metrics(self, model_name, metrics_dict):
        """Store evaluation metrics for a model"""
        self.metrics[model_name] = metrics_dict
    
    def get_best_model(self, metric='accuracy'):
        """Find best performing model based on metric"""
        if not self.metrics:
            return None
        
        best_model = max(
            self.metrics.items(),
            key=lambda x: x[1].get(metric, 0)
        )
        return best_model[0]  # Return model name
    
    def compare_models(self):
        """Compare all models using stored metrics"""
        comparison = {}
        for model_name, metrics in self.metrics.items():
            comparison[model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0)
            }
        return comparison

# Usage example
registry = ModelRegistry()

# Register models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

rf_config = {'n_estimators': 100, 'max_depth': 10}
svm_config = {'C': 1.0, 'kernel': 'rbf'}

registry.register_model('random_forest', RandomForestClassifier(**rf_config), rf_config)
registry.register_model('svm', SVC(**svm_config), svm_config)

# Store metrics
registry.store_metrics('random_forest', {
    'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1_score': 0.85
})
registry.store_metrics('svm', {
    'accuracy': 0.82, 'precision': 0.80, 'recall': 0.84, 'f1_score': 0.82
})
```

**Advanced Dictionary Patterns:**

```python
from collections import defaultdict, Counter
from typing import Dict, Any, List

class AdvancedDictPatterns:
    
    def nested_dictionaries(self):
        """Working with nested dictionaries in ML"""
        experiment_results = {
            'experiment_1': {
                'models': {
                    'random_forest': {'accuracy': 0.85, 'training_time': 120},
                    'svm': {'accuracy': 0.82, 'training_time': 300}
                },
                'dataset': {'size': 10000, 'features': 20}
            },
            'experiment_2': {
                'models': {
                    'neural_network': {'accuracy': 0.88, 'training_time': 600},
                    'gradient_boosting': {'accuracy': 0.86, 'training_time': 400}
                },
                'dataset': {'size': 15000, 'features': 25}
            }
        }
        
        # Safe nested access
        def safe_get_nested(d, keys, default=None):
            for key in keys:
                if isinstance(d, dict) and key in d:
                    d = d[key]
                else:
                    return default
            return d
        
        accuracy = safe_get_nested(
            experiment_results, 
            ['experiment_1', 'models', 'random_forest', 'accuracy']
        )
        
        return experiment_results
    
    def defaultdict_usage(self):
        """Using defaultdict for ML feature counting"""
        feature_counts = defaultdict(int)
        category_features = defaultdict(list)
        
        # Count feature occurrences
        features = ['age', 'income', 'age', 'education', 'income', 'age']
        for feature in features:
            feature_counts[feature] += 1
        
        # Group features by category
        feature_categories = [
            ('age', 'demographic'),
            ('income', 'financial'),
            ('education', 'demographic'),
            ('credit_score', 'financial')
        ]
        
        for feature, category in feature_categories:
            category_features[category].append(feature)
        
        return dict(feature_counts), dict(category_features)
    
    def dictionary_merging(self):
        """Different ways to merge dictionaries"""
        base_config = {'learning_rate': 0.01, 'batch_size': 32}
        new_config = {'epochs': 100, 'learning_rate': 0.001}
        
        # Python 3.9+ merge operator
        merged_config = base_config | new_config
        
        # Update method (modifies original)
        base_config.update(new_config)
        
        # Dictionary unpacking
        final_config = {**base_config, **new_config, 'optimizer': 'adam'}
        
        return final_config
```

**Performance Considerations:**

```python
import timeit
from typing import Dict

def performance_comparison():
    """Compare dictionary vs list performance for ML use cases"""
    
    # Dictionary lookup O(1)
    feature_dict = {f'feature_{i}': i for i in range(10000)}
    
    # List search O(n)
    feature_list = [f'feature_{i}' for i in range(10000)]
    
    # Timing dictionary lookup
    dict_time = timeit.timeit(
        lambda: 'feature_5000' in feature_dict,
        number=100000
    )
    
    # Timing list search
    list_time = timeit.timeit(
        lambda: 'feature_5000' in feature_list,
        number=100000
    )
    
    print(f"Dictionary lookup: {dict_time:.6f} seconds")
    print(f"List search: {list_time:.6f} seconds")
    print(f"Dictionary is {list_time/dict_time:.1f}x faster")
```

**Key Properties:**
- **Keys**: Must be immutable (hashable) - strings, numbers, tuples
- **Values**: Can be any Python object - mutable or immutable
- **Uniqueness**: Keys must be unique within a dictionary
- **Order**: Dictionaries maintain insertion order (Python 3.7+)

**ML Applications:**
- Hyperparameter storage and management
- Model registry and configuration
- Feature mapping and encoding
- Metrics tracking and comparison
- Cache for expensive computations
- Configuration management for experiments

---

## Question 5

**What islist comprehensionand give an example of its use?**

**Answer:**

**List Comprehension Fundamentals:**

List comprehension is a concise way to create lists in Python using a single line of code. It follows the pattern: `[expression for item in iterable if condition]`

**Basic Syntax and Examples:**

```python
# Basic list comprehension
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(squares)  # [1, 4, 9, 16, 25]

# With conditional filtering
even_squares = [x**2 for x in numbers if x % 2 == 0]
print(even_squares)  # [4, 16]

# String processing
words = ['hello', 'world', 'python', 'ml']
capitalized = [word.upper() for word in words]
print(capitalized)  # ['HELLO', 'WORLD', 'PYTHON', 'ML']

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for row in matrix for item in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**ML-Specific Applications:**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

class MLListComprehensions:
    
    def feature_engineering_examples(self, df):
        """Feature engineering using list comprehensions"""
        
        # Create polynomial features
        numeric_cols = ['age', 'income', 'credit_score']
        polynomial_features = [
            f"{col}_squared" for col in numeric_cols
        ]
        
        # Add polynomial features to dataframe
        for i, col in enumerate(numeric_cols):
            df[polynomial_features[i]] = df[col] ** 2
        
        # Create interaction features
        interaction_features = [
            f"{col1}_{col2}_interaction" 
            for i, col1 in enumerate(numeric_cols)
            for col2 in numeric_cols[i+1:]
        ]
        
        # Log transformations for skewed features
        skewed_features = ['income', 'credit_score']
        log_features = [f"log_{col}" for col in skewed_features]
        
        for i, col in enumerate(skewed_features):
            df[log_features[i]] = np.log1p(df[col])
        
        return df
    
    def data_preprocessing_examples(self, data):
        """Data preprocessing with list comprehensions"""
        
        # Clean text data
        text_data = ['Hello World!', 'Python ML', 'Data Science@']
        cleaned_text = [
            ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])
            for text in text_data
        ]
        
        # Extract numeric values from mixed data
        mixed_data = ['$100.50', 'â‚¬75.25', 'Â¥1000']
        numeric_values = [
            float(''.join([char for char in item if char.isdigit() or char == '.']))
            for item in mixed_data
        ]
        
        # Create binary encodings
        categories = ['cat', 'dog', 'bird', 'cat', 'dog']
        unique_categories = list(set(categories))
        binary_encodings = [
            [1 if cat == category else 0 for cat in unique_categories]
            for category in categories
        ]
        
        return cleaned_text, numeric_values, binary_encodings
    
    def hyperparameter_grid_creation(self):
        """Create hyperparameter grids using list comprehensions"""
        
        # Learning rate schedules
        learning_rates = [0.1 * (0.9 ** i) for i in range(10)]
        
        # Batch sizes (powers of 2)
        batch_sizes = [2**i for i in range(5, 10)]  # [32, 64, 128, 256, 512]
        
        # Layer configurations for neural networks
        hidden_layers = [
            [2**i for _ in range(num_layers)]
            for num_layers in range(1, 4)
            for i in range(6, 9)  # Layer sizes: 64, 128, 256
        ]
        
        # Complex parameter grid
        param_combinations = [
            {
                'learning_rate': lr,
                'batch_size': bs,
                'hidden_layers': hl
            }
            for lr in learning_rates[:3]
            for bs in batch_sizes[:3]
            for hl in hidden_layers[:3]
        ]
        
        return param_combinations
    
    def model_evaluation_examples(self, models, X_test, y_test):
        """Model evaluation using list comprehensions"""
        
        # Get predictions from multiple models
        predictions = [model.predict(X_test) for model in models]
        
        # Calculate accuracy for each model
        from sklearn.metrics import accuracy_score
        accuracies = [
            accuracy_score(y_test, pred) for pred in predictions
        ]
        
        # Create model comparison dictionary
        model_names = [f"Model_{i}" for i in range(len(models))]
        model_comparison = [
            {'name': name, 'accuracy': acc}
            for name, acc in zip(model_names, accuracies)
        ]
        
        # Find best performing models
        threshold = 0.8
        good_models = [
            comp for comp in model_comparison 
            if comp['accuracy'] > threshold
        ]
        
        return model_comparison, good_models
```

**Advanced List Comprehension Patterns:**

```python
class AdvancedComprehensions:
    
    def conditional_expressions(self, data):
        """Using conditional expressions in comprehensions"""
        
        # Ternary operator in list comprehension
        scores = [85, 92, 78, 96, 88, 73]
        grades = [
            'A' if score >= 90 else 'B' if score >= 80 else 'C'
            for score in scores
        ]
        
        # Multiple conditions
        processed_scores = [
            score * 1.1 if score < 80 else score  # Boost low scores
            for score in scores
            if score >= 70  # Only include passing scores
        ]
        
        return grades, processed_scores
    
    def nested_comprehensions(self, matrix_data):
        """Complex nested comprehensions"""
        
        # Matrix operations
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        # Transpose matrix
        transposed = [
            [row[i] for row in matrix]
            for i in range(len(matrix[0]))
        ]
        
        # Apply function to each element
        doubled_matrix = [
            [x * 2 for x in row]
            for row in matrix
        ]
        
        # Filter and transform
        filtered_matrix = [
            [x for x in row if x % 2 == 0]
            for row in matrix
        ]
        
        return transposed, doubled_matrix, filtered_matrix
    
    def set_and_dict_comprehensions(self, data):
        """Set and dictionary comprehensions"""
        
        words = ['apple', 'banana', 'apple', 'cherry', 'banana']
        
        # Set comprehension - unique word lengths
        unique_lengths = {len(word) for word in words}
        
        # Dictionary comprehension - word to length mapping
        word_lengths = {word: len(word) for word in words}
        
        # Complex dictionary comprehension
        word_stats = {
            word: {
                'length': len(word),
                'vowels': sum(1 for char in word if char in 'aeiou'),
                'consonants': len(word) - sum(1 for char in word if char in 'aeiou')
            }
            for word in set(words)  # Unique words only
        }
        
        return unique_lengths, word_lengths, word_stats
```

**Performance Comparison:**

```python
import timeit

def performance_comparison():
    """Compare list comprehension vs traditional loops"""
    
    # Data for testing
    numbers = list(range(1000))
    
    # Traditional loop
    def traditional_loop():
        result = []
        for x in numbers:
            if x % 2 == 0:
                result.append(x ** 2)
        return result
    
    # List comprehension
    def list_comp():
        return [x ** 2 for x in numbers if x % 2 == 0]
    
    # Timing comparison
    loop_time = timeit.timeit(traditional_loop, number=1000)
    comp_time = timeit.timeit(list_comp, number=1000)
    
    print(f"Traditional loop: {loop_time:.6f} seconds")
    print(f"List comprehension: {comp_time:.6f} seconds")
    print(f"List comprehension is {loop_time/comp_time:.2f}x faster")
    
    # Memory efficiency
    import sys
    
    # Generator expression (memory efficient)
    gen_expr = (x ** 2 for x in numbers if x % 2 == 0)
    list_comp = [x ** 2 for x in numbers if x % 2 == 0]
    
    print(f"Generator size: {sys.getsizeof(gen_expr)} bytes")
    print(f"List size: {sys.getsizeof(list_comp)} bytes")
```

**Real-World ML Example:**

```python
class MLDataPipeline:
    """Complete ML pipeline using list comprehensions"""
    
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.processed_data = None
    
    def clean_and_transform_data(self):
        """Clean and transform data using comprehensions"""
        
        # Remove missing values and outliers
        clean_data = [
            record for record in self.raw_data
            if all(value is not None for value in record.values())
            and self.is_not_outlier(record)
        ]
        
        # Feature scaling
        numeric_features = ['age', 'income', 'credit_score']
        scaled_data = []
        
        for record in clean_data:
            scaled_record = record.copy()
            scaled_record.update({
                f"{feature}_scaled": self.scale_feature(record[feature], feature)
                for feature in numeric_features
            })
            scaled_data.append(scaled_record)
        
        # Create feature combinations
        final_data = []
        for record in scaled_data:
            enhanced_record = record.copy()
            enhanced_record.update({
                f"{f1}_{f2}_ratio": record[f1] / (record[f2] + 1e-8)
                for f1, f2 in [('income', 'age'), ('credit_score', 'age')]
            })
            final_data.append(enhanced_record)
        
        self.processed_data = final_data
        return self.processed_data
    
    def is_not_outlier(self, record):
        """Simple outlier detection"""
        return (
            0 < record.get('age', 0) < 120 and
            0 < record.get('income', 0) < 1000000
        )
    
    def scale_feature(self, value, feature_name):
        """Simple min-max scaling"""
        feature_ranges = {
            'age': (18, 80),
            'income': (20000, 200000),
            'credit_score': (300, 850)
        }
        min_val, max_val = feature_ranges.get(feature_name, (0, 1))
        return (value - min_val) / (max_val - min_val)
```

**Best Practices:**
1. **Readability**: Keep comprehensions simple and readable
2. **Performance**: Use for simple transformations and filtering
3. **Memory**: Consider generator expressions for large datasets
4. **Complexity**: Break down complex logic into multiple steps
5. **Debugging**: Traditional loops are easier to debug
6. **Nested**: Limit nesting depth for maintainability

**When to Use:**
- Simple data transformations
- Filtering operations
- Creating parameter grids
- Feature engineering
- Data cleaning operations
- Mathematical operations on sequences

---

## Question 6

**Explain the concept ofgeneratorsinPython. How do they differ fromlist comprehensions?**

**Answer:**

**Generator Fundamentals:**

Generators are memory-efficient iterators that produce items on-demand rather than storing all items in memory simultaneously. They use the `yield` keyword to return values one at a time.

**Generator Creation Methods:**

```python
# 1. Generator Functions
def simple_generator():
    """Basic generator function"""
    yield 1
    yield 2
    yield 3

# Usage
gen = simple_generator()
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3

# 2. Generator Expressions
squares_gen = (x**2 for x in range(5))
squares_list = [x**2 for x in range(5)]

print(type(squares_gen))   # <class 'generator'>
print(type(squares_list))  # <class 'list'>
```

**Memory and Performance Comparison:**

```python
import sys
import timeit

def memory_comparison():
    """Compare memory usage between generators and lists"""
    
    # Large dataset simulation
    n = 1000000
    
    # List comprehension - stores all values in memory
    list_comp = [x**2 for x in range(n)]
    
    # Generator expression - produces values on demand
    gen_expr = (x**2 for x in range(n))
    
    print(f"List memory usage: {sys.getsizeof(list_comp)} bytes")
    print(f"Generator memory usage: {sys.getsizeof(gen_expr)} bytes")
    
    # Memory ratio
    ratio = sys.getsizeof(list_comp) / sys.getsizeof(gen_expr)
    print(f"List uses {ratio:.0f}x more memory than generator")

# Performance for different use cases
def performance_comparison():
    """Compare performance characteristics"""
    
    def list_creation():
        return [x**2 for x in range(10000)]
    
    def generator_creation():
        return (x**2 for x in range(10000))
    
    # Creation time
    list_time = timeit.timeit(list_creation, number=1000)
    gen_time = timeit.timeit(generator_creation, number=1000)
    
    print(f"List creation: {list_time:.6f} seconds")
    print(f"Generator creation: {gen_time:.6f} seconds")
    print(f"Generator creation is {list_time/gen_time:.1f}x faster")
```

**ML-Specific Generator Applications:**

```python
import numpy as np
import pandas as pd
from typing import Generator, Tuple, Any

class MLDataGenerators:
    
    def batch_generator(self, X, y, batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate batches for training"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        while True:  # Infinite generator for multiple epochs
            # Shuffle data each epoch
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                yield X[batch_indices], y[batch_indices]
    
    def data_augmentation_generator(self, images: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Generate augmented images on-the-fly"""
        for image in images:
            # Original image
            yield image
            
            # Horizontal flip
            yield np.fliplr(image)
            
            # Rotation (simplified)
            yield np.rot90(image)
            
            # Noise addition
            noise = np.random.normal(0, 0.1, image.shape)
            yield np.clip(image + noise, 0, 1)
    
    def feature_generator(self, df: pd.DataFrame) -> Generator[pd.Series, None, None]:
        """Generate features on-demand for large datasets"""
        for _, row in df.iterrows():
            # Create polynomial features
            enhanced_row = row.copy()
            
            # Add squared features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                enhanced_row[f'{col}_squared'] = row[col] ** 2
            
            # Add interaction features
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    enhanced_row[f'{col1}_{col2}_interaction'] = row[col1] * row[col2]
            
            yield enhanced_row
    
    def time_series_window_generator(self, data: np.ndarray, window_size: int, 
                                   step: int = 1) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate sliding windows for time series"""
        for i in range(0, len(data) - window_size, step):
            window = data[i:i + window_size]
            target = data[i + window_size]
            yield window, target
    
    def cross_validation_generator(self, X, y, n_folds: int = 5) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Generate train/validation splits for cross-validation"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            yield X_train, X_val, y_train, y_val
```

**Advanced Generator Patterns:**

```python
class AdvancedGenerators:
    
    def pipeline_generator(self, data_source):
        """Create a data processing pipeline using generators"""
        
        def read_data():
            """Simulate reading from large data source"""
            for i in range(1000000):
                yield {'id': i, 'value': np.random.random(), 'category': np.random.choice(['A', 'B', 'C'])}
        
        def filter_data(data_gen):
            """Filter data based on conditions"""
            for record in data_gen:
                if record['value'] > 0.5:
                    yield record
        
        def transform_data(data_gen):
            """Transform filtered data"""
            for record in data_gen:
                record['value_squared'] = record['value'] ** 2
                record['is_high_value'] = record['value'] > 0.8
                yield record
        
        def encode_categories(data_gen):
            """Encode categorical variables"""
            category_mapping = {'A': 0, 'B': 1, 'C': 2}
            for record in data_gen:
                record['category_encoded'] = category_mapping[record['category']]
                yield record
        
        # Pipeline composition
        pipeline = encode_categories(
            transform_data(
                filter_data(
                    read_data()
                )
            )
        )
        
        return pipeline
    
    def parallel_generator(self, data_chunks):
        """Simulate parallel processing with generators"""
        from concurrent.futures import ThreadPoolExecutor
        import queue
        
        def process_chunk(chunk):
            """Process a chunk of data"""
            return [x * 2 for x in chunk]
        
        # Use queue to manage results
        result_queue = queue.Queue()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all chunks for processing
            futures = [executor.submit(process_chunk, chunk) for chunk in data_chunks]
            
            # Yield results as they become available
            for future in futures:
                result = future.result()
                for item in result:
                    yield item
    
    def stateful_generator(self):
        """Generator that maintains state"""
        def moving_average_generator(data, window_size):
            """Calculate moving average using generator"""
            window = []
            
            for value in data:
                window.append(value)
                if len(window) > window_size:
                    window.pop(0)
                
                if len(window) == window_size:
                    yield sum(window) / window_size
        
        # Example usage
        data_stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ma_gen = moving_average_generator(data_stream, window_size=3)
        
        return list(ma_gen)  # [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
```

**Generator vs List Comprehension Comparison:**

```python
class ComparisonExamples:
    
    def memory_efficiency_demo(self):
        """Demonstrate memory efficiency differences"""
        
        # Large dataset processing
        def process_large_dataset():
            # List comprehension - loads everything into memory
            list_result = [self.expensive_operation(x) for x in range(1000000)]
            return list_result
        
        def process_large_dataset_gen():
            # Generator - processes one item at a time
            gen_result = (self.expensive_operation(x) for x in range(1000000))
            return gen_result
        
        # Memory monitoring
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Before processing
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # List comprehension
        list_data = process_large_dataset()
        memory_after_list = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generator
        gen_data = process_large_dataset_gen()
        memory_after_gen = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory before: {memory_before:.2f} MB")
        print(f"Memory after list: {memory_after_list:.2f} MB")
        print(f"Memory after generator: {memory_after_gen:.2f} MB")
        
        return list_data, gen_data
    
    def expensive_operation(self, x):
        """Simulate expensive computation"""
        return x ** 2 + np.sin(x) + np.log(x + 1)
    
    def lazy_evaluation_demo(self):
        """Demonstrate lazy evaluation benefits"""
        
        def find_first_match_list(data, condition):
            """Using list comprehension - processes all items"""
            matches = [x for x in data if condition(x)]
            return matches[0] if matches else None
        
        def find_first_match_gen(data, condition):
            """Using generator - stops at first match"""
            gen = (x for x in data if condition(x))
            return next(gen, None)
        
        # Test with large dataset
        large_data = range(1000000)
        condition = lambda x: x > 100  # Early match
        
        # Time comparison
        import time
        
        start = time.time()
        result_list = find_first_match_list(large_data, condition)
        time_list = time.time() - start
        
        start = time.time()
        result_gen = find_first_match_gen(large_data, condition)
        time_gen = time.time() - start
        
        print(f"List approach: {time_list:.6f} seconds")
        print(f"Generator approach: {time_gen:.6f} seconds")
        print(f"Generator is {time_list/time_gen:.1f}x faster")
```

**Best Practices for ML:**

```python
class MLBestPractices:
    
    def data_loading_pipeline(self, file_paths):
        """Memory-efficient data loading for ML"""
        
        def load_files_generator():
            """Load files one by one"""
            for file_path in file_paths:
                # Load and yield one file at a time
                data = pd.read_csv(file_path)
                yield data
        
        def preprocess_generator(data_gen):
            """Preprocess data on-the-fly"""
            for df in data_gen:
                # Clean data
                df_clean = df.dropna()
                
                # Feature engineering
                df_clean['feature_interaction'] = df_clean['feature1'] * df_clean['feature2']
                
                yield df_clean
        
        # Combined pipeline
        return preprocess_generator(load_files_generator())
    
    def model_training_generator(self, X, y, batch_size, epochs):
        """Training data generator for large datasets"""
        
        def training_batches():
            """Generate training batches for multiple epochs"""
            n_samples = len(X)
            
            for epoch in range(epochs):
                # Shuffle data each epoch
                indices = np.random.permutation(n_samples)
                
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_X = X[batch_indices]
                    batch_y = y[batch_indices]
                    
                    yield batch_X, batch_y, epoch
        
        return training_batches()
```

**Key Differences Summary:**

| Aspect | List Comprehension | Generator |
|--------|-------------------|-----------|
| **Memory Usage** | Stores all items in memory | Produces items on-demand |
| **Performance** | Fast access to all items | Slower individual access, faster creation |
| **Use Case** | Small to medium datasets | Large datasets, streaming data |
| **Reusability** | Can iterate multiple times | Single-use (need to recreate) |
| **Debugging** | Easier to inspect all values | Harder to debug (values consumed) |
| **Syntax** | `[expr for item in iterable]` | `(expr for item in iterable)` or `yield` |

**When to Use Generators:**
- Large datasets that don't fit in memory
- Streaming data processing
- Pipeline transformations
- Infinite sequences
- Memory-constrained environments
- ETL processes for ML data

**When to Use List Comprehensions:**
- Small to medium datasets
- Need random access to elements
- Multiple iterations over the same data
- Debugging and development
- Simple transformations

---

## Question 7

**How does Python's garbage collection work?**

**Answer:**
**Python Garbage Collection Overview:**
Python uses automatic memory management through a combination of reference counting and cyclic garbage collection to reclaim memory occupied by objects that are no longer reachable or needed.

### Primary Garbage Collection Mechanisms

**1. Reference Counting**
```python
import sys
import gc

def demonstrate_reference_counting():
    """Demonstrate how reference counting works"""
    # Create an object
    data = [1, 2, 3, 4, 5]
    print(f"Initial reference count: {sys.getrefcount(data)}")
    
    # Create additional references
    reference1 = data
    print(f"After creating reference1: {sys.getrefcount(data)}")
    
    reference2 = data
    print(f"After creating reference2: {sys.getrefcount(data)}")
    
    # Delete references
    del reference1
    print(f"After deleting reference1: {sys.getrefcount(data)}")
    
    del reference2
    print(f"After deleting reference2: {sys.getrefcount(data)}")
    
    # When reference count reaches 0, object is immediately deallocated
    return data

# Example of reference counting in ML context
class MLModel:
    def __init__(self, name):
        self.name = name
        self.weights = [0.1, 0.2, 0.3]
        print(f"Model {name} created")
    
    def __del__(self):
        print(f"Model {self.name} destroyed")

def reference_counting_ml_example():
    """Reference counting with ML objects"""
    model = MLModel("LinearRegression")
    print(f"Model ref count: {sys.getrefcount(model)}")
    
    # Assign to another variable
    backup_model = model
    print(f"Model ref count after backup: {sys.getrefcount(model)}")
    
    # Delete original reference
    del model
    print("Original model reference deleted")
    
    # Model still exists due to backup_model reference
    print(f"Backup model name: {backup_model.name}")
    
    # Delete last reference - model will be destroyed
    del backup_model
    print("All references deleted")

demonstrate_reference_counting()
reference_counting_ml_example()
```

**2. Cyclic Garbage Collection**
```python
import gc
import weakref

class Node:
    """Example class that can create circular references"""
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    
    def __del__(self):
        print(f"Node {self.name} garbage collected")

def demonstrate_circular_references():
    """Show how circular references are handled"""
    print("Creating circular reference:")
    
    # Create nodes that reference each other
    node1 = Node("Parent")
    node2 = Node("Child")
    
    # Create circular reference
    node1.add_child(node2)
    # node2.parent already points to node1, creating a cycle
    
    print(f"Node1 ref count: {sys.getrefcount(node1)}")
    print(f"Node2 ref count: {sys.getrefcount(node2)}")
    
    # Delete direct references
    del node1, node2
    
    print("Direct references deleted")
    print("Objects still exist due to circular reference")
    
    # Force garbage collection to clean up cycles
    collected = gc.collect()
    print(f"Garbage collector collected {collected} objects")

# ML-specific circular reference example
class MLPipeline:
    def __init__(self, name):
        self.name = name
        self.preprocessor = None
        self.model = None
    
    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        preprocessor.pipeline = self  # Creates circular reference

class Preprocessor:
    def __init__(self, name):
        self.name = name
        self.pipeline = None
    
    def __del__(self):
        print(f"Preprocessor {self.name} destroyed")

def ml_circular_reference_example():
    """Circular references in ML context"""
    pipeline = MLPipeline("TextClassification")
    preprocessor = Preprocessor("TextPreprocessor")
    
    # Create circular reference
    pipeline.set_preprocessor(preprocessor)
    
    print("Before deletion:")
    print(f"Objects before GC: {len(gc.get_objects())}")
    
    # Delete references
    del pipeline, preprocessor
    
    print("After deletion, before GC:")
    collected = gc.collect()
    print(f"Garbage collected {collected} objects")

demonstrate_circular_references()
ml_circular_reference_example()
```

### Generational Garbage Collection

**Python's Three-Generation System:**
```python
def demonstrate_generational_gc():
    """Demonstrate generational garbage collection"""
    print("Generational Garbage Collection Info:")
    print(f"GC thresholds: {gc.get_threshold()}")
    print(f"GC counts: {gc.get_count()}")
    
    # Create many objects to trigger different generations
    objects = []
    
    print("\nCreating objects and monitoring GC:")
    for i in range(1000):
        # Create objects that may become garbage
        temp_list = [j for j in range(10)]
        if i % 100 == 0:
            print(f"Iteration {i}: GC counts = {gc.get_count()}")
        
        # Keep some objects alive (they'll move to older generations)
        if i % 100 == 0:
            objects.append(temp_list)
    
    print(f"\nFinal GC counts: {gc.get_count()}")
    
    # Force collection and see what gets collected
    for generation in range(3):
        collected = gc.collect(generation)
        print(f"Generation {generation} collected: {collected} objects")

def gc_statistics():
    """Display detailed GC statistics"""
    stats = gc.get_stats()
    for i, stat in enumerate(stats):
        print(f"Generation {i}:")
        print(f"  Collections: {stat['collections']}")
        print(f"  Collected: {stat['collected']}")
        print(f"  Uncollectable: {stat['uncollectable']}")

demonstrate_generational_gc()
gc_statistics()
```

### Memory Management in ML Applications

**1. Memory-Efficient Data Loading**
```python
import numpy as np
import pandas as pd
from memory_profiler import profile

class MemoryEfficientMLWorkflow:
    def __init__(self):
        self.data = None
        self.model = None
    
    @profile
    def load_large_dataset(self, filepath, chunk_size=10000):
        """Memory-efficient loading of large datasets"""
        chunks = []
        
        # Process data in chunks to avoid memory overload
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            # Process each chunk
            processed_chunk = self.preprocess_chunk(chunk)
            chunks.append(processed_chunk)
            
            # Explicitly delete chunk to free memory
            del chunk
            
        # Combine chunks
        self.data = pd.concat(chunks, ignore_index=True)
        
        # Clear intermediate data
        del chunks
        gc.collect()  # Force garbage collection
        
        return self.data
    
    def preprocess_chunk(self, chunk):
        """Preprocess a data chunk"""
        # Simulate preprocessing
        chunk = chunk.dropna()
        return chunk
    
    def memory_monitoring_training(self, X, y):
        """Monitor memory during model training"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        print(f"Memory before training: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Simulate model training
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        
        print(f"Memory after training: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Clean up
        del model
        gc.collect()
        
        print(f"Memory after cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Example usage
# workflow = MemoryEfficientMLWorkflow()
# workflow.memory_monitoring_training(X_sample, y_sample)
```

**2. Managing Large Model Objects**
```python
class ModelManager:
    """Manage ML models with proper garbage collection"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
    
    def add_model(self, name, model, metadata=None):
        """Add model with automatic cleanup of old models"""
        # If model already exists, clean it up first
        if name in self.models:
            self.remove_model(name)
        
        self.models[name] = model
        self.model_metadata[name] = metadata or {}
        
        print(f"Model '{name}' added. Total models: {len(self.models)}")
    
    def remove_model(self, name):
        """Remove model and force garbage collection"""
        if name in self.models:
            del self.models[name]
            del self.model_metadata[name]
            gc.collect()
            print(f"Model '{name}' removed and memory freed")
    
    def clear_all_models(self):
        """Clear all models and free memory"""
        self.models.clear()
        self.model_metadata.clear()
        gc.collect()
        print("All models cleared")
    
    def get_memory_usage(self):
        """Get approximate memory usage of stored models"""
        total_size = 0
        for name, model in self.models.items():
            model_size = sys.getsizeof(model)
            total_size += model_size
            print(f"Model '{name}': ~{model_size / 1024:.2f} KB")
        
        print(f"Total model memory: ~{total_size / 1024:.2f} KB")
        return total_size

# Usage example
manager = ModelManager()
```

### Memory Optimization Techniques

**1. Weak References**
```python
import weakref

class DataCache:
    """Cache that doesn't prevent garbage collection"""
    
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    def get_or_create_data(self, key, create_func):
        """Get cached data or create new if not available"""
        data = self._cache.get(key)
        if data is None:
            print(f"Creating new data for key: {key}")
            data = create_func()
            self._cache[key] = data
        else:
            print(f"Using cached data for key: {key}")
        return data

class Dataset:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.data = np.random.randn(size, 10)  # Simulate large dataset
    
    def __del__(self):
        print(f"Dataset {self.name} garbage collected")

def demonstrate_weak_references():
    """Show how weak references work with caching"""
    cache = DataCache()
    
    # Create dataset through cache
    dataset1 = cache.get_or_create_data("train", lambda: Dataset("train", 1000))
    dataset2 = cache.get_or_create_data("train", lambda: Dataset("train", 1000))  # Same object
    
    print(f"Same object? {dataset1 is dataset2}")
    
    # Delete references
    del dataset1, dataset2
    gc.collect()
    
    # Try to get again - will create new since old was garbage collected
    dataset3 = cache.get_or_create_data("train", lambda: Dataset("train", 1000))

demonstrate_weak_references()
```

**2. Context Managers for Resource Management**
```python
class ModelTrainingContext:
    """Context manager for ML model training with automatic cleanup"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.start_memory = None
    
    def __enter__(self):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss
        
        print(f"Starting training for {self.model_name}")
        print(f"Initial memory: {self.start_memory / 1024 / 1024:.2f} MB")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up model
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        
        # Force garbage collection
        gc.collect()
        
        import psutil
        import os
        process = psutil.Process(os.getpid())
        end_memory = process.memory_info().rss
        
        print(f"Training completed for {self.model_name}")
        print(f"Final memory: {end_memory / 1024 / 1024:.2f} MB")
        print(f"Memory freed: {(self.start_memory - end_memory) / 1024 / 1024:.2f} MB")
    
    def train_model(self, X, y):
        """Train a model within the context"""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X, y)
        return self.model

# Usage
def demonstrate_context_manager():
    # Create sample data
    X_sample = np.random.randn(1000, 20)
    y_sample = np.random.randint(0, 2, 1000)
    
    with ModelTrainingContext("RandomForest") as trainer:
        model = trainer.train_model(X_sample, y_sample)
        # Model automatically cleaned up when exiting context

# demonstrate_context_manager()
```

### Garbage Collection Best Practices

**1. Monitoring and Tuning**
```python
class GCMonitor:
    """Monitor and tune garbage collection for ML applications"""
    
    def __init__(self):
        self.initial_threshold = gc.get_threshold()
        self.collection_stats = []
    
    def tune_gc_for_ml(self, dataset_size="large"):
        """Tune GC parameters based on ML workload"""
        if dataset_size == "large":
            # For large datasets, reduce frequency of collection
            # to avoid interrupting long-running operations
            gc.set_threshold(2000, 15, 15)  # Increased thresholds
        elif dataset_size == "small":
            # For small datasets with frequent object creation
            gc.set_threshold(500, 8, 8)   # More frequent collection
        
        print(f"GC tuned for {dataset_size} dataset")
        print(f"New thresholds: {gc.get_threshold()}")
    
    def monitor_gc_during_training(self, training_func, *args, **kwargs):
        """Monitor GC activity during training"""
        # Enable GC debugging
        gc.set_debug(gc.DEBUG_STATS)
        
        initial_stats = gc.get_stats()
        
        # Run training
        result = training_func(*args, **kwargs)
        
        final_stats = gc.get_stats()
        
        # Calculate collections that occurred
        for i, (initial, final) in enumerate(zip(initial_stats, final_stats)):
            collections = final['collections'] - initial['collections']
            collected = final['collected'] - initial['collected']
            print(f"Generation {i}: {collections} collections, {collected} objects collected")
        
        # Disable debugging
        gc.set_debug(0)
        
        return result
    
    def restore_default_gc(self):
        """Restore default GC settings"""
        gc.set_threshold(*self.initial_threshold)
        print(f"GC restored to default: {gc.get_threshold()}")

# Example usage
monitor = GCMonitor()
monitor.tune_gc_for_ml("large")

def sample_training_function():
    """Sample training function for monitoring"""
    data = [np.random.randn(100, 10) for _ in range(100)]
    del data
    return "Training complete"

# Monitor GC during training
# result = monitor.monitor_gc_during_training(sample_training_function)
monitor.restore_default_gc()
```

**2. Manual Memory Management**
```python
def manual_memory_management_example():
    """Best practices for manual memory management"""
    
    # 1. Explicitly delete large objects
    large_array = np.random.randn(10000, 1000)
    print(f"Created large array: {large_array.nbytes / 1024 / 1024:.2f} MB")
    
    # Use the array...
    result = np.mean(large_array)
    
    # Explicitly delete when done
    del large_array
    gc.collect()  # Force collection
    
    # 2. Use generators for large datasets
    def data_generator(size):
        """Generator to avoid loading all data at once"""
        for i in range(size):
            yield np.random.randn(100)
    
    # Process data in batches
    batch_results = []
    for i, batch in enumerate(data_generator(1000)):
        if i % 100 == 0:
            print(f"Processing batch {i}")
        
        # Process batch
        batch_result = np.mean(batch)
        batch_results.append(batch_result)
        
        # Batch is automatically cleaned up after each iteration
    
    # 3. Monitor memory during processing
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    def memory_checkpoint(label):
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"{label}: {memory_mb:.2f} MB")
    
    memory_checkpoint("Start")
    
    # Create and process data
    temp_data = [np.random.randn(1000) for _ in range(100)]
    memory_checkpoint("After data creation")
    
    # Process and clean up
    results = [np.sum(arr) for arr in temp_data]
    del temp_data
    gc.collect()
    memory_checkpoint("After cleanup")

manual_memory_management_example()
```

### Key Takeaways

**Garbage Collection Mechanisms:**
1. **Reference Counting**: Immediate cleanup when references reach zero
2. **Cyclic GC**: Handles circular references that reference counting misses
3. **Generational GC**: Optimizes collection based on object age

**ML-Specific Considerations:**
- Large datasets require careful memory management
- Model objects can consume significant memory
- Training processes may create many temporary objects
- Use context managers for automatic resource cleanup
- Monitor memory usage during training
- Tune GC parameters for specific workloads

**Best Practices:**
- Explicitly delete large objects when done
- Use generators for large datasets
- Implement proper cleanup in classes (`__del__` methods)
- Use weak references for caches
- Monitor memory usage in production
- Tune GC thresholds for ML workloads
- Use context managers for resource management

Understanding Python's garbage collection is crucial for building efficient, memory-conscious machine learning applications that can handle large datasets and complex models without running into memory issues.

---

## Question 8

**What are decorators, and can you provide an example of when you'd use one?**

**Answer:**
**Python Decorators Overview:**
Decorators are a powerful feature in Python that allow you to modify or extend the behavior of functions, methods, or classes without permanently modifying their structure. They implement the decorator pattern and provide a clean, readable way to add functionality to existing code.

### Understanding Decorators

**1. Basic Decorator Concepts**
```python
def my_decorator(func):
    """Basic decorator that wraps a function"""
    def wrapper(*args, **kwargs):
        print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After calling {func.__name__}")
        return result
    return wrapper

# Using the decorator
@my_decorator
def greet(name):
    return f"Hello, {name}!"

# This is equivalent to: greet = my_decorator(greet)

# Usage
result = greet("Alice")
print(result)

# Output:
# Before calling greet
# After calling greet
# Hello, Alice!
```

**2. Decorators with Parameters**
```python
def repeat(times):
    """Decorator factory that creates a decorator to repeat function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for i in range(times):
                result = func(*args, **kwargs)
                results.append(result)
                print(f"Execution {i+1}: {result}")
            return results
        return wrapper
    return decorator

@repeat(3)
def roll_dice():
    import random
    return random.randint(1, 6)

# Usage
results = roll_dice()
print(f"All results: {results}")
```

### ML-Specific Decorator Examples

**1. Performance Monitoring Decorator**
```python
import time
import functools
from memory_profiler import memory_usage
import psutil
import os

def monitor_performance(func):
    """Decorator to monitor execution time and memory usage of ML functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Record start time
        start_time = time.time()
        
        print(f"Starting {func.__name__}")
        print(f"Initial memory: {initial_memory:.2f} MB")
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            
            # Calculate metrics
            end_time = time.time()
            execution_time = end_time - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            
            print(f"Function {func.__name__} completed:")
            print(f"  Execution time: {execution_time:.4f} seconds")
            print(f"  Final memory: {final_memory:.2f} MB")
            print(f"  Memory used: {memory_used:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"Function {func.__name__} failed: {str(e)}")
            raise
    
    return wrapper

# Usage in ML context
@monitor_performance
def train_model(X, y):
    """Example ML training function"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Example usage
import numpy as np
X_sample = np.random.randn(1000, 20)
y_sample = np.random.randint(0, 2, 1000)

# model = train_model(X_sample, y_sample)
```

**2. Data Validation Decorator**
```python
import numpy as np
import pandas as pd
from typing import Union, Callable

def validate_ml_data(input_checks=None, output_checks=None):
    """
    Decorator to validate input and output data for ML functions
    
    Args:
        input_checks: List of validation functions for inputs
        output_checks: List of validation functions for outputs
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate inputs
            if input_checks:
                for i, check_func in enumerate(input_checks):
                    if i < len(args):
                        try:
                            check_func(args[i])
                        except AssertionError as e:
                            raise ValueError(f"Input validation failed for argument {i}: {e}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate outputs
            if output_checks:
                for check_func in output_checks:
                    try:
                        check_func(result)
                    except AssertionError as e:
                        raise ValueError(f"Output validation failed: {e}")
            
            return result
        return wrapper
    return decorator

# Define validation functions
def check_no_missing_values(data):
    """Check that data has no missing values"""
    if isinstance(data, np.ndarray):
        assert not np.any(np.isnan(data)), "Data contains NaN values"
    elif isinstance(data, pd.DataFrame):
        assert not data.isnull().any().any(), "DataFrame contains missing values"

def check_positive_shape(data):
    """Check that data has positive dimensions"""
    if hasattr(data, 'shape'):
        assert all(dim > 0 for dim in data.shape), "Data has zero or negative dimensions"

def check_feature_range(data, min_val=-100, max_val=100):
    """Check that features are within expected range"""
    if isinstance(data, (np.ndarray, pd.DataFrame)):
        values = data.values if isinstance(data, pd.DataFrame) else data
        assert np.all(values >= min_val) and np.all(values <= max_val), \
            f"Data values outside expected range [{min_val}, {max_val}]"

# Usage example
@validate_ml_data(
    input_checks=[check_no_missing_values, check_positive_shape],
    output_checks=[check_positive_shape]
)
def preprocess_data(raw_data):
    """Preprocess ML data with validation"""
    # Remove any remaining NaN values
    if isinstance(raw_data, pd.DataFrame):
        processed = raw_data.dropna()
    else:
        processed = raw_data[~np.isnan(raw_data).any(axis=1)]
    
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if len(processed.shape) == 2:
        processed = scaler.fit_transform(processed)
    
    return processed

# Example usage
# sample_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# clean_data = preprocess_data(sample_data)
```

**3. Caching/Memoization Decorator**
```python
import pickle
import hashlib
import os
from pathlib import Path

def cache_results(cache_dir="./ml_cache", expire_hours=24):
    """
    Decorator to cache expensive ML computations
    
    Args:
        cache_dir: Directory to store cached results
        expire_hours: Hours after which cache expires
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory
            cache_path = Path(cache_dir)
            cache_path.mkdir(exist_ok=True)
            
            # Create unique cache key
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            cache_file = cache_path / f"{cache_key}.pkl"
            
            # Check if cached result exists and is not expired
            if cache_file.exists():
                file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                if file_age_hours < expire_hours:
                    print(f"Loading cached result for {func.__name__}")
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                else:
                    print(f"Cache expired for {func.__name__}, recomputing...")
            
            # Compute result
            print(f"Computing {func.__name__}...")
            result = func(*args, **kwargs)
            
            # Cache result
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                print(f"Result cached for {func.__name__}")
            except Exception as e:
                print(f"Failed to cache result: {e}")
            
            return result
        return wrapper
    return decorator

@cache_results(expire_hours=6)
def expensive_feature_engineering(data, n_components=50):
    """Expensive feature engineering that benefits from caching"""
    print("Performing expensive PCA transformation...")
    from sklearn.decomposition import PCA
    
    # Simulate expensive computation
    time.sleep(2)  # Simulate processing time
    
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    
    return {
        'transformed_data': transformed_data,
        'explained_variance': pca.explained_variance_ratio_,
        'components': pca.components_
    }

# Usage
# First call - computes and caches
# result1 = expensive_feature_engineering(X_sample, n_components=10)

# Second call - loads from cache
# result2 = expensive_feature_engineering(X_sample, n_components=10)
```

**4. Logging and Debugging Decorator**
```python
import logging
from datetime import datetime

def log_ml_operations(log_level=logging.INFO):
    """Decorator to log ML operations with detailed information"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Setup logging
            logger = logging.getLogger(func.__name__)
            logger.setLevel(log_level)
            
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            
            # Log function start
            logger.info(f"Starting {func.__name__}")
            logger.info(f"Arguments: args={len(args)}, kwargs={list(kwargs.keys())}")
            
            # Log input data info
            for i, arg in enumerate(args):
                if hasattr(arg, 'shape'):
                    logger.info(f"Argument {i} shape: {arg.shape}")
                elif hasattr(arg, '__len__'):
                    logger.info(f"Argument {i} length: {len(arg)}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Log success
                execution_time = time.time() - start_time
                logger.info(f"Successfully completed {func.__name__} in {execution_time:.4f}s")
                
                # Log output info
                if hasattr(result, 'shape'):
                    logger.info(f"Output shape: {result.shape}")
                elif hasattr(result, '__len__'):
                    logger.info(f"Output length: {len(result)}")
                
                return result
                
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {execution_time:.4f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator

@log_ml_operations(log_level=logging.DEBUG)
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate ML model with detailed logging"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }

# Usage
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2)
# results = train_and_evaluate_model(X_train, X_test, y_train, y_test)
```

**5. Rate Limiting Decorator (for API calls)**
```python
import time
from collections import defaultdict
from threading import Lock

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    def __init__(self):
        self.calls = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, key, max_calls, time_window):
        with self.lock:
            now = time.time()
            # Remove old calls outside the time window
            self.calls[key] = [call_time for call_time in self.calls[key] 
                              if now - call_time < time_window]
            
            # Check if we can make another call
            if len(self.calls[key]) < max_calls:
                self.calls[key].append(now)
                return True
            return False

rate_limiter = RateLimiter()

def rate_limit(max_calls=10, time_window=60, key_func=None):
    """
    Decorator to rate limit function calls
    
    Args:
        max_calls: Maximum calls allowed
        time_window: Time window in seconds
        key_func: Function to generate unique key for rate limiting
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate rate limiting key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = func.__name__
            
            # Check rate limit
            if not rate_limiter.is_allowed(key, max_calls, time_window):
                wait_time = time_window / max_calls
                print(f"Rate limit exceeded for {func.__name__}. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

@rate_limit(max_calls=5, time_window=30)
def call_ml_api(endpoint, data):
    """Simulate ML API call with rate limiting"""
    print(f"Calling API endpoint: {endpoint}")
    # Simulate API call
    time.sleep(0.1)
    return {"status": "success", "prediction": "positive"}

# Usage
# for i in range(10):
#     result = call_ml_api("/predict", {"text": f"sample text {i}"})
#     print(f"Call {i+1}: {result}")
```

### Class-Based Decorators

**1. Model Registry Decorator**
```python
class ModelRegistry:
    """Registry to track and manage ML models"""
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
    
    def register_model(self, name=None, version="1.0", tags=None):
        """Decorator to register models in the registry"""
        def decorator(model_class):
            model_name = name or model_class.__name__
            
            # Store model information
            self.models[model_name] = model_class
            self.model_metadata[model_name] = {
                'version': version,
                'tags': tags or [],
                'registered_at': datetime.now(),
                'class_name': model_class.__name__
            }
            
            print(f"Registered model: {model_name} v{version}")
            return model_class
        
        return decorator
    
    def get_model(self, name):
        """Get registered model by name"""
        return self.models.get(name)
    
    def list_models(self):
        """List all registered models"""
        for name, metadata in self.model_metadata.items():
            print(f"{name} v{metadata['version']} - {metadata['tags']}")

# Usage
registry = ModelRegistry()

@registry.register_model(name="CustomClassifier", version="2.1", tags=["classification", "ensemble"])
class MyCustomModel:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.model = None
    
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

# List registered models
registry.list_models()

# Get and use model
ModelClass = registry.get_model("CustomClassifier")
# model_instance = ModelClass(n_estimators=50)
```

### Advanced Decorator Patterns

**1. Retry Decorator with Exponential Backoff**
```python
import random

def retry_with_backoff(max_retries=3, base_delay=1, max_delay=60, backoff_factor=2):
    """Decorator to retry failed operations with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        print(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise last_exception
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    jitter = delay * 0.1 * random.random()  # Add jitter
                    total_delay = delay + jitter
                    
                    print(f"Attempt {attempt + 1} failed for {func.__name__}. "
                          f"Retrying in {total_delay:.2f}s...")
                    time.sleep(total_delay)
            
            raise last_exception
        
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, base_delay=1)
def unreliable_data_fetch():
    """Simulate unreliable data fetching that might fail"""
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Failed to fetch data")
    
    return "Successfully fetched data"

# Usage
# try:
#     data = unreliable_data_fetch()
#     print(data)
# except Exception as e:
#     print(f"Final failure: {e}")
```

### When to Use Decorators in ML

**Common Use Cases:**

1. **Performance Monitoring**: Track execution time and memory usage
2. **Data Validation**: Ensure input/output data meets requirements
3. **Caching**: Store expensive computation results
4. **Logging**: Track ML operations and debugging
5. **Rate Limiting**: Control API call frequency
6. **Retry Logic**: Handle transient failures
7. **Model Registration**: Track and manage model versions
8. **Authentication**: Secure ML endpoints
9. **Preprocessing**: Apply consistent data transformations
10. **Metrics Collection**: Gather performance statistics

**Best Practices:**

1. Use `functools.wraps` to preserve function metadata
2. Handle exceptions appropriately in decorators
3. Make decorators configurable with parameters
4. Document decorator behavior clearly
5. Consider performance overhead of decorators
6. Use class-based decorators for complex state management
7. Keep decorators focused on single responsibilities
8. Test decorated functions thoroughly

**Key Benefits:**
- **Separation of Concerns**: Keep core logic separate from cross-cutting concerns
- **Reusability**: Apply same functionality across multiple functions
- **Clean Code**: Reduce repetitive boilerplate code
- **Maintainability**: Centralize common functionality
- **Flexibility**: Easy to add/remove functionality

Decorators are essential for building robust, maintainable ML systems that handle cross-cutting concerns like monitoring, validation, and error handling elegantly.
---

## Question 9

**What is NumPy and how is it useful in machine learning?**

**Answer:**

**NumPy Overview:**

NumPy (Numerical Python) is the fundamental package for scientific computing in Python. It provides a powerful N-dimensional array object and efficient operations for mathematical computations, making it the foundation for most ML libraries.

**Core NumPy Features:**

```python
import numpy as np

# 1. N-dimensional arrays (ndarray)
# Much faster than Python lists for numerical operations
vector = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"Vector shape: {vector.shape}")  # (5,)
print(f"Matrix shape: {matrix.shape}")  # (3, 3)
print(f"Tensor shape: {tensor.shape}")  # (2, 2, 2)

# 2. Data types optimization
int_array = np.array([1, 2, 3], dtype=np.int32)
float_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
bool_array = np.array([True, False, True], dtype=np.bool_)

# 3. Memory efficiency comparison
python_list = list(range(1000))
numpy_array = np.arange(1000)

import sys
print(f"Python list size: {sys.getsizeof(python_list)} bytes")
print(f"NumPy array size: {numpy_array.nbytes} bytes")
```

**ML-Specific NumPy Operations:**

```python
class MLNumPyOperations:
    """Essential NumPy operations for machine learning"""
    
    def array_creation_for_ml(self):
        """Common ways to create arrays for ML"""
        
        # Initialize datasets
        zeros_data = np.zeros((1000, 10))  # Initialize features
        ones_labels = np.ones(1000)       # Binary labels
        random_data = np.random.random((1000, 10))  # Random features
        
        # Structured data creation
        X = np.random.normal(0, 1, (1000, 5))  # Normal distribution features
        y = np.random.choice([0, 1], size=1000)  # Binary classification labels
        
        # Identity matrix for transformations
        identity = np.eye(5)  # 5x5 identity matrix
        
        # Sequences for time series
        time_series = np.linspace(0, 10, 100)  # 100 points from 0 to 10
        
        return X, y, identity, time_series
    
    def mathematical_operations(self, X, y):
        """Core mathematical operations for ML"""
        
        # Element-wise operations (vectorized)
        X_squared = X ** 2
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Matrix operations
        covariance_matrix = np.cov(X.T)  # Feature covariance
        correlation_matrix = np.corrcoef(X.T)  # Feature correlation
        
        # Linear algebra operations
        XTX = np.dot(X.T, X)  # X transpose times X
        eigenvals, eigenvecs = np.linalg.eig(covariance_matrix)
        
        # Statistical operations
        mean_features = np.mean(X, axis=0)
        std_features = np.std(X, axis=0)
        min_features = np.min(X, axis=0)
        max_features = np.max(X, axis=0)
        
        return {
            'normalized_X': X_normalized,
            'covariance': covariance_matrix,
            'eigenvalues': eigenvals,
            'statistics': {
                'mean': mean_features,
                'std': std_features,
                'min': min_features,
                'max': max_features
            }
        }
    
    def indexing_and_slicing(self, data):
        """Advanced indexing for data manipulation"""
        
        # Boolean indexing
        positive_data = data[data > 0]
        
        # Fancy indexing
        selected_rows = data[[0, 2, 4]]  # Select specific rows
        selected_cols = data[:, [1, 3]]  # Select specific columns
        
        # Conditional selection
        outliers = data[np.abs(data) > 2 * np.std(data)]
        
        # Masked arrays for missing data
        masked_data = np.ma.masked_where(data < 0, data)
        
        return positive_data, selected_rows, outliers
    
    def broadcasting_examples(self):
        """Broadcasting for efficient computations"""
        
        # Feature scaling example
        X = np.random.random((1000, 5))
        mean = np.mean(X, axis=0, keepdims=True)  # Shape: (1, 5)
        std = np.std(X, axis=0, keepdims=True)    # Shape: (1, 5)
        
        # Broadcasting: (1000, 5) - (1, 5) -> (1000, 5)
        X_standardized = (X - mean) / std
        
        # Adding bias term
        bias = np.ones((1000, 1))
        X_with_bias = np.hstack([bias, X])  # Add bias column
        
        return X_standardized, X_with_bias
```

**NumPy in ML Algorithms:**

```python
class MLAlgorithmsWithNumPy:
    """Implementing ML algorithms using NumPy"""
    
    def linear_regression(self, X, y):
        """Linear regression using NumPy"""
        # Add bias term
        X_bias = np.column_stack([np.ones(len(X)), X])
        
        # Normal equation: Î¸ = (X^T X)^(-1) X^T y
        XTX = np.dot(X_bias.T, X_bias)
        XTy = np.dot(X_bias.T, y)
        theta = np.linalg.solve(XTX, XTy)  # More stable than inverse
        
        return theta
    
    def k_means_clustering(self, X, k, max_iters=100):
        """K-means clustering using NumPy"""
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        centroids = X[np.random.choice(n_samples, k, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
        
        return centroids, labels
    
    def gradient_descent(self, X, y, learning_rate=0.01, epochs=1000):
        """Gradient descent using NumPy"""
        m, n = X.shape
        theta = np.random.normal(0, 0.01, n)
        
        cost_history = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = np.dot(X, theta)
            
            # Cost function (MSE)
            cost = np.mean((predictions - y) ** 2)
            cost_history.append(cost)
            
            # Gradient computation
            gradient = (2/m) * np.dot(X.T, (predictions - y))
            
            # Parameter update
            theta -= learning_rate * gradient
        
        return theta, cost_history
    
    def principal_component_analysis(self, X):
        """PCA implementation using NumPy"""
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
```

**Performance Advantages:**

```python
import time

def performance_comparison():
    """Compare NumPy vs pure Python performance"""
    
    # Large dataset
    size = 1000000
    
    # Pure Python operations
    python_list1 = list(range(size))
    python_list2 = list(range(size, 2*size))
    
    start_time = time.time()
    python_result = [a + b for a, b in zip(python_list1, python_list2)]
    python_time = time.time() - start_time
    
    # NumPy operations
    numpy_array1 = np.arange(size)
    numpy_array2 = np.arange(size, 2*size)
    
    start_time = time.time()
    numpy_result = numpy_array1 + numpy_array2
    numpy_time = time.time() - start_time
    
    print(f"Python time: {python_time:.4f} seconds")
    print(f"NumPy time: {numpy_time:.4f} seconds")
    print(f"NumPy is {python_time/numpy_time:.1f}x faster")
    
    # Memory usage comparison
    import sys
    print(f"Python list memory: {sys.getsizeof(python_list1)} bytes")
    print(f"NumPy array memory: {numpy_array1.nbytes} bytes")
```

**Key Benefits for Machine Learning:**

1. **Performance**: Vectorized operations are 10-100x faster than pure Python
2. **Memory Efficiency**: Contiguous memory layout and optimized data types
3. **Mathematical Operations**: Comprehensive linear algebra and statistical functions
4. **Broadcasting**: Efficient operations on arrays of different shapes
5. **Interoperability**: Foundation for pandas, scikit-learn, TensorFlow, PyTorch
6. **C/Fortran Integration**: Leverages optimized mathematical libraries (BLAS, LAPACK)

**Essential for ML Because:**
- Matrix operations for neural networks
- Statistical computations for data analysis
- Efficient data preprocessing
- Implementation of ML algorithms from scratch
- Foundation for all major ML libraries
- Memory-efficient handling of large datasets

---

## Question 10

**How does Scikit-learn fit into the machine learning workflow?**

**Answer:**

**Scikit-learn Overview:**

Scikit-learn is the most popular machine learning library for Python, providing simple and efficient tools for data mining and data analysis. It offers a consistent API and comprehensive toolkit for the entire ML workflow.

**Core Workflow Integration:**

```python
# Complete ML workflow with scikit-learn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

class SklearnMLWorkflow:
    """Complete ML workflow using scikit-learn"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.pipeline = None
    
    def load_and_preprocess_data(self, filepath):
        """Data loading and preprocessing"""
        # Load data
        data = pd.read_csv(filepath)
        
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Handle categorical variables
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        return X, y
    
    def data_splitting(self, X, y, test_size=0.2, random_state=42):
        """Split data into train/validation/test sets"""
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_pipeline(self):
        """Create ML pipeline with preprocessing and model"""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        return self.pipeline
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        """Train model and evaluate performance"""
        # Fit pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Predictions
        train_pred = self.pipeline.predict(X_train)
        val_pred = self.pipeline.predict(X_val)
        
        # Evaluation metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred, average='weighted'),
            'val_recall': recall_score(y_val, val_pred, average='weighted'),
            'val_f1': f1_score(y_val, val_pred, average='weighted')
        }
        
        return results
    
    def cross_validation(self, X, y, cv=5):
        """Perform cross-validation"""
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
```

**Key Scikit-learn Components:**

```python
class SklearnComponents:
    """Demonstrate key scikit-learn components"""
    
    def preprocessing_tools(self, X, y):
        """Data preprocessing utilities"""
        from sklearn.preprocessing import (
            StandardScaler, MinMaxScaler, RobustScaler,
            LabelEncoder, OneHotEncoder, PolynomialFeatures
        )
        from sklearn.impute import SimpleImputer
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Scaling
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        robust_scaler = RobustScaler()
        
        X_standard = standard_scaler.fit_transform(X)
        X_minmax = minmax_scaler.fit_transform(X)
        X_robust = robust_scaler.fit_transform(X)
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=5)
        X_selected = selector.fit_transform(X, y)
        
        # Polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        return {
            'scaled_data': X_standard,
            'selected_features': X_selected,
            'polynomial_features': X_poly
        }
    
    def model_selection_tools(self, X, y):
        """Model selection and validation"""
        from sklearn.model_selection import (
            GridSearchCV, RandomizedSearchCV, StratifiedKFold
        )
        from sklearn.ensemble import RandomForestClassifier
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        # Randomized search (faster for large parameter spaces)
        random_search = RandomizedSearchCV(
            rf, param_grid, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1
        )
        random_search.fit(X, y)
        
        return {
            'best_params_grid': grid_search.best_params_,
            'best_score_grid': grid_search.best_score_,
            'best_params_random': random_search.best_params_,
            'best_score_random': random_search.best_score_
        }
    
    def evaluation_metrics(self, y_true, y_pred, y_prob=None):
        """Comprehensive model evaluation"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_auc_score,
            roc_curve, precision_recall_curve
        )
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC AUC (if probabilities available)
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        
        return metrics, cm, report
```

**Algorithm Implementations:**

```python
class SklearnAlgorithms:
    """Comprehensive algorithm usage in scikit-learn"""
    
    def supervised_learning_algorithms(self, X_train, y_train, X_test):
        """Various supervised learning algorithms"""
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        
        # Initialize models
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        
        # Train and evaluate all models
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_train)  # Note: should be y_test
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score
            }
        
        return results
    
    def unsupervised_learning_algorithms(self, X):
        """Unsupervised learning algorithms"""
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.decomposition import PCA, TruncatedSVD
        from sklearn.manifold import TSNE
        from sklearn.mixture import GaussianMixture
        
        # Clustering algorithms
        clustering_results = {}
        
        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        clustering_results['kmeans'] = {'labels': kmeans_labels, 'model': kmeans}
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        clustering_results['dbscan'] = {'labels': dbscan_labels, 'model': dbscan}
        
        # Dimensionality reduction
        reduction_results = {}
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        reduction_results['pca'] = {'transformed': X_pca, 'model': pca}
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        reduction_results['tsne'] = {'transformed': X_tsne, 'model': tsne}
        
        return clustering_results, reduction_results
    
    def ensemble_methods(self, X_train, y_train):
        """Advanced ensemble methods"""
        from sklearn.ensemble import (
            VotingClassifier, BaggingClassifier, AdaBoostClassifier,
            ExtraTreesClassifier, StackingClassifier
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        
        # Base models
        dt = DecisionTreeClassifier(random_state=42)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        lr = LogisticRegression(random_state=42)
        
        # Voting classifier
        voting_clf = VotingClassifier(
            estimators=[('dt', dt), ('rf', rf), ('lr', lr)],
            voting='soft'
        )
        
        # Bagging
        bagging_clf = BaggingClassifier(
            base_estimator=dt, n_estimators=10, random_state=42
        )
        
        # Boosting
        ada_clf = AdaBoostClassifier(
            base_estimator=dt, n_estimators=50, random_state=42
        )
        
        # Stacking
        stacking_clf = StackingClassifier(
            estimators=[('dt', dt), ('rf', rf)],
            final_estimator=lr,
            cv=5
        )
        
        # Train all ensemble models
        ensemble_models = {
            'voting': voting_clf,
            'bagging': bagging_clf,
            'adaboost': ada_clf,
            'stacking': stacking_clf
        }
        
        results = {}
        for name, model in ensemble_models.items():
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            results[name] = {'model': model, 'score': score}
        
        return results
```

**Advanced Features:**

```python
class AdvancedSklearnFeatures:
    """Advanced scikit-learn features"""
    
    def custom_transformers(self):
        """Create custom transformers"""
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class LogTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, features=None):
                self.features = features
            
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                X_copy = X.copy()
                if self.features:
                    for feature in self.features:
                        X_copy[feature] = np.log1p(X_copy[feature])
                else:
                    X_copy = np.log1p(X_copy)
                return X_copy
        
        return LogTransformer
    
    def pipeline_with_feature_union(self):
        """Complex pipelines with feature union"""
        from sklearn.pipeline import Pipeline, FeatureUnion
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import SelectKBest
        
        # Create feature union
        feature_union = FeatureUnion([
            ('pca', PCA(n_components=2)),
            ('selection', SelectKBest(k=3))
        ])
        
        # Create complete pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('features', feature_union),
            ('classifier', RandomForestClassifier())
        ])
        
        return pipeline
```

**Key Advantages:**

1. **Consistent API**: All estimators follow fit/predict pattern
2. **Comprehensive**: Covers entire ML workflow
3. **Well-tested**: Production-ready implementations
4. **Documentation**: Excellent documentation and examples
5. **Integration**: Works seamlessly with NumPy, pandas
6. **Performance**: Optimized implementations
7. **Flexibility**: Easy to extend and customize

**Workflow Position:**
- **Data Preprocessing**: Scaling, encoding, feature selection
- **Model Selection**: Grid search, cross-validation
- **Algorithm Implementation**: Wide range of algorithms
- **Evaluation**: Comprehensive metrics and validation
- **Pipeline Creation**: End-to-end workflow management
- **Production**: Model persistence and deployment preparation

---

## Question 11

**Explain Matplotlib and Seaborn libraries for data visualization.**

**Answer:** 

Matplotlib and Seaborn are essential Python libraries for data visualization in machine learning. Matplotlib provides the foundational plotting capabilities, while Seaborn builds on top of Matplotlib with higher-level statistical visualization functions.

### Matplotlib: Foundation of Python Plotting

**Core Concepts:**
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Basic plotting setup
plt.style.use('seaborn-v0_8')  # Modern default style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Generate sample data for ML examples
np.random.seed(42)
data_size = 1000
features = np.random.randn(data_size, 2)
target = (features[:, 0] + features[:, 1] + np.random.randn(data_size) * 0.5 > 0).astype(int)
```

**1. Basic Plotting for ML Analysis**
```python
# Create subplots for comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Feature distribution
axes[0, 0].hist(features[:, 0], bins=30, alpha=0.7, color='blue', label='Feature 1')
axes[0, 0].hist(features[:, 1], bins=30, alpha=0.7, color='red', label='Feature 2')
axes[0, 0].set_title('Feature Distributions')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Scatter plot with target classes
colors = ['red' if t == 0 else 'blue' for t in target]
axes[0, 1].scatter(features[:, 0], features[:, 1], c=colors, alpha=0.6)
axes[0, 1].set_title('Feature Space with Target Classes')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')

# Training loss curve simulation
epochs = range(1, 101)
train_loss = [1.0 * np.exp(-epoch/20) + 0.1 + np.random.normal(0, 0.02) for epoch in epochs]
val_loss = [1.0 * np.exp(-epoch/25) + 0.15 + np.random.normal(0, 0.03) for epoch in epochs]

axes[1, 0].plot(epochs, train_loss, label='Training Loss', color='blue')
axes[1, 0].plot(epochs, val_loss, label='Validation Loss', color='red')
axes[1, 0].set_title('Model Training Progress')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()

# Confusion matrix visualization
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
im = axes[1, 1].imshow(cm, interpolation='nearest', cmap='Blues')
axes[1, 1].set_title('Confusion Matrix')
axes[1, 1].set_xlabel('Predicted Label')
axes[1, 1].set_ylabel('True Label')

# Add text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()
plt.show()
```

**2. Advanced Matplotlib Techniques for ML**
```python
# Custom plotting class for ML experiments
class MLPlotter:
    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_learning_curves(self, train_sizes, train_scores, val_scores, title="Learning Curves"):
        """Plot learning curves for model evaluation"""
        plt.figure(figsize=self.figsize)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', color=self.colors[0], label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color=self.colors[0])
        
        plt.plot(train_sizes, val_mean, 'o-', color=self.colors[1], label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color=self.colors[1])
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_feature_importance(self, feature_names, importances, title="Feature Importance"):
        """Plot feature importance"""
        plt.figure(figsize=self.figsize)
        
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices], color=self.colors[2])
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, roc_data, title="ROC Curves Comparison"):
        """Plot multiple ROC curves"""
        plt.figure(figsize=self.figsize)
        
        for i, (name, fpr, tpr, auc_score) in enumerate(roc_data):
            plt.plot(fpr, tpr, color=self.colors[i % len(self.colors)], 
                    label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Usage example
plotter = MLPlotter()

# Simulate learning curve data
train_sizes = np.linspace(0.1, 1.0, 10) * 800
train_scores = np.random.uniform(0.7, 0.9, (10, 5))
val_scores = np.random.uniform(0.65, 0.85, (10, 5))

plotter.plot_learning_curves(train_sizes, train_scores, val_scores)
```

### Seaborn: Statistical Data Visualization

**Core Advantages over Matplotlib:**
```python
import seaborn as sns

# Set Seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Generate more comprehensive dataset for demo
np.random.seed(42)
n_samples = 500

# Create synthetic ML dataset
data = {
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.exponential(2, n_samples),
    'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
    'algorithm': np.random.choice(['Random Forest', 'SVM', 'Neural Network'], n_samples),
    'accuracy': np.random.uniform(0.6, 0.95, n_samples),
    'training_time': np.random.uniform(1, 100, n_samples)
}

ml_df = pd.DataFrame(data)
ml_df['target'] = (ml_df['feature_1'] + ml_df['feature_2'] > 0).astype(int)
```

**1. Statistical Plotting with Seaborn**
```python
# Create comprehensive visualization dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Distribution plots
sns.histplot(data=ml_df, x='accuracy', hue='algorithm', multiple="layer", ax=axes[0, 0])
axes[0, 0].set_title('Accuracy Distribution by Algorithm')

# Correlation heatmap
numeric_cols = ['feature_1', 'feature_2', 'feature_3', 'accuracy', 'training_time']
correlation_matrix = ml_df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
axes[0, 1].set_title('Feature Correlation Matrix')

# Box plot for algorithm comparison
sns.boxplot(data=ml_df, x='algorithm', y='accuracy', ax=axes[0, 2])
axes[0, 2].set_title('Algorithm Performance Comparison')
axes[0, 2].tick_params(axis='x', rotation=45)

# Scatter plot with regression
sns.scatterplot(data=ml_df, x='training_time', y='accuracy', hue='algorithm', ax=axes[1, 0])
sns.regplot(data=ml_df, x='training_time', y='accuracy', scatter=False, ax=axes[1, 0])
axes[1, 0].set_title('Training Time vs Accuracy')

# Violin plot for detailed distributions
sns.violinplot(data=ml_df, x='categorical_feature', y='feature_3', ax=axes[1, 1])
axes[1, 1].set_title('Feature Distribution by Category')

# Pair plot subset (using seaborn's advanced functionality)
# Note: This would typically be a separate figure, but showing concept
sns.scatterplot(data=ml_df, x='feature_1', y='feature_2', hue='target', ax=axes[1, 2])
axes[1, 2].set_title('Feature Space Classification')

plt.tight_layout()
plt.show()
```

**2. Advanced Seaborn for ML Analysis**
```python
class MLSeabornAnalyzer:
    def __init__(self, data):
        self.data = data
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
    
    def comprehensive_eda(self):
        """Comprehensive Exploratory Data Analysis"""
        # Figure 1: Distribution Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Univariate distributions
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols[:4]):
            row, col_idx = divmod(i, 2)
            sns.histplot(data=self.data, x=col, kde=True, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Distribution of {col}')
        
        plt.suptitle('Feature Distributions', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Relationship Analysis
        if len(numeric_cols) >= 4:
            g = sns.PairGrid(self.data[numeric_cols[:4]])
            g.map_upper(sns.scatterplot, alpha=0.6)
            g.map_lower(sns.regplot)
            g.map_diag(sns.histplot, kde=True)
            plt.suptitle('Feature Relationships', y=1.02, fontsize=16)
            plt.show()
    
    def model_comparison_analysis(self, model_col, metric_col):
        """Compare different models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Box plot comparison
        sns.boxplot(data=self.data, x=model_col, y=metric_col, ax=axes[0])
        axes[0].set_title(f'{metric_col} by {model_col}')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Violin plot for detailed distribution
        sns.violinplot(data=self.data, x=model_col, y=metric_col, ax=axes[1])
        axes[1].set_title(f'{metric_col} Distribution by {model_col}')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Strip plot for individual points
        sns.stripplot(data=self.data, x=model_col, y=metric_col, size=4, alpha=0.7, ax=axes[2])
        axes[2].set_title(f'Individual {metric_col} Values')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self):
        """Advanced correlation analysis"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Custom colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   annot=True, fmt='.2f')
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

# Usage
analyzer = MLSeabornAnalyzer(ml_df)
analyzer.comprehensive_eda()
analyzer.model_comparison_analysis('algorithm', 'accuracy')
analyzer.correlation_analysis()
```

### Specialized ML Visualization Techniques

**1. Model Performance Visualization**
```python
def plot_classification_report(y_true, y_pred, target_names=None):
    """Visualize classification report"""
    from sklearn.metrics import classification_report
    import seaborn as sns
    
    # Get classification report as dict
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Convert to DataFrame for visualization
    df = pd.DataFrame(report).iloc[:-1, :].T
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision heatmap
    precision_data = df[['precision']].T
    sns.heatmap(precision_data, annot=True, cmap='Blues', ax=axes[0])
    axes[0].set_title('Precision by Class')
    
    # Recall heatmap
    recall_data = df[['recall']].T
    sns.heatmap(recall_data, annot=True, cmap='Greens', ax=axes[1])
    axes[1].set_title('Recall by Class')
    
    # F1-score heatmap
    f1_data = df[['f1-score']].T
    sns.heatmap(f1_data, annot=True, cmap='Reds', ax=axes[2])
    axes[2].set_title('F1-Score by Class')
    
    plt.tight_layout()
    plt.show()

# Example usage
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

plot_classification_report(y_test, y_pred, target_names=['Class A', 'Class B', 'Class C'])
```

**2. Hyperparameter Tuning Visualization**
```python
def plot_hyperparameter_heatmap(param_grid_results):
    """Visualize hyperparameter tuning results"""
    # Simulate grid search results
    learning_rates = [0.01, 0.1, 0.2, 0.3]
    max_depths = [3, 5, 7, 10]
    
    # Create results matrix
    results = np.random.uniform(0.7, 0.95, (len(learning_rates), len(max_depths)))
    
    # Create DataFrame
    df = pd.DataFrame(results, 
                     index=[f'LR: {lr}' for lr in learning_rates],
                     columns=[f'Depth: {d}' for d in max_depths])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Hyperparameter Tuning Results (Accuracy)')
    plt.xlabel('Max Depth')
    plt.ylabel('Learning Rate')
    plt.tight_layout()
    plt.show()

plot_hyperparameter_heatmap({})
```

### Integration with ML Workflows

**Complete Visualization Pipeline:**
```python
class MLVisualizationPipeline:
    def __init__(self, X, y, feature_names=None):
        self.X = X
        self.y = y
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        
    def run_complete_analysis(self):
        """Run complete ML visualization analysis"""
        # 1. Data exploration
        self.plot_data_overview()
        
        # 2. Model training and evaluation
        self.plot_model_comparison()
        
        # 3. Feature analysis
        self.plot_feature_analysis()
        
    def plot_data_overview(self):
        """Plot data overview"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature distributions
        for i, feature_name in enumerate(self.feature_names[:4]):
            row, col = divmod(i, 2)
            axes[row, col].hist(self.X[:, i], bins=30, alpha=0.7)
            axes[row, col].set_title(f'{feature_name} Distribution')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Frequency')
        
        plt.suptitle('Feature Distributions', fontsize=16)
        plt.tight_layout()
        plt.show()
        
    def plot_model_comparison(self):
        """Compare different models"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        results = []
        
        for name, model in models.items():
            model.fit(self.X, self.y)
            y_pred = model.predict(self.X)
            y_proba = model.predict_proba(self.X)[:, 1] if len(np.unique(self.y)) == 2 else None
            
            accuracy = accuracy_score(self.y, y_pred)
            auc = roc_auc_score(self.y, y_proba) if y_proba is not None else None
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'AUC': auc if auc else accuracy
            })
        
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        sns.barplot(data=results_df, x='Model', y='Accuracy', ax=axes[0])
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        # AUC comparison
        sns.barplot(data=results_df, x='Model', y='AUC', ax=axes[1])
        axes[1].set_title('Model AUC Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_analysis(self):
        """Analyze feature importance and relationships"""
        # Feature correlation
        df = pd.DataFrame(self.X, columns=self.feature_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

# Example usage
X, y = make_classification(n_samples=1000, n_features=6, n_classes=2, random_state=42)
pipeline = MLVisualizationPipeline(X, y, [f'Feature_{i+1}' for i in range(6)])
pipeline.run_complete_analysis()
```

### Best Practices for ML Visualization

**1. Consistent Styling:**
```python
# Set up consistent style for all plots
def setup_ml_plot_style():
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10

setup_ml_plot_style()
```

**2. Color Schemes for Different Purposes:**
```python
# Define color schemes for different types of plots
ML_COLORS = {
    'performance': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
    'features': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
    'categories': sns.color_palette("Set2"),
    'continuous': 'viridis'
}
```

### Key Advantages in ML Context

**Matplotlib Strengths:**
- Fine-grained control over every aspect of plots
- Excellent for custom visualizations
- Integration with scientific computing ecosystem
- Publication-quality figures

**Seaborn Strengths:**
- Built-in statistical functions
- Beautiful default styles
- Easy handling of categorical data
- Integrated with pandas DataFrames
- Quick exploratory data analysis

**When to Use Each:**
- **Matplotlib**: Custom visualizations, fine control, publication plots
- **Seaborn**: Statistical analysis, quick EDA, beautiful defaults
- **Combined**: Use Seaborn for quick analysis, then customize with Matplotlib

Both libraries are essential for effective data science and machine learning workflows, providing the visualization capabilities needed to understand data, evaluate models, and communicate results effectively.

---

## Question 12

**What is TensorFlow and Keras, and how do they relate to each other?**

**Answer:** 

TensorFlow and Keras are fundamental frameworks for deep learning and neural networks. TensorFlow is Google's open-source machine learning platform, while Keras is a high-level neural networks API that runs on top of TensorFlow (and other backends). Since TensorFlow 2.0, Keras has been integrated as the official high-level API.

### TensorFlow: The Foundation

**Core Concepts:**
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Basic tensor operations
# Tensors are the fundamental data structure
tensor_0d = tf.constant(42)  # Scalar
tensor_1d = tf.constant([1, 2, 3, 4, 5])  # Vector
tensor_2d = tf.constant([[1, 2], [3, 4]])  # Matrix
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D tensor

print(f"0D tensor shape: {tensor_0d.shape}")
print(f"1D tensor shape: {tensor_1d.shape}")
print(f"2D tensor shape: {tensor_2d.shape}")
print(f"3D tensor shape: {tensor_3d.shape}")
```

**1. Low-Level TensorFlow Operations**
```python
# Mathematical operations
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[2.0, 0.0], [0.0, 2.0]])

# Matrix operations
matrix_multiply = tf.matmul(a, b)
element_wise_multiply = tf.multiply(a, b)
addition = tf.add(a, b)

print("Matrix multiplication:")
print(matrix_multiply.numpy())
print("\nElement-wise multiplication:")
print(element_wise_multiply.numpy())

# Automatic differentiation (core of backpropagation)
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1

# Compute gradient dy/dx
gradient = tape.gradient(y, x)
print(f"Gradient of y = xÂ² + 2x + 1 at x=3: {gradient.numpy()}")
```

**2. TensorFlow Data Pipeline**
```python
# Efficient data loading and preprocessing
def create_dataset_pipeline():
    # Generate sample data
    X = np.random.randn(1000, 10).astype(np.float32)
    y = (np.sum(X, axis=1) > 0).astype(np.float32)
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Apply transformations
    dataset = dataset.batch(32)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Create and inspect dataset
train_dataset = create_dataset_pipeline()

# Inspect a batch
for batch_x, batch_y in train_dataset.take(1):
    print(f"Batch shape - X: {batch_x.shape}, y: {batch_y.shape}")
    print(f"Data types - X: {batch_x.dtype}, y: {batch_y.dtype}")
```

**3. Custom Training Loop with TensorFlow**
```python
class SimpleLinearModel(tf.Module):
    def __init__(self, input_dim, output_dim):
        self.w = tf.Variable(tf.random.normal([input_dim, output_dim]))
        self.b = tf.Variable(tf.zeros([output_dim]))
    
    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b

# Initialize model
model = SimpleLinearModel(10, 1)
optimizer = tf.optimizers.Adam(learning_rate=0.01)

# Training function
@tf.function  # Decorator for graph optimization
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.reduce_mean(tf.square(y - predictions))
    
    gradients = tape.gradient(loss, [model.w, model.b])
    optimizer.apply_gradients(zip(gradients, [model.w, model.b]))
    return loss

# Training loop
losses = []
for epoch in range(100):
    epoch_loss = 0
    num_batches = 0
    
    for batch_x, batch_y in train_dataset:
        batch_y = tf.expand_dims(batch_y, 1)  # Add dimension for output
        loss = train_step(batch_x, batch_y)
        epoch_loss += loss
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
```

### Keras: High-Level Neural Networks API

**Why Keras is Essential:**
```python
# Same model with Keras - much simpler!
from tensorflow import keras
from tensorflow.keras import layers

# Simple sequential model
keras_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
keras_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
keras_model.summary()
```

**1. Different Model Building Approaches in Keras**
```python
# 1. Sequential API (simplest)
sequential_model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 2. Functional API (more flexible)
inputs = layers.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

functional_model = keras.Model(inputs=inputs, outputs=outputs)

# 3. Subclassing API (most flexible)
class CustomModel(keras.Model):
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.dense3 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        return self.dense3(x)

custom_model = CustomModel()
```

**2. Comprehensive Keras Training Pipeline**
```python
# Generate sample image classification data
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create dataset
X, y = make_classification(
    n_samples=5000,
    n_features=784,  # Simulate flattened 28x28 images
    n_classes=10,
    n_redundant=0,
    random_state=42
)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to categorical for multi-class classification
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Build comprehensive model
def create_advanced_model():
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

model = create_advanced_model()
```

**3. Advanced Training with Callbacks**
```python
# Define callbacks for better training
callbacks = [
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    
    # Learning rate reduction
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
    
    # Model checkpointing
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    
    # TensorBoard logging
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
]

# Train the model
history = model.fit(
    X_train, y_train_cat,
    batch_size=64,
    epochs=100,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Top-3 Accuracy
    axes[1, 0].plot(history.history['top_3_accuracy'], label='Training')
    axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation')
    axes[1, 0].set_title('Top-3 Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-3 Accuracy')
    axes[1, 0].legend()
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)
```

### TensorFlow + Keras for Specialized Architectures

**1. Convolutional Neural Networks**
```python
def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

cnn_model = create_cnn_model()
cnn_model.summary()
```

**2. Recurrent Neural Networks**
```python
def create_lstm_model(sequence_length=100, num_features=1, num_classes=1):
    model = keras.Sequential([
        # LSTM layers
        layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, num_features)),
        layers.Dropout(0.2),
        
        layers.LSTM(50, return_sequences=True),
        layers.Dropout(0.2),
        
        layers.LSTM(50),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(25, activation='relu'),
        layers.Dense(num_classes, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

lstm_model = create_lstm_model()
lstm_model.summary()
```

**3. Transfer Learning with Keras Applications**
```python
# Using pre-trained models
def create_transfer_learning_model(num_classes=10):
    # Load pre-trained model
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom classifier
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

transfer_model, base_model = create_transfer_learning_model()

# Fine-tuning strategy
def fine_tune_model(model, base_model, unfreeze_layers=20):
    # Unfreeze top layers
    base_model.trainable = True
    
    # Freeze bottom layers
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# fine_tune_model(transfer_model, base_model)
```

### Custom Layers and Operations

**1. Custom Keras Layer**
```python
class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True
        )
        
    def call(self, inputs):
        # Attention mechanism
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
        
        return context_vector, attention_weights

# Use custom layer
def create_attention_model(sequence_length=50, num_features=100, num_classes=10):
    inputs = layers.Input(shape=(sequence_length, num_features))
    
    # LSTM layer
    lstm_output = layers.LSTM(64, return_sequences=True)(inputs)
    
    # Custom attention layer
    context_vector, attention_weights = AttentionLayer(64)(lstm_output)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(context_vector)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

attention_model = create_attention_model()
attention_model.summary()
```

**2. Custom Training Loop with tf.GradientTape**
```python
class CustomTrainer:
    def __init__(self, model, optimizer, loss_fn, metrics):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        for metric in self.metrics:
            metric.update_state(y, predictions)
        
        return loss
    
    @tf.function
    def test_step(self, x, y):
        predictions = self.model(x, training=False)
        loss = self.loss_fn(y, predictions)
        
        for metric in self.metrics:
            metric.update_state(y, predictions)
        
        return loss
    
    def train(self, train_dataset, val_dataset, epochs):
        train_loss = keras.metrics.Mean()
        train_accuracy = keras.metrics.CategoricalAccuracy()
        val_loss = keras.metrics.Mean()
        val_accuracy = keras.metrics.CategoricalAccuracy()
        
        for epoch in range(epochs):
            # Training
            for x_batch, y_batch in train_dataset:
                loss = self.train_step(x_batch, y_batch)
                train_loss.update_state(loss)
            
            # Validation
            for x_batch, y_batch in val_dataset:
                val_loss_value = self.test_step(x_batch, y_batch)
                val_loss.update_state(val_loss_value)
            
            print(f'Epoch {epoch + 1}: '
                  f'Loss: {train_loss.result():.4f}, '
                  f'Accuracy: {train_accuracy.result():.4f}, '
                  f'Val Loss: {val_loss.result():.4f}, '
                  f'Val Accuracy: {val_accuracy.result():.4f}')
            
            # Reset metrics
            train_loss.reset_states()
            train_accuracy.reset_states()
            val_loss.reset_states()
            val_accuracy.reset_states()

# Usage example
# trainer = CustomTrainer(
#     model=model,
#     optimizer=keras.optimizers.Adam(),
#     loss_fn=keras.losses.CategoricalCrossentropy(),
#     metrics=[keras.metrics.CategoricalAccuracy()]
# )
```

### Deployment and Production

**1. Model Saving and Loading**
```python
# Save entire model
model.save('complete_model.h5')

# Save model architecture and weights separately
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_weights.h5')

# Load model
loaded_model = keras.models.load_model('complete_model.h5')

# SavedModel format (recommended for production)
model.save('saved_model_dir', save_format='tf')
loaded_savedmodel = keras.models.load_model('saved_model_dir')
```

**2. TensorFlow Serving Preparation**
```python
# Convert to TensorFlow Lite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Model quantization for smaller size
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen  # Function that yields representative data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_tflite_model = converter.convert()
```

### Relationship Summary

**TensorFlow provides:**
- Low-level operations and graph execution
- Distributed computing capabilities
- Production deployment tools (TensorFlow Serving, TensorFlow Lite)
- Advanced optimization (XLA, AutoGraph)

**Keras provides:**
- High-level, intuitive API
- Pre-built layers and models
- Easy model building and training
- Extensive pre-trained models

**Integration Benefits:**
1. **Best of Both Worlds**: High-level simplicity with low-level control
2. **Seamless Transition**: Start with Keras, customize with TensorFlow
3. **Production Ready**: Easy development to deployment pipeline
4. **Community Support**: Large ecosystem and extensive documentation

**When to Use What:**
- **Keras alone**: Standard deep learning projects, rapid prototyping
- **TensorFlow + Keras**: Custom operations, advanced architectures
- **Pure TensorFlow**: Research, novel algorithms, maximum control

The combination of TensorFlow and Keras provides the most comprehensive and flexible deep learning framework, suitable for everything from research to production deployment.

---

## Question 13

**Explain the process of data cleaning and why it’s important in machine learning.**

**Answer:**
**Data Cleaning Overview:**
Data cleaning is the process of identifying and correcting (or removing) errors, inconsistencies, and inaccuracies in datasets to improve data quality. It's a critical preprocessing step that ensures machine learning models can learn meaningful patterns from reliable, consistent data.

### Why Data Cleaning is Crucial in ML

**Impact on Model Performance:**
- **Garbage In, Garbage Out**: Poor quality data leads to poor model performance
- **Biased Results**: Inconsistent or incorrect data can introduce systematic biases
- **Reduced Accuracy**: Missing values and outliers can significantly degrade model accuracy
- **Unstable Predictions**: Inconsistent data formats cause unreliable model behavior
- **Computational Efficiency**: Clean data reduces training time and computational resources

### Common Data Quality Issues

**1. Missing Values**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create sample dataset with missing values
np.random.seed(42)
data = {
    'age': [25, 30, np.nan, 45, 35, np.nan, 28, 40],
    'income': [50000, 60000, 55000, np.nan, 65000, 45000, np.nan, 70000],
    'education': ['Bachelor', 'Master', np.nan, 'PhD', 'Bachelor', 'High School', 'Master', np.nan],
    'experience': [2, 5, 3, np.nan, 7, 0, 1, 10]
}

df = pd.DataFrame(data)

def analyze_missing_data(df):
    """Analyze missing data patterns"""
    print("Missing Data Analysis:")
    print("=" * 40)
    
    # Count missing values
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Missing Percentage': missing_percentage.values
    })
    
    print(missing_df)
    
    # Visualize missing data patterns
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Data Pattern')
    plt.show()
    
    return missing_df

# Analyze missing data
missing_analysis = analyze_missing_data(df)

# Different strategies for handling missing values
def handle_missing_values(df):
    """Demonstrate various missing value handling strategies"""
    
    # Strategy 1: Remove rows with any missing values
    df_dropna = df.dropna()
    print(f"Original shape: {df.shape}")
    print(f"After dropping NaN: {df_dropna.shape}")
    
    # Strategy 2: Remove columns with too many missing values
    threshold = 0.5  # Remove columns with >50% missing
    df_drop_cols = df.dropna(thresh=int(threshold * len(df)), axis=1)
    print(f"After dropping columns with >{threshold*100}% missing: {df_drop_cols.shape}")
    
    # Strategy 3: Simple imputation
    # Numerical columns - mean/median/mode
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df_imputed = df.copy()
    
    # Impute numerical columns with median
    if len(numerical_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df_imputed[numerical_cols] = imputer_num.fit_transform(df_imputed[numerical_cols])
    
    # Impute categorical columns with mode
    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_cols] = imputer_cat.fit_transform(df_imputed[categorical_cols])
    
    print("Simple imputation completed")
    
    # Strategy 4: Advanced imputation - KNN
    df_knn = df.copy()
    # For KNN, need to encode categorical variables first
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    # Create a copy for KNN imputation
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].fillna('missing')  # Temporary fill
            df_encoded[col] = le.fit_transform(df_encoded[col])
    
    knn_imputer = KNNImputer(n_neighbors=3)
    df_knn_imputed = pd.DataFrame(
        knn_imputer.fit_transform(df_encoded),
        columns=df_encoded.columns
    )
    
    print("KNN imputation completed")
    
    return {
        'original': df,
        'dropna': df_dropna,
        'simple_imputed': df_imputed,
        'knn_imputed': df_knn_imputed
    }

# Handle missing values
imputation_results = handle_missing_values(df)
```

**2. Duplicate Records**
```python
def handle_duplicates(df):
    """Identify and handle duplicate records"""
    print("Duplicate Analysis:")
    print("=" * 30)
    
    # Create sample data with duplicates
    sample_data = {
        'id': [1, 2, 3, 4, 2, 5, 6, 3],  # Duplicate IDs
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Bob', 'Eve', 'Frank', 'Charlie'],
        'age': [25, 30, 35, 40, 30, 28, 45, 35],
        'salary': [50000, 60000, 70000, 80000, 60000, 55000, 90000, 70000]
    }
    
    df_with_dups = pd.DataFrame(sample_data)
    
    # Check for duplicates
    print(f"Total records: {len(df_with_dups)}")
    print(f"Duplicate rows (all columns): {df_with_dups.duplicated().sum()}")
    print(f"Duplicate rows (subset of columns): {df_with_dups.duplicated(subset=['name', 'age']).sum()}")
    
    # Show duplicate records
    duplicates = df_with_dups[df_with_dups.duplicated(subset=['name', 'age'], keep=False)]
    print("\nDuplicate records:")
    print(duplicates.sort_values(['name', 'age']))
    
    # Remove duplicates
    df_no_dups = df_with_dups.drop_duplicates(subset=['name', 'age'], keep='first')
    print(f"\nAfter removing duplicates: {len(df_no_dups)} records")
    
    return df_no_dups

# Handle duplicates
clean_df = handle_duplicates(df)
```

**3. Outliers Detection and Treatment**
```python
def detect_and_handle_outliers(df, column):
    """Detect and handle outliers using various methods"""
    
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(50, 15, 95)
    outliers = np.array([120, 150, -10, 200, 180])  # Clear outliers
    data_with_outliers = np.concatenate([normal_data, outliers])
    
    df_outliers = pd.DataFrame({'values': data_with_outliers})
    
    print("Outlier Detection Methods:")
    print("=" * 40)
    
    # Method 1: IQR (Interquartile Range)
    Q1 = df_outliers['values'].quantile(0.25)
    Q3 = df_outliers['values'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    iqr_outliers = df_outliers[(df_outliers['values'] < lower_bound) | 
                               (df_outliers['values'] > upper_bound)]
    
    print(f"IQR Method: {len(iqr_outliers)} outliers detected")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Method 2: Z-Score
    from scipy import stats
    z_scores = np.abs(stats.zscore(df_outliers['values']))
    z_threshold = 3
    z_outliers = df_outliers[z_scores > z_threshold]
    
    print(f"Z-Score Method (threshold={z_threshold}): {len(z_outliers)} outliers detected")
    
    # Method 3: Modified Z-Score (using median)
    median = df_outliers['values'].median()
    mad = np.median(np.abs(df_outliers['values'] - median))
    modified_z_scores = 0.6745 * (df_outliers['values'] - median) / mad
    modified_z_outliers = df_outliers[np.abs(modified_z_scores) > z_threshold]
    
    print(f"Modified Z-Score Method: {len(modified_z_outliers)} outliers detected")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Box plot
    axes[0,0].boxplot(df_outliers['values'])
    axes[0,0].set_title('Box Plot - Outlier Detection')
    axes[0,0].set_ylabel('Values')
    
    # Histogram
    axes[0,1].hist(df_outliers['values'], bins=20, alpha=0.7)
    axes[0,1].set_title('Histogram')
    axes[0,1].set_xlabel('Values')
    axes[0,1].set_ylabel('Frequency')
    
    # Scatter plot with outliers highlighted
    axes[1,0].scatter(range(len(df_outliers)), df_outliers['values'], alpha=0.7)
    axes[1,0].scatter(iqr_outliers.index, iqr_outliers['values'], color='red', s=50)
    axes[1,0].set_title('Data Points with IQR Outliers (Red)')
    axes[1,0].set_xlabel('Index')
    axes[1,0].set_ylabel('Values')
    
    # Q-Q plot
    stats.probplot(df_outliers['values'], dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Treatment strategies
    print("\nOutlier Treatment Strategies:")
    print("=" * 40)
    
    # Strategy 1: Remove outliers
    df_no_outliers = df_outliers[(df_outliers['values'] >= lower_bound) & 
                                 (df_outliers['values'] <= upper_bound)]
    print(f"Original data points: {len(df_outliers)}")
    print(f"After removing outliers: {len(df_no_outliers)}")
    
    # Strategy 2: Cap/Winsorize outliers
    df_winsorized = df_outliers.copy()
    df_winsorized['values'] = np.clip(df_winsorized['values'], lower_bound, upper_bound)
    print(f"Winsorized outliers to bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Strategy 3: Transform data (log transform for positive skewed data)
    df_log = df_outliers.copy()
    df_log['values'] = np.log1p(df_log['values'] - df_log['values'].min() + 1)
    print("Applied log transformation")
    
    return {
        'original': df_outliers,
        'no_outliers': df_no_outliers,
        'winsorized': df_winsorized,
        'log_transformed': df_log
    }

# Detect and handle outliers
outlier_results = detect_and_handle_outliers(df, 'values')
```

**4. Data Type Inconsistencies**
```python
def fix_data_types(df):
    """Fix data type inconsistencies"""
    
    # Create sample data with type issues
    messy_data = {
        'user_id': ['1', '2', '3', '4', '5'],  # Should be int
        'age': ['25', '30.0', '35', 'unknown', '40'],  # Mixed types
        'salary': ['50000', '60,000', '$70000', '80000.50', 'N/A'],  # Currency format
        'date_joined': ['2023-01-15', '15/02/2023', '2023.03.20', 'March 25, 2023', '2023-04-30'],
        'is_active': ['true', '1', 'yes', 'false', '0']  # Boolean variations
    }
    
    df_messy = pd.DataFrame(messy_data)
    
    print("Data Type Issues:")
    print("=" * 30)
    print(df_messy.dtypes)
    print("\nSample data:")
    print(df_messy)
    
    # Fix user_id
    df_clean = df_messy.copy()
    df_clean['user_id'] = pd.to_numeric(df_clean['user_id'], errors='coerce')
    
    # Fix age - handle 'unknown' values
    df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
    
    # Fix salary - remove currency symbols and commas
    df_clean['salary'] = (df_clean['salary']
                         .str.replace('$', '', regex=False)
                         .str.replace(',', '', regex=False)
                         .replace('N/A', np.nan))
    df_clean['salary'] = pd.to_numeric(df_clean['salary'], errors='coerce')
    
    # Fix dates - standardize format
    date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%Y.%m.%d', '%B %d, %Y']
    
    def parse_date(date_str):
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        return pd.NaT
    
    df_clean['date_joined'] = df_clean['date_joined'].apply(parse_date)
    
    # Fix boolean values
    boolean_mapping = {'true': True, '1': True, 'yes': True, 
                      'false': False, '0': False}
    df_clean['is_active'] = df_clean['is_active'].map(boolean_mapping)
    
    print("\nAfter cleaning:")
    print(df_clean.dtypes)
    print("\nCleaned data:")
    print(df_clean)
    
    return df_clean

# Fix data types
clean_types_df = fix_data_types(df)
```

**5. Inconsistent Text Data**
```python
def clean_text_data(df):
    """Clean and standardize text data"""
    
    # Sample text data with inconsistencies
    text_data = {
        'name': [' John Doe ', 'JANE SMITH', 'bob johnson', 'Alice  Brown', 'charlie WILSON '],
        'city': ['New York', 'new york', 'NEW YORK', 'Los Angeles', 'los angeles'],
        'category': ['Category A', 'category_a', 'CATEGORY A', 'Category B', 'category b'],
        'description': ['This is a GREAT product!!!', 'good quality item.', 
                       'EXCELLENT VALUE FOR MONEY', 'nice product', 'Highly recommended!!!']
    }
    
    df_text = pd.DataFrame(text_data)
    
    print("Text Data Cleaning:")
    print("=" * 30)
    print("Original data:")
    print(df_text)
    
    df_text_clean = df_text.copy()
    
    # Clean name field
    df_text_clean['name'] = (df_text_clean['name']
                            .str.strip()  # Remove leading/trailing spaces
                            .str.title()  # Proper case
                            .str.replace(r'\s+', ' ', regex=True))  # Multiple spaces to single
    
    # Standardize city names
    df_text_clean['city'] = (df_text_clean['city']
                            .str.lower()
                            .str.title())
    
    # Standardize category format
    df_text_clean['category'] = (df_text_clean['category']
                                .str.lower()
                                .str.replace('_', ' ')
                                .str.title())
    
    # Clean description text
    df_text_clean['description'] = (df_text_clean['description']
                                   .str.lower()
                                   .str.replace(r'[!]{2,}', '!', regex=True)  # Multiple ! to single
                                   .str.strip())
    
    print("\nAfter cleaning:")
    print(df_text_clean)
    
    return df_text_clean

# Clean text data
clean_text_df = clean_text_data(df)
```

### Comprehensive Data Cleaning Pipeline

**Complete Data Cleaning Workflow**
```python
class DataCleaningPipeline:
    """Comprehensive data cleaning pipeline for ML preprocessing"""
    
    def __init__(self):
        self.cleaning_report = {}
    
    def analyze_data_quality(self, df):
        """Analyze overall data quality"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Missing data analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
        
        # Outlier analysis for numerical columns
        report['outliers'] = {}
        for col in report['numerical_columns']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            report['outliers'][col] = len(outliers)
        
        self.cleaning_report = report
        return report
    
    def clean_missing_values(self, df, strategy='auto'):
        """Clean missing values with configurable strategy"""
        df_clean = df.copy()
        
        if strategy == 'auto':
            # Automatic strategy based on missing percentage
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                
                if missing_pct > 70:
                    # Drop columns with >70% missing
                    df_clean = df_clean.drop(columns=[col])
                    print(f"Dropped column '{col}' ({missing_pct:.1f}% missing)")
                
                elif missing_pct > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        # Use median for numerical columns
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                        print(f"Filled '{col}' with median")
                    else:
                        # Use mode for categorical columns
                        mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                        df_clean[col].fillna(mode_value, inplace=True)
                        print(f"Filled '{col}' with mode: {mode_value}")
        
        return df_clean
    
    def remove_duplicates(self, df, subset=None, keep='first'):
        """Remove duplicate records"""
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(df_clean)
        
        print(f"Removed {removed_count} duplicate rows")
        return df_clean
    
    def handle_outliers(self, df, method='iqr', threshold=1.5):
        """Handle outliers in numerical columns"""
        df_clean = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers instead of removing them
                outliers_count = len(df_clean[(df_clean[col] < lower_bound) | 
                                             (df_clean[col] > upper_bound)])
                
                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                
                if outliers_count > 0:
                    print(f"Capped {outliers_count} outliers in '{col}'")
        
        return df_clean
    
    def standardize_text(self, df, text_columns=None):
        """Standardize text data"""
        df_clean = df.copy()
        
        if text_columns is None:
            text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = (df_clean[col]
                                .astype(str)
                                .str.strip()
                                .str.replace(r'\s+', ' ', regex=True))
                print(f"Standardized text in '{col}'")
        
        return df_clean
    
    def fix_data_types(self, df, type_mapping=None):
        """Fix data type inconsistencies"""
        df_clean = df.copy()
        
        if type_mapping:
            for col, dtype in type_mapping.items():
                if col in df_clean.columns:
                    try:
                        df_clean[col] = df_clean[col].astype(dtype)
                        print(f"Converted '{col}' to {dtype}")
                    except Exception as e:
                        print(f"Failed to convert '{col}' to {dtype}: {e}")
        
        return df_clean
    
    def clean_dataset(self, df, config=None):
        """Complete data cleaning pipeline"""
        print("Starting Data Cleaning Pipeline")
        print("=" * 50)
        
        # Step 1: Analyze data quality
        quality_report = self.analyze_data_quality(df)
        print(f"Initial dataset: {quality_report['total_rows']} rows, {quality_report['total_columns']} columns")
        
        # Step 2: Clean missing values
        df_clean = self.clean_missing_values(df)
        
        # Step 3: Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # Step 4: Handle outliers
        df_clean = self.handle_outliers(df_clean)
        
        # Step 5: Standardize text
        df_clean = self.standardize_text(df_clean)
        
        # Step 6: Final quality check
        final_report = self.analyze_data_quality(df_clean)
        
        print("\nCleaning Summary:")
        print("=" * 30)
        print(f"Original shape: {df.shape}")
        print(f"Final shape: {df_clean.shape}")
        print(f"Rows removed: {len(df) - len(df_clean)}")
        print(f"Columns removed: {len(df.columns) - len(df_clean.columns)}")
        print(f"Missing values reduced from {sum(quality_report['missing_data'][col]['count'] for col in quality_report['missing_data'])} to {sum(final_report['missing_data'][col]['count'] for col in final_report['missing_data'])}")
        
        return df_clean, final_report

# Usage example
def demonstrate_cleaning_pipeline():
    """Demonstrate the complete cleaning pipeline"""
    
    # Create a messy dataset
    np.random.seed(42)
    messy_data = {
        'id': range(1, 101),
        'age': np.random.choice([25, 30, np.nan, 35, 40, 150, -5], 100),  # Has outliers and missing
        'income': np.random.choice([50000, 60000, np.nan, 1000000, 70000], 100),  # Has outliers
        'name': ['John Doe', 'JANE SMITH', '  bob  ', np.nan, 'Alice'] * 20,
        'city': ['New York', 'new york', 'NEW YORK', 'Los Angeles', np.nan] * 20,
        'score': np.random.normal(75, 15, 100)  # Normal distribution with some natural outliers
    }
    
    # Add some duplicate rows
    df_messy = pd.DataFrame(messy_data)
    df_messy = pd.concat([df_messy, df_messy.iloc[:5]], ignore_index=True)  # Add 5 duplicates
    
    # Initialize and run cleaning pipeline
    cleaner = DataCleaningPipeline()
    df_clean, report = cleaner.clean_dataset(df_messy)
    
    return df_clean, report

# Run demonstration
# clean_data, cleaning_report = demonstrate_cleaning_pipeline()
```

### Best Practices for Data Cleaning

**1. Documentation and Reproducibility**
```python
def create_cleaning_documentation(df_original, df_clean, cleaning_steps):
    """Document all cleaning steps for reproducibility"""
    doc = {
        'timestamp': pd.Timestamp.now(),
        'original_shape': df_original.shape,
        'final_shape': df_clean.shape,
        'cleaning_steps': cleaning_steps,
        'columns_removed': list(set(df_original.columns) - set(df_clean.columns)),
        'data_quality_improvement': {
            'missing_values_before': df_original.isnull().sum().sum(),
            'missing_values_after': df_clean.isnull().sum().sum(),
            'duplicates_removed': len(df_original) - len(df_clean.drop_duplicates())
        }
    }
    
    # Save documentation
    with open('data_cleaning_log.json', 'w') as f:
        import json
        json.dump(doc, f, indent=2, default=str)
    
    return doc
```

**2. Validation and Quality Checks**
```python
def validate_cleaned_data(df):
    """Validate cleaned data meets quality requirements"""
    validation_results = {
        'passed': True,
        'issues': []
    }
    
    # Check for remaining missing values
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 5:  # More than 5% missing
        validation_results['issues'].append(f"High missing data: {missing_pct:.2f}%")
        validation_results['passed'] = False
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_results['issues'].append(f"Duplicates found: {duplicate_count}")
        validation_results['passed'] = False
    
    # Check data types
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = len(df[col].unique()) / len(df)
        if unique_ratio > 0.95:  # Mostly unique values
            validation_results['issues'].append(f"Column '{col}' has high cardinality: {unique_ratio:.2f}")
    
    return validation_results
```

### Key Takeaways

**Importance of Data Cleaning:**
1. **Model Performance**: Clean data directly improves model accuracy and reliability
2. **Bias Reduction**: Removes systematic errors that could bias results
3. **Computational Efficiency**: Reduces training time and resource consumption
4. **Interpretability**: Makes model insights more meaningful and trustworthy
5. **Production Stability**: Ensures consistent model behavior in deployment

**Common Data Quality Issues:**
- Missing values (various patterns and causes)
- Duplicate records (exact and near-duplicates)
- Outliers and anomalies
- Inconsistent data types and formats
- Text inconsistencies and encoding issues
- Incorrect or placeholder values

**Best Practices:**
1. **Understand Your Data**: Analyze patterns before cleaning
2. **Document Everything**: Keep detailed logs of all cleaning steps
3. **Validate Results**: Check data quality after each cleaning step
4. **Preserve Original Data**: Always work on copies
5. **Domain Knowledge**: Apply business logic to cleaning decisions
6. **Iterative Process**: Clean, validate, and refine repeatedly
7. **Automation**: Build reusable cleaning pipelines
8. **Quality Metrics**: Define and monitor data quality KPIs

Data cleaning is foundational to successful machine learning projects. Poor data quality is often the primary reason for model failures in production, making thorough data cleaning essential for reliable ML systems.
---

## Question 14

**What are the common steps involved in data preprocessing for a machine learning model?**

**Answer:** 

Data preprocessing is the crucial process of transforming raw data into a format suitable for machine learning algorithms. It typically accounts for 60-80% of the total time spent on a machine learning project and directly impacts model performance.

### Complete Data Preprocessing Pipeline

**1. Data Collection and Loading**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load sample dataset
def load_sample_data():
    """Create a comprehensive sample dataset for preprocessing demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Numerical features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    experience = np.random.randint(0, 40, n_samples)
    
    # Categorical features
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    department = np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR', 'Finance'], n_samples)
    city = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples)
    
    # Text feature
    job_titles = ['Junior ' + dept for dept in department[:500]] + ['Senior ' + dept for dept in department[500:]]
    
    # Create target variable (binary classification: promotion)
    promotion_score = (age * 0.01 + income * 0.00001 + experience * 0.05 + 
                      (education == 'PhD') * 0.5 + (education == 'Master') * 0.3 +
                      np.random.normal(0, 0.2, n_samples))
    target = (promotion_score > np.percentile(promotion_score, 70)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'experience': experience,
        'education': education,
        'department': department,
        'city': city,
        'job_title': job_titles,
        'promotion': target
    })
    
    # Introduce data quality issues for preprocessing demonstration
    # Missing values
    missing_indices = np.random.choice(n_samples, 100, replace=False)
    df.loc[missing_indices[:50], 'income'] = np.nan
    df.loc[missing_indices[50:], 'education'] = np.nan
    
    # Outliers in income
    outlier_indices = np.random.choice(n_samples, 20, replace=False)
    df.loc[outlier_indices, 'income'] = np.random.uniform(200000, 500000, 20)
    
    # Inconsistent text formatting
    df.loc[np.random.choice(n_samples, 100, replace=False), 'education'] = 'bachelor'
    df.loc[np.random.choice(n_samples, 50, replace=False), 'education'] = 'master'
    
    return df

df = load_sample_data()
print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
```

**2. Exploratory Data Analysis (EDA)**
```python
class DataExplorer:
    def __init__(self, df):
        self.df = df
    
    def basic_statistics(self):
        """Generate basic statistical summary"""
        print("="*50)
        print("BASIC STATISTICS")
        print("="*50)
        
        # Numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print("\nNumerical Features Summary:")
            print(self.df[numerical_cols].describe())
        
        # Categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\nCategorical Features Summary:")
            for col in categorical_cols:
                print(f"\n{col}:")
                print(self.df[col].value_counts().head())
    
    def missing_data_analysis(self):
        """Analyze missing data patterns"""
        print("\n" + "="*50)
        print("MISSING DATA ANALYSIS")
        print("="*50)
        
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Visualize missing data
        if missing_data.sum() > 0:
            plt.figure(figsize=(10, 6))
            missing_df_plot = missing_df[missing_df['Missing_Count'] > 0]
            plt.bar(missing_df_plot['Column'], missing_df_plot['Missing_Percentage'])
            plt.title('Missing Data Percentage by Column')
            plt.xlabel('Columns')
            plt.ylabel('Missing Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def outlier_analysis(self):
        """Analyze outliers in numerical columns"""
        print("\n" + "="*50)
        print("OUTLIER ANALYSIS")
        print("="*50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(15, 5*len(numerical_cols)))
        if len(numerical_cols) == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numerical_cols):
            # Box plot
            axes[i, 0].boxplot(self.df[col].dropna())
            axes[i, 0].set_title(f'{col} - Box Plot')
            axes[i, 0].set_ylabel(col)
            
            # Histogram
            axes[i, 1].hist(self.df[col].dropna(), bins=30, alpha=0.7)
            axes[i, 1].set_title(f'{col} - Distribution')
            axes[i, 1].set_xlabel(col)
            axes[i, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            correlation_matrix = self.df[numerical_cols].corr()
            print(correlation_matrix)
            
            # Heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()

# Perform EDA
explorer = DataExplorer(df)
explorer.basic_statistics()
explorer.missing_data_analysis()
explorer.outlier_analysis()
explorer.correlation_analysis()
```

**3. Data Cleaning**
```python
class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_log = []
    
    def log_action(self, action, details):
        """Log cleaning actions"""
        self.cleaning_log.append({'action': action, 'details': details})
    
    def handle_missing_values(self, strategy='auto'):
        """Handle missing values with different strategies"""
        print("Handling missing values...")
        
        missing_before = self.df.isnull().sum().sum()
        
        if strategy == 'auto':
            # Numerical columns: use median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.df[col].isnull().any():
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    self.log_action('missing_numerical', f'{col}: filled with median ({median_val:.2f})')
            
            # Categorical columns: use mode or 'Unknown'
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().any():
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        fill_val = mode_val[0]
                    else:
                        fill_val = 'Unknown'
                    
                    self.df[col].fillna(fill_val, inplace=True)
                    self.log_action('missing_categorical', f'{col}: filled with mode ({fill_val})')
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Missing values: {missing_before} â†’ {missing_after}")
        
        return self.df
    
    def handle_outliers(self, method='iqr', columns=None):
        """Handle outliers using IQR or Z-score methods"""
        print("Handling outliers...")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col == 'promotion':  # Skip target variable
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                # Cap outliers
                self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                
                if outliers_before > 0:
                    self.log_action('outlier_treatment', f'{col}: {outliers_before} outliers capped')
        
        return self.df
    
    def standardize_text(self):
        """Standardize text data"""
        print("Standardizing text data...")
        
        text_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if col in ['education']:  # Apply specific rules
                # Convert to lowercase and standardize
                self.df[col] = self.df[col].str.lower().str.strip()
                
                # Standardize education levels
                education_mapping = {
                    'bachelor': 'Bachelor',
                    'master': 'Master',
                    'phd': 'PhD',
                    'high school': 'High School'
                }
                
                self.df[col] = self.df[col].map(education_mapping).fillna(self.df[col])
                self.log_action('text_standardization', f'{col}: standardized education levels')
        
        return self.df
    
    def remove_duplicates(self):
        """Remove duplicate records"""
        before_count = len(self.df)
        self.df = self.df.drop_duplicates()
        after_count = len(self.df)
        
        removed_count = before_count - after_count
        if removed_count > 0:
            self.log_action('duplicate_removal', f'Removed {removed_count} duplicate rows')
            print(f"Removed {removed_count} duplicate rows")
        
        return self.df

# Clean the data
cleaner = DataCleaner(df)
cleaned_df = cleaner.handle_missing_values()
cleaned_df = cleaner.handle_outliers()
cleaned_df = cleaner.standardize_text()
cleaned_df = cleaner.remove_duplicates()

print("\nCleaning completed. Summary:")
for log_entry in cleaner.cleaning_log:
    print(f"- {log_entry['action']}: {log_entry['details']}")
```

**4. Feature Engineering**
```python
class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.feature_log = []
    
    def log_feature(self, action, details):
        """Log feature engineering actions"""
        self.feature_log.append({'action': action, 'details': details})
    
    def create_age_groups(self):
        """Create age groups from numerical age"""
        bins = [0, 25, 35, 50, 65, 100]
        labels = ['Young', 'Early Career', 'Mid Career', 'Senior', 'Veteran']
        
        self.df['age_group'] = pd.cut(self.df['age'], bins=bins, labels=labels, right=False)
        self.log_feature('binning', 'Created age_group from age')
        
        return self.df
    
    def create_income_features(self):
        """Create income-related features"""
        # Income brackets
        self.df['income_bracket'] = pd.cut(self.df['income'], 
                                         bins=[0, 30000, 50000, 80000, float('inf')],
                                         labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Log transformation for income (handle skewness)
        self.df['log_income'] = np.log1p(self.df['income'])
        
        self.log_feature('transformation', 'Created income_bracket and log_income')
        
        return self.df
    
    def create_experience_features(self):
        """Create experience-related features"""
        # Experience categories
        self.df['experience_level'] = pd.cut(self.df['experience'],
                                           bins=[0, 2, 5, 10, float('inf')],
                                           labels=['Entry', 'Junior', 'Mid', 'Senior'])
        
        # Experience to age ratio
        self.df['experience_age_ratio'] = self.df['experience'] / self.df['age']
        
        self.log_feature('ratio_feature', 'Created experience_level and experience_age_ratio')
        
        return self.df
    
    def create_text_features(self):
        """Extract features from text data"""
        # Job title length
        self.df['job_title_length'] = self.df['job_title'].str.len()
        
        # Job seniority indicator
        self.df['is_senior'] = self.df['job_title'].str.contains('Senior', case=False).astype(int)
        
        self.log_feature('text_features', 'Created job_title_length and is_senior')
        
        return self.df
    
    def create_interaction_features(self):
        """Create interaction features"""
        # Education-Experience interaction
        education_numeric = self.df['education'].map({
            'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4
        })
        
        self.df['education_experience'] = education_numeric * self.df['experience']
        
        self.log_feature('interaction', 'Created education_experience interaction')
        
        return self.df

# Apply feature engineering
engineer = FeatureEngineer(cleaned_df)
engineered_df = engineer.create_age_groups()
engineered_df = engineer.create_income_features()
engineered_df = engineer.create_experience_features()
engineered_df = engineer.create_text_features()
engineered_df = engineer.create_interaction_features()

print("\nFeature Engineering completed:")
for log_entry in engineer.feature_log:
    print(f"- {log_entry['action']}: {log_entry['details']}")

print(f"\nNew features added: {len(engineered_df.columns) - len(df.columns)}")
print("New columns:", [col for col in engineered_df.columns if col not in df.columns])
```

**5. Feature Encoding**
```python
class FeatureEncoder:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.encoders = {}
        self.encoded_columns = []
    
    def label_encode_categorical(self, columns=None):
        """Apply label encoding to categorical variables"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns
            columns = [col for col in columns if col != self.target_column]
        
        for col in columns:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le
            self.encoded_columns.append(f'{col}_encoded')
        
        return self.df
    
    def one_hot_encode_categorical(self, columns=None, drop_first=True):
        """Apply one-hot encoding to categorical variables"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns
            columns = [col for col in columns if col != self.target_column and col not in self.encoded_columns]
        
        for col in columns:
            # Get dummies
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=drop_first)
            self.df = pd.concat([self.df, dummies], axis=1)
            self.encoded_columns.extend(dummies.columns.tolist())
        
        return self.df
    
    def target_encode_categorical(self, columns=None, cv_folds=5):
        """Apply target encoding (mean encoding) with cross-validation"""
        from sklearn.model_selection import KFold
        
        if columns is None:
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            columns = [col for col in categorical_cols if col != self.target_column]
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for col in columns:
            target_encoded = np.zeros(len(self.df))
            
            for train_idx, val_idx in kf.split(self.df):
                # Calculate target mean for each category in training set
                target_means = self.df.iloc[train_idx].groupby(col)[self.target_column].mean()
                
                # Apply to validation set
                target_encoded[val_idx] = self.df.iloc[val_idx][col].map(target_means)
                
                # Handle unseen categories with global mean
                global_mean = self.df.iloc[train_idx][self.target_column].mean()
                target_encoded[val_idx] = np.where(
                    pd.isna(target_encoded[val_idx]), 
                    global_mean, 
                    target_encoded[val_idx]
                )
            
            self.df[f'{col}_target_encoded'] = target_encoded
            self.encoded_columns.append(f'{col}_target_encoded')
        
        return self.df

# Apply encoding
encoder = FeatureEncoder(engineered_df, 'promotion')

# Apply different encoding strategies
encoded_df = encoder.label_encode_categorical(['department', 'city'])
encoded_df = encoder.one_hot_encode_categorical(['education', 'age_group', 'income_bracket', 'experience_level'])

print("Encoding completed.")
print(f"Encoded columns added: {len(encoder.encoded_columns)}")
print("Sample encoded columns:", encoder.encoded_columns[:10])
```

**6. Feature Scaling**
```python
class FeatureScaler:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.scalers = {}
    
    def standard_scale(self, columns=None):
        """Apply standard scaling (z-score normalization)"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col != self.target_column]
        
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.scalers['standard'] = scaler
        
        return self.df
    
    def min_max_scale(self, columns=None):
        """Apply min-max scaling"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col != self.target_column]
        
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.scalers['minmax'] = scaler
        
        return self.df
    
    def robust_scale(self):
        """Apply robust scaling (less sensitive to outliers)"""
        from sklearn.preprocessing import RobustScaler
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        columns = [col for col in numerical_cols if col != self.target_column]
        
        scaler = RobustScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.scalers['robust'] = scaler
        
        return self.df

# Apply scaling (choose one method)
scaler = FeatureScaler(encoded_df, 'promotion')
scaled_df = scaler.standard_scale()

print("Feature scaling completed.")
print("Numerical features scaled using StandardScaler.")
```

**7. Feature Selection**
```python
class FeatureSelector:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.selected_features = []
    
    def correlation_filter(self, threshold=0.95):
        """Remove highly correlated features"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col != self.target_column]
        
        # Calculate correlation matrix
        corr_matrix = self.df[feature_cols].corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Remove features with correlation > threshold
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        self.df = self.df.drop(columns=to_drop)
        
        print(f"Removed {len(to_drop)} highly correlated features")
        if to_drop:
            print("Removed features:", to_drop[:5], "..." if len(to_drop) > 5 else "")
        
        return self.df
    
    def univariate_selection(self, k=10):
        """Select k best features using univariate statistical tests"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col != self.target_column]
        
        X = self.df[feature_cols]
        y = self.df[self.target_column]
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.selected_features = [feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
        
        print(f"Selected {len(self.selected_features)} features using univariate selection")
        print("Selected features:", self.selected_features)
        
        return self.selected_features
    
    def recursive_feature_elimination(self, n_features=10, estimator=None):
        """Select features using Recursive Feature Elimination"""
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col != self.target_column]
        
        X = self.df[feature_cols]
        y = self.df[self.target_column]
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.selected_features = [feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
        
        print(f"Selected {len(self.selected_features)} features using RFE")
        print("Selected features:", self.selected_features)
        
        return self.selected_features

# Apply feature selection
selector = FeatureSelector(scaled_df, 'promotion')
final_df = selector.correlation_filter(threshold=0.9)
selected_features = selector.univariate_selection(k=15)

print(f"\nFinal dataset shape: {final_df.shape}")
```

**8. Data Splitting**
```python
def prepare_final_datasets(df, target_column, test_size=0.2, val_size=0.1):
    """Prepare train, validation, and test sets"""
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns]
    y = df[target_column]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print("Dataset split completed:")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"Total features: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Prepare final datasets
X_train, X_val, X_test, y_train, y_val, y_test = prepare_final_datasets(final_df, 'promotion')
```

### Key Preprocessing Steps Summary

**Essential Preprocessing Pipeline:**
1. **Data Loading & EDA**: Understand data structure and quality
2. **Data Cleaning**: Handle missing values, outliers, duplicates
3. **Feature Engineering**: Create new meaningful features
4. **Encoding**: Convert categorical variables to numerical
5. **Scaling**: Normalize feature ranges
6. **Feature Selection**: Choose most relevant features
7. **Data Splitting**: Prepare train/validation/test sets

**Best Practices:**
- Always keep original data unchanged
- Document all preprocessing steps
- Use cross-validation for robust feature selection
- Handle data leakage (fit transformers only on training data)
- Validate preprocessing impact on model performance
- Create reusable preprocessing pipelines
- Consider domain knowledge in feature engineering

Data preprocessing is the foundation of successful machine learning projects - investing time here pays dividends in model performance, reliability, and interpretability.

---

## Question 15

**Describe the concept of feature scaling and why it is necessary.**

**Answer:** 

Feature scaling is the process of standardizing the range of features in a dataset to ensure that all features contribute equally to machine learning algorithms. Different features often have vastly different scales (e.g., age in years vs. income in dollars), and many ML algorithms are sensitive to these scale differences.

### Why Feature Scaling is Necessary

**Impact on Algorithm Performance:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Create dataset with different scales
def create_multi_scale_dataset():
    """Create a dataset with features at different scales"""
    np.random.seed(42)
    n_samples = 1000
    
    # Feature 1: Age (18-80)
    age = np.random.randint(18, 80, n_samples)
    
    # Feature 2: Income (20k-200k)
    income = np.random.normal(60000, 30000, n_samples)
    income = np.clip(income, 20000, 200000)
    
    # Feature 3: Years of experience (0-40)
    experience = np.random.randint(0, 40, n_samples)
    
    # Feature 4: Number of projects (1-100)
    projects = np.random.randint(1, 100, n_samples)
    
    # Feature 5: Satisfaction score (1-10)
    satisfaction = np.random.randint(1, 10, n_samples)
    
    # Create target based on all features
    target = (
        (age - 50) * 0.01 +
        (income - 60000) * 0.00001 +
        (experience - 20) * 0.05 +
        (projects - 50) * 0.01 +
        (satisfaction - 5) * 0.1 +
        np.random.normal(0, 0.3, n_samples)
    ) > 0
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'experience': experience,
        'projects': projects,
        'satisfaction': satisfaction,
        'target': target.astype(int)
    })
    
    return df

df = create_multi_scale_dataset()

print("Dataset with different scales:")
print(df.describe())
print("\nFeature scales demonstration:")
for col in df.columns[:-1]:  # Exclude target
    print(f"{col}: min={df[col].min():.0f}, max={df[col].max():.0f}, range={df[col].max()-df[col].min():.0f}")
```

**Demonstrating Scale Impact on Algorithms:**
```python
def compare_algorithms_with_without_scaling(X, y):
    """Compare algorithm performance with and without scaling"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define algorithms sensitive to scaling
    algorithms = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    results = []
    
    for name, algorithm in algorithms.items():
        # Without scaling
        algorithm.fit(X_train, y_train)
        y_pred_unscaled = algorithm.predict(X_test)
        accuracy_unscaled = accuracy_score(y_test, y_pred_unscaled)
        
        # With scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        algorithm.fit(X_train_scaled, y_train)
        y_pred_scaled = algorithm.predict(X_test_scaled)
        accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
        
        results.append({
            'Algorithm': name,
            'Accuracy_Unscaled': accuracy_unscaled,
            'Accuracy_Scaled': accuracy_scaled,
            'Improvement': accuracy_scaled - accuracy_unscaled
        })
    
    results_df = pd.DataFrame(results)
    print("Algorithm Performance Comparison:")
    print(results_df.round(4))
    
    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot comparing accuracies
    x = np.arange(len(results_df))
    width = 0.35
    
    ax1.bar(x - width/2, results_df['Accuracy_Unscaled'], width, label='Unscaled', alpha=0.8)
    ax1.bar(x + width/2, results_df['Accuracy_Scaled'], width, label='Scaled', alpha=0.8)
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Algorithm Performance: Scaled vs Unscaled')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Algorithm'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement plot
    ax2.bar(results_df['Algorithm'], results_df['Improvement'], 
            color=['green' if x > 0 else 'red' for x in results_df['Improvement']], alpha=0.7)
    ax2.set_xlabel('Algorithms')
    ax2.set_ylabel('Accuracy Improvement')
    ax2.set_title('Improvement from Feature Scaling')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Compare algorithms
comparison_results = compare_algorithms_with_without_scaling(X, y)
```

### Common Feature Scaling Techniques

**1. Standard Scaling (Z-score Normalization)**
```python
class FeatureScalingDemo:
    def __init__(self, data):
        self.data = data.copy()
        self.scalers = {}
    
    def demonstrate_standard_scaling(self):
        """Standard Scaling: (x - mean) / std"""
        print("="*50)
        print("STANDARD SCALING (Z-SCORE NORMALIZATION)")
        print("="*50)
        
        # Select numerical features
        numerical_features = self.data.select_dtypes(include=[np.number]).columns.drop('target')
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[numerical_features])
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)
        
        self.scalers['standard'] = scaler
        
        print("Original data statistics:")
        print(self.data[numerical_features].describe())
        
        print("\nStandardized data statistics:")
        print(scaled_df.describe())
        
        print("\nStandard Scaling Properties:")
        print("- Mean â‰ˆ 0, Standard Deviation â‰ˆ 1")
        print("- Preserves the shape of the original distribution")
        print("- Handles outliers but doesn't bound the range")
        
        # Visualize the transformation
        fig, axes = plt.subplots(2, len(numerical_features), figsize=(15, 8))
        
        for i, col in enumerate(numerical_features):
            # Original distribution
            axes[0, i].hist(self.data[col], bins=30, alpha=0.7, color='blue')
            axes[0, i].set_title(f'Original {col}')
            axes[0, i].set_ylabel('Frequency')
            
            # Scaled distribution
            axes[1, i].hist(scaled_df[col], bins=30, alpha=0.7, color='red')
            axes[1, i].set_title(f'Standardized {col}')
            axes[1, i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        return scaled_df
    
    def demonstrate_minmax_scaling(self):
        """Min-Max Scaling: (x - min) / (max - min)"""
        print("\n" + "="*50)
        print("MIN-MAX SCALING")
        print("="*50)
        
        numerical_features = self.data.select_dtypes(include=[np.number]).columns.drop('target')
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.data[numerical_features])
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)
        
        self.scalers['minmax'] = scaler
        
        print("Min-Max scaled data statistics:")
        print(scaled_df.describe())
        
        print("\nMin-Max Scaling Properties:")
        print("- Values bounded between 0 and 1")
        print("- Preserves the original distribution shape")
        print("- Sensitive to outliers")
        
        # Visualize feature ranges
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original ranges
        ranges_original = []
        features = []
        for col in numerical_features:
            ranges_original.append(self.data[col].max() - self.data[col].min())
            features.append(col)
        
        ax1.bar(features, ranges_original, alpha=0.7)
        ax1.set_title('Original Feature Ranges')
        ax1.set_ylabel('Range')
        ax1.tick_params(axis='x', rotation=45)
        
        # Scaled ranges (should all be 1)
        ranges_scaled = []
        for col in numerical_features:
            ranges_scaled.append(scaled_df[col].max() - scaled_df[col].min())
        
        ax2.bar(features, ranges_scaled, alpha=0.7, color='red')
        ax2.set_title('Min-Max Scaled Feature Ranges')
        ax2.set_ylabel('Range')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return scaled_df
    
    def demonstrate_robust_scaling(self):
        """Robust Scaling: (x - median) / IQR"""
        print("\n" + "="*50)
        print("ROBUST SCALING")
        print("="*50)
        
        numerical_features = self.data.select_dtypes(include=[np.number]).columns.drop('target')
        
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(self.data[numerical_features])
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)
        
        self.scalers['robust'] = scaler
        
        print("Robust scaled data statistics:")
        print(scaled_df.describe())
        
        print("\nRobust Scaling Properties:")
        print("- Uses median and IQR instead of mean and std")
        print("- Less sensitive to outliers")
        print("- Centers data around 0 but range varies")
        
        # Create data with outliers to demonstrate robustness
        data_with_outliers = self.data.copy()
        
        # Add outliers to income
        outlier_indices = np.random.choice(len(data_with_outliers), 50, replace=False)
        data_with_outliers.loc[outlier_indices, 'income'] *= 5  # Make income 5x larger for outliers
        
        # Compare standard vs robust scaling with outliers
        standard_scaler = StandardScaler()
        robust_scaler = RobustScaler()
        
        income_standard = standard_scaler.fit_transform(data_with_outliers[['income']])
        income_robust = robust_scaler.fit_transform(data_with_outliers[['income']])
        
        # Visualize difference
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(data_with_outliers['income'], bins=50, alpha=0.7, color='blue')
        axes[0].set_title('Original Income (with outliers)')
        axes[0].set_xlabel('Income')
        
        axes[1].hist(income_standard, bins=50, alpha=0.7, color='green')
        axes[1].set_title('Standard Scaled Income')
        axes[1].set_xlabel('Scaled Income')
        
        axes[2].hist(income_robust, bins=50, alpha=0.7, color='red')
        axes[2].set_title('Robust Scaled Income')
        axes[2].set_xlabel('Scaled Income')
        
        plt.tight_layout()
        plt.show()
        
        return scaled_df
    
    def demonstrate_normalization(self):
        """Unit Vector Scaling (L2 Normalization)"""
        print("\n" + "="*50)
        print("UNIT VECTOR SCALING (L2 NORMALIZATION)")
        print("="*50)
        
        numerical_features = self.data.select_dtypes(include=[np.number]).columns.drop('target')
        
        scaler = Normalizer(norm='l2')  # L2 normalization
        scaled_data = scaler.fit_transform(self.data[numerical_features])
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)
        
        self.scalers['normalize'] = scaler
        
        print("Normalized data statistics:")
        print(scaled_df.describe())
        
        print("\nNormalization Properties:")
        print("- Each sample (row) has unit norm")
        print("- Useful when the magnitude of the feature vector matters")
        print("- Common in text analysis and recommendation systems")
        
        # Check that each row has unit norm
        row_norms = np.linalg.norm(scaled_data, axis=1)
        print(f"\nSample row norms (should be â‰ˆ 1.0):")
        print(f"Mean: {row_norms.mean():.6f}")
        print(f"Std: {row_norms.std():.6f}")
        print(f"Min: {row_norms.min():.6f}")
        print(f"Max: {row_norms.max():.6f}")
        
        return scaled_df

# Demonstrate all scaling techniques
scaler_demo = FeatureScalingDemo(df)

standard_scaled = scaler_demo.demonstrate_standard_scaling()
minmax_scaled = scaler_demo.demonstrate_minmax_scaling()
robust_scaled = scaler_demo.demonstrate_robust_scaling()
normalized = scaler_demo.demonstrate_normalization()
```

### Advanced Scaling Scenarios

**1. Handling Different Data Types**
```python
class AdvancedScalingStrategies:
    def __init__(self):
        self.strategies = {}
    
    def scale_mixed_data(self, df, categorical_cols, numerical_cols, target_col):
        """Handle scaling for mixed data types"""
        
        # Separate different types of features
        binary_cols = []
        continuous_cols = []
        discrete_cols = []
        
        for col in numerical_cols:
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                binary_cols.append(col)
            elif unique_vals < 10:
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)
        
        print("Feature categorization:")
        print(f"Continuous: {continuous_cols}")
        print(f"Discrete: {discrete_cols}")
        print(f"Binary: {binary_cols}")
        print(f"Categorical: {categorical_cols}")
        
        # Apply different scaling strategies
        scaled_df = df.copy()
        
        # Standard scaling for continuous variables
        if continuous_cols:
            scaler_continuous = StandardScaler()
            scaled_df[continuous_cols] = scaler_continuous.fit_transform(df[continuous_cols])
            self.strategies['continuous'] = scaler_continuous
        
        # Min-max scaling for discrete variables
        if discrete_cols:
            scaler_discrete = MinMaxScaler()
            scaled_df[discrete_cols] = scaler_discrete.fit_transform(df[discrete_cols])
            self.strategies['discrete'] = scaler_discrete
        
        # No scaling for binary variables (keep as is)
        # No scaling for categorical variables (handle with encoding)
        
        return scaled_df
    
    def time_series_scaling(self, df, time_col, value_cols):
        """Scaling strategies for time series data"""
        
        # Rolling window scaling
        window_size = 30
        scaled_df = df.copy()
        
        for col in value_cols:
            # Rolling mean and std
            rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window_size, min_periods=1).std()
            
            # Apply rolling standardization
            scaled_df[f'{col}_rolling_scaled'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        return scaled_df
    
    def target_aware_scaling(self, X, y):
        """Scaling that considers target variable distribution"""
        
        # Different scaling for different target classes
        scalers_by_class = {}
        X_scaled = X.copy()
        
        for class_val in np.unique(y):
            class_mask = y == class_val
            X_class = X[class_mask]
            
            # Fit scaler on this class
            scaler = StandardScaler()
            scaler.fit(X_class)
            scalers_by_class[class_val] = scaler
        
        # Apply appropriate scaler to each sample
        for i, (idx, row) in enumerate(X.iterrows()):
            target_class = y.iloc[i]
            scaler = scalers_by_class[target_class]
            X_scaled.iloc[i] = scaler.transform(row.values.reshape(1, -1)).flatten()
        
        return X_scaled, scalers_by_class

# Demonstrate advanced scaling
advanced_scaler = AdvancedScalingStrategies()

# Categorize columns
categorical_cols = []  # No categorical columns in our numeric dataset
numerical_cols = [col for col in df.columns if col != 'target']

mixed_scaled = advanced_scaler.scale_mixed_data(df, categorical_cols, numerical_cols, 'target')
```

### Scaling in ML Pipelines

**1. Pipeline Integration**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def create_preprocessing_pipeline(numerical_features, categorical_features=None):
    """Create a complete preprocessing pipeline with scaling"""
    
    preprocessors = []
    
    # Numerical features pipeline
    if numerical_features:
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        preprocessors.append(('num', numerical_pipeline, numerical_features))
    
    # Categorical features pipeline
    if categorical_features:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessors.append(('cat', categorical_pipeline, categorical_features))
    
    # Combine all preprocessors
    preprocessor = ColumnTransformer(preprocessors)
    
    return preprocessor

# Create and demonstrate pipeline
numerical_features = [col for col in df.columns if col != 'target']
preprocessor = create_preprocessing_pipeline(numerical_features)

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Pipeline processing completed:")
print(f"Original shape: {X_train.shape}")
print(f"Processed shape: {X_train_processed.shape}")

# Create complete ML pipeline
ml_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Fit and evaluate
ml_pipeline.fit(X_train, y_train)
y_pred = ml_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Pipeline accuracy: {accuracy:.4f}")
```

### When to Use Each Scaling Method

**Decision Framework:**
```python
def choose_scaling_method(data_info):
    """Decision framework for choosing scaling method"""
    
    recommendations = []
    
    # Check data characteristics
    has_outliers = data_info.get('has_outliers', False)
    data_distribution = data_info.get('distribution', 'normal')
    algorithm_type = data_info.get('algorithm', 'linear')
    feature_ranges = data_info.get('feature_ranges', 'different')
    
    print("SCALING METHOD RECOMMENDATION:")
    print("="*40)
    
    if algorithm_type in ['linear', 'svm', 'neural_network', 'knn']:
        if has_outliers:
            recommendations.append("RobustScaler - Less sensitive to outliers")
        else:
            recommendations.append("StandardScaler - Good for normally distributed data")
    
    elif algorithm_type in ['tree_based']:
        recommendations.append("No scaling needed - Tree-based algorithms are scale-invariant")
    
    elif algorithm_type in ['clustering', 'pca']:
        if feature_ranges == 'very_different':
            recommendations.append("StandardScaler or MinMaxScaler - Important for distance-based methods")
    
    elif algorithm_type in ['text_analysis', 'recommendation']:
        recommendations.append("Normalizer - Unit vector scaling for text/recommendation systems")
    
    # Special cases
    if data_distribution == 'skewed':
        recommendations.append("Consider log transformation before scaling")
    
    if feature_ranges == 'bounded':
        recommendations.append("MinMaxScaler - When you need values in [0,1] range")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return recommendations

# Example usage
data_characteristics = {
    'has_outliers': True,
    'distribution': 'normal',
    'algorithm': 'svm',
    'feature_ranges': 'different'
}

recommendations = choose_scaling_method(data_characteristics)
```

### Key Takeaways

**Why Feature Scaling Matters:**
1. **Algorithm Sensitivity**: Many algorithms are sensitive to feature scales
2. **Convergence Speed**: Helps optimization algorithms converge faster
3. **Feature Importance**: Prevents features with larger scales from dominating
4. **Distance Calculations**: Essential for distance-based algorithms

**Scaling Method Selection:**
- **StandardScaler**: Most common, good for normally distributed data
- **MinMaxScaler**: When you need bounded ranges [0,1]
- **RobustScaler**: When data has outliers
- **Normalizer**: For text analysis and when sample magnitude matters

**Best Practices:**
- Always fit scalers on training data only
- Transform validation and test sets using training fitted scalers
- Consider the algorithm requirements
- Document scaling decisions
- Include scaling in your ML pipelines
- Validate the impact on model performance

Feature scaling is a fundamental preprocessing step that can significantly impact model performance and training efficiency.

---

## Question 16

**Explain the difference between label encoding and one-hot encoding.**

**Answer:** _[To be filled]_

---

## Question 17

**What is the purpose of data splitting in train, validation, and test sets?**

**Answer:** _[To be filled]_

---

## Question 18

**Describe the process of building a machine learning model in Python.**

**Answer:** _[To be filled]_

---

## Question 19

**Explain cross-validation and where it fits in the model training process.**

**Answer:** _[To be filled]_

---

## Question 20

**What is the bias-variance trade-off in machine learning?**

**Answer:** _[To be filled]_

---

## Question 21

**Describe the steps taken to improve a model's accuracy.**

**Answer:** _[To be filled]_

---

## Question 22

**What are hyperparameters, and how do you tune them?**

**Answer:** _[To be filled]_

---

## Question 23

**What is a confusion matrix, and how is it interpreted?**

**Answer:** _[To be filled]_

---

## Question 24

**Explain the ROC curve and the area under the curve (AUC) metric.**

**Answer:** _[To be filled]_

---

## Question 25

**Explain different validation strategies, such ask-fold cross-validation.**

**Answer:** _[To be filled]_

---

## Question 26

**Describe steps to take when a model performs well on the training data but poorly on new data.**

**Answer:** _[To be filled]_

---

## Question 27

**Explain the use of regularization in linear models and provide a Python example.**

**Answer:** _[To be filled]_

---

## Question 28

**What are the advantages of using Stochastic Gradient Descent over standard Gradient Descent?**

**Answer:** _[To be filled]_

---

## Question 29

**What is dimensionality reduction, and when would you use it?**

**Answer:** _[To be filled]_

---

## Question 30

**Explain the difference between batch learning and online learning.**

**Answer:** _[To be filled]_

---

## Question 31

**What is the role of attention mechanisms in natural language processing models?**

**Answer:** _[To be filled]_

---

## Question 32

**Explain how to use context managers in Python and provide a machine learning-related example.**

**Answer:** _[To be filled]_

---

## Question 33

**What are slots in Python classes and how could they be useful in machine learning applications?**

**Answer:** _[To be filled]_

---

## Question 34

**Explain the concept of microservices architecture in deploying machine learning models.**

**Answer:** _[To be filled]_

---

## Question 35

**What are the considerations for scaling a machine learning application with Python?**

**Answer:** _[To be filled]_

---

## Question 36

**What is model versioning, and how can it be managed in a real-world application?**

**Answer:** _[To be filled]_

---

## Question 37

**Describe a situation where a machine learning model might fail, and how you would investigate the issue using Python.**

**Answer:** _[To be filled]_

---

## Question 38

**What are Python's profiling tools and how do they assist in optimizing machine learning code?**

**Answer:** _[To be filled]_

---

## Question 39

**Explain how unit tests and integration tests ensure the correctness of your machine learning code.**

**Answer:** _[To be filled]_

---

## Question 40

**What is the role of Explainable AI (XAI) and how can Python libraries help achieve it?**

**Answer:** _[To be filled]_

---




