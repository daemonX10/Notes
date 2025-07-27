## Question 5

**Implement anXGBoost modelon a givendatasetand useSHAP valuesto interpret the model's predictions.**

**Answer:**

Here's a comprehensive implementation using XGBoost with SHAP (SHapley Additive exPlanations) for model interpretation:

```python
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')

# Initialize SHAP JavaScript visualizations for notebooks
shap.initjs()

class XGBoostSHAPAnalyzer:
    """
    Comprehensive XGBoost model with SHAP interpretation
    
    This class combines XGBoost modeling with SHAP analysis for 
    complete model interpretability and prediction explanation.
    """
    
    def __init__(self, task_type='classification', random_state=42):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        task_type : str, default='classification'
            Type of ML task: 'classification' or 'regression'
        random_state : int, default=42
            Random state for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self, dataset='breast_cancer'):
        """
        Load and prepare sample datasets
        
        Parameters:
        -----------
        dataset : str, default='breast_cancer'
            Dataset to load: 'breast_cancer', 'diabetes'
        
        Returns:
        --------
        tuple: Prepared train/test splits
        """
        if dataset == 'breast_cancer':
            data = load_breast_cancer()
            self.task_type = 'classification'
        elif dataset == 'diabetes':
            data = load_diabetes()
            self.task_type = 'regression'
        else:
            raise ValueError("Supported datasets: 'breast_cancer', 'diabetes'")
        
        X, y = data.data, data.target
        self.feature_names = data.feature_names
        
        # Split the data
        if self.task_type == 'classification':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
        
        print(f"Dataset: {dataset}")
        print(f"Task type: {self.task_type}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_xgboost_model(self, **model_params):
        """
        Train XGBoost model with specified parameters
        
        Parameters:
        -----------
        **model_params : Additional parameters for XGBoost
        """
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state
        }
        
        # Update with user parameters
        default_params.update(model_params)
        
        # Initialize model based on task type
        if self.task_type == 'classification':
            self.model = xgb.XGBClassifier(**default_params)
        else:
            self.model = xgb.XGBRegressor(**default_params)
        
        # Train the model
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            verbose=False
        )
        
        # Evaluate model
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        if self.task_type == 'classification':
            train_score = accuracy_score(self.y_train, train_pred)
            test_score = accuracy_score(self.y_test, test_pred)
            print(f"Training Accuracy: {train_score:.4f}")
            print(f"Test Accuracy: {test_score:.4f}")
            
            # Detailed classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, test_pred))
            
        else:
            train_score = mean_squared_error(self.y_train, train_pred, squared=False)
            test_score = mean_squared_error(self.y_test, test_pred, squared=False)
            print(f"Training RMSE: {train_score:.4f}")
            print(f"Test RMSE: {test_score:.4f}")
        
        print("XGBoost model training completed!")
    
    def initialize_shap_explainer(self, explainer_type='tree'):
        """
        Initialize SHAP explainer for the trained model
        
        Parameters:
        -----------
        explainer_type : str, default='tree'
            Type of SHAP explainer: 'tree', 'kernel'
        """
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        print(f"Initializing SHAP {explainer_type} explainer...")
        
        if explainer_type == 'tree':
            # TreeExplainer - most efficient for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            # KernelExplainer - model-agnostic but slower
            self.explainer = shap.KernelExplainer(
                self.model.predict, shap.sample(self.X_train, 100)
            )
        else:
            raise ValueError("Supported explainers: 'tree', 'kernel'")
        
        print("SHAP explainer initialized!")
    
    def calculate_shap_values(self, data='test', max_samples=None):
        """
        Calculate SHAP values for the specified data
        
        Parameters:
        -----------
        data : str or array-like, default='test'
            Data to explain: 'test', 'train', or custom array
        max_samples : int, optional
            Maximum number of samples to explain (for efficiency)
        """
        if self.explainer is None:
            self.initialize_shap_explainer()
        
        # Select data to explain
        if data == 'test':
            X_explain = self.X_test
        elif data == 'train':
            X_explain = self.X_train
        else:
            X_explain = data
        
        # Limit samples for efficiency
        if max_samples is not None and len(X_explain) > max_samples:
            indices = np.random.choice(len(X_explain), max_samples, replace=False)
            X_explain = X_explain[indices]
            print(f"Explaining {max_samples} random samples...")
        else:
            print(f"Explaining {len(X_explain)} samples...")
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_explain)
        self.X_explain = X_explain
        
        print("SHAP values calculated!")
        return self.shap_values
    
    def plot_shap_summary(self, plot_type='dot', max_display=15):
        """
        Create SHAP summary plots
        
        Parameters:
        -----------
        plot_type : str, default='dot'
            Type of plot: 'dot', 'bar'
        max_display : int, default=15
            Maximum number of features to display
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        print(f"Creating SHAP {plot_type} summary plot...")
        
        # For multi-class classification, use first class
        shap_vals = self.shap_values
        if isinstance(self.shap_values, list) and len(self.shap_values) > 1:
            shap_vals = self.shap_values[1]  # Positive class for binary classification
        
        plt.figure(figsize=(10, 8))
        if plot_type == 'dot':
            shap.summary_plot(
                shap_vals, self.X_explain, 
                feature_names=self.feature_names,
                max_display=max_display, show=False
            )
            plt.title('SHAP Feature Importance (Dot Plot)')
            
        elif plot_type == 'bar':
            shap.summary_plot(
                shap_vals, self.X_explain,
                feature_names=self.feature_names,
                plot_type='bar', max_display=max_display, show=False
            )
            plt.title('SHAP Feature Importance (Bar Plot)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, sample_idx=0):
        """
        Create SHAP waterfall plot for individual prediction
        
        Parameters:
        -----------
        sample_idx : int, default=0
            Index of sample to explain
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        print(f"Creating SHAP waterfall plot for sample {sample_idx}...")
        
        # For multi-class, use positive class
        shap_vals = self.shap_values
        expected_value = self.explainer.expected_value
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
            expected_value = expected_value[1] if isinstance(expected_value, np.ndarray) else expected_value
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[sample_idx],
                base_values=expected_value,
                data=self.X_explain[sample_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}')
        plt.tight_layout()
        plt.show()
    
    def analyze_global_feature_importance(self):
        """
        Analyze global feature importance using SHAP values
        
        Returns:
        --------
        pd.DataFrame: Global importance analysis
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # For multi-class, use positive class
        shap_vals = self.shap_values
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        
        # Calculate global importance metrics
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        mean_shap = shap_vals.mean(axis=0)
        std_shap = shap_vals.std(axis=0)
        
        # Create analysis DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': mean_shap,
            'std_shap': std_shap
        })
        
        # Sort by absolute importance
        importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        print("Global SHAP Feature Importance Analysis:")
        print("=" * 50)
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:25s} "
                  f"| Mean |SHAP|: {row['mean_abs_shap']:8.4f} "
                  f"| Mean SHAP: {row['mean_shap']:8.4f}")
        
        return importance_df
    
    def create_comprehensive_report(self, save_plots=False):
        """
        Create comprehensive SHAP analysis report
        """
        print("Creating comprehensive SHAP analysis report...")
        print("=" * 50)
        
        # 1. Calculate SHAP values if not done
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # 2. Global importance analysis
        importance_df = self.analyze_global_feature_importance()
        
        # 3. Summary plots
        self.plot_shap_summary(plot_type='dot')
        self.plot_shap_summary(plot_type='bar')
        
        # 4. Individual prediction explanations
        for i in range(min(2, len(self.X_explain))):
            self.plot_shap_waterfall(sample_idx=i)
        
        print(f"\nComprehensive SHAP analysis completed!")
        return importance_df

def demonstrate_xgboost_shap_analysis():
    """
    Complete demonstration of XGBoost with SHAP analysis
    """
    print("=== XGBoost + SHAP Analysis Demonstration ===\n")
    
    # Demo 1: Classification task
    print("DEMO 1: Classification Analysis (Breast Cancer Dataset)")
    print("-" * 60)
    
    analyzer = XGBoostSHAPAnalyzer(task_type='classification')
    
    # Load and prepare data
    analyzer.load_and_prepare_data('breast_cancer')
    
    # Train XGBoost model
    analyzer.train_xgboost_model(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1
    )
    
    # Comprehensive SHAP analysis
    importance_df = analyzer.create_comprehensive_report()
    
    print("\n" + "="*60 + "\n")
    
    # Demo 2: Regression task
    print("DEMO 2: Regression Analysis (Diabetes Dataset)")
    print("-" * 60)
    
    analyzer_reg = XGBoostSHAPAnalyzer(task_type='regression')
    
    # Load and prepare data
    analyzer_reg.load_and_prepare_data('diabetes')
    
    # Train XGBoost model
    analyzer_reg.train_xgboost_model(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    # SHAP analysis
    analyzer_reg.create_comprehensive_report()
    
    return analyzer, analyzer_reg

# Quick function for custom data analysis
def quick_shap_analysis(X, y, feature_names=None, task_type='classification'):
    """
    Quick SHAP analysis for custom data
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    feature_names : list, optional
        Feature names
    task_type : str, default='classification'
        Task type
    
    Returns:
    --------
    XGBoostSHAPAnalyzer: Configured analyzer object
    """
    analyzer = XGBoostSHAPAnalyzer(task_type=task_type)
    
    # Set data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == 'classification' else None
    )
    
    analyzer.X_train = X_train
    analyzer.X_test = X_test
    analyzer.y_train = y_train
    analyzer.y_test = y_test
    
    if feature_names is not None:
        analyzer.feature_names = feature_names
    else:
        analyzer.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train and analyze
    analyzer.train_xgboost_model()
    importance_df = analyzer.create_comprehensive_report()
    
    return analyzer

# Run the demonstration
if __name__ == "__main__":
    try:
        analyzer_clf, analyzer_reg = demonstrate_xgboost_shap_analysis()
        
        print("\n" + "="*60)
        print("XGBoost + SHAP Analysis Complete!")
        print("="*60)
        
    except ImportError as e:
        print("Error: SHAP library not installed.")
        print("Install with: pip install shap")
        print("Then run the analysis again.")
```

**Key Features of this Implementation:**

1. **Complete SHAP Integration:**
   - TreeExplainer for XGBoost models (most efficient)
   - Global and local interpretability
   - Multiple visualization types

2. **Comprehensive Analysis:**
   - Summary plots (dot and bar) showing feature impact
   - Waterfall plots for individual predictions
   - Global importance rankings
   - Feature contribution analysis

3. **Both Classification and Regression:**
   - Handles binary/multi-class classification
   - Supports regression tasks
   - Automatic task detection

4. **Production-Ready Features:**
   - Automated report generation
   - Efficient sampling for large datasets
   - Error handling and validation

**SHAP Value Interpretation:**
- **Positive SHAP values:** Push prediction above baseline (expected value)
- **Negative SHAP values:** Push prediction below baseline
- **Magnitude:** Indicates strength of feature impact
- **Additivity:** Sum of SHAP values + baseline = model prediction

**Usage Benefits:**
- **Model Debugging:** Identify unexpected feature behavior
- **Feature Selection:** Remove low-impact features
- **Stakeholder Communication:** Visual explanations
- **Regulatory Compliance:** Explainable AI requirements
- **Trust and Transparency:** Understand model decisions

**Installation Requirements:**
```bash
pip install shap xgboost scikit-learn matplotlib pandas numpy
```
