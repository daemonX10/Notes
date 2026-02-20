# 01_python_ml — Cross-Reference Audit Report

## Executive Summary

| Metric | Count |
|--------|-------|
| PythonMl.md total questions | **100** |
| PythonMl questions covered by category files | **100 / 100 (100%)** |
| Correctly placed questions (with answers) | **103** (across 4 files) |
| **MISPLACED questions (FastAI)** | **50** (in theory_questions.md) |
| Duplicate stub questions | **44** (across 3 files) |

### Verdict
All 100 PythonMl.md questions are properly covered. However, **theory_questions.md is severely contaminated** with 50 FastAI-specific questions and 35 duplicate stubs that do not belong. Two other files also contain smaller numbers of duplicate stubs.

---

## 1. File-by-File Analysis

### 1A. theory_questions.md (6,383 lines)

**Legitimate PythonML Questions (Q1–Q40) — 40 questions, ALL CORRECT ✅**

| theory Q# | Question Title | PythonMl Q# |
|-----------|---------------|-------------|
| Q1 | Explain difference between Python 2 and Python 3 | Q1 |
| Q2 | How does Python manage memory? | Q2 |
| Q3 | What is PEP 8 and why is it important? | Q3 |
| Q4 | Describe how a dictionary works in Python | Q5 |
| Q5 | What is list comprehension? | Q6 |
| Q6 | Explain the concept of generators | Q7 |
| Q7 | How does Python's garbage collection work? | Q9 |
| Q8 | What are decorators? | Q10 |
| Q9 | What is NumPy and how is it useful in ML? | Q12 |
| Q10 | How does Scikit-learn fit into the ML workflow? | Q14 |
| Q11 | Explain Matplotlib and Seaborn | Q15 |
| Q12 | What is TensorFlow and Keras? | Q18 |
| Q13 | Explain the process of data cleaning | Q19 |
| Q14 | Common steps in data preprocessing | Q20 |
| Q15 | Feature scaling and why it is necessary | Q22 |
| Q16 | Label encoding vs one-hot encoding | Q23 |
| Q17 | Data splitting into train/validation/test | Q24 |
| Q18 | Building a ML model in Python | Q27 |
| Q19 | Cross-validation | Q29 |
| Q20 | Bias-variance tradeoff | Q30 |
| Q21 | Steps to improve model accuracy | Q31 |
| Q22 | Hyperparameters and tuning | Q32 |
| Q23 | Confusion matrix | Q36 |
| Q24 | ROC curve and AUC | Q38 |
| Q25 | Validation strategies (k-fold) | Q41 |
| Q26 | Model performs well on training but poorly on new data | Q51 |
| Q27 | Regularization in linear models | Q54 |
| Q28 | SGD vs standard gradient descent | Q55 |
| Q29 | Dimensionality reduction | Q59 |
| Q30 | Batch learning vs online learning | Q64 |
| Q31 | Attention mechanisms in NLP | Q65 |
| Q32 | Context managers | Q75 |
| Q33 | Slots in Python classes | Q76 |
| Q34 | Microservices architecture | Q78 |
| Q35 | Scaling a ML application | Q79 |
| Q36 | Model versioning | Q82 |
| Q37 | ML model failure investigation | Q83 |
| Q38 | Python profiling tools | Q85 |
| Q39 | Unit tests and integration tests | Q86 |
| Q40 | Explainable AI (XAI) | Q97 |

**❌ MISPLACED: FastAI Questions (50 questions) — DO NOT BELONG**

These appear after Q40 in theory_questions.md. They are FastAI-specific and have **no match** in PythonMl.md. There is also **no FastAI topic folder** anywhere in the workspace (checked `Part 1-Foundations`, `Part 2-ML & DL Stack`, `Part 3-Big Data & Compute`).

| FastAI Q# | Question Title | Belongs To |
|-----------|---------------|------------|
| Q1 | Transfer learning capabilities for computer vision | **No topic file exists** |
| Q2 | Data loading and augmentation pipelines | **No topic file exists** |
| Q3 | Model fine-tuning strategies | **No topic file exists** |
| Q4 | FastAI high-level API vs PyTorch lower-level | **No topic file exists** |
| Q5 | Mixed precision and distributed training | **No topic file exists** |
| Q6 | MLOps and experiment tracking integration | **No topic file exists** |
| Q7 | Model deployment and production serving | **No topic file exists** |
| Q8 | Tabular learning capabilities | **No topic file exists** |
| Q9 | Custom loss functions and metrics | **No topic file exists** |
| Q10 | Model interpretability and explanation workflows | **No topic file exists** |
| Q11 | Data sources and preprocessing pipelines | **No topic file exists** |
| Q12 | Learning rate finding and scheduling techniques | **No topic file exists** |
| Q13 | Model ensembling and prediction combination | **No topic file exists** |
| Q14 | Memory usage and computational efficiency | **No topic file exists** |
| Q15 | NLP tasks and text classification | **No topic file exists** |
| Q16 | Custom architectures vs pre-built models | **No topic file exists** |
| Q17 | Model validation and cross-validation strategies | **No topic file exists** |
| Q18 | Hyperparameter tuning and optimization | **No topic file exists** |
| Q19 | Cloud platforms and distributed computing | **No topic file exists** |
| Q20 | Progressive resizing and training curriculum | **No topic file exists** |
| Q21 | Hardware configurations and acceleration | **No topic file exists** |
| Q22 | Model versioning and reproducibility | **No topic file exists** |
| Q23 | Real-time inference and streaming | **No topic file exists** |
| Q24 | DataBlock API vs PyTorch DataLoader | **No topic file exists** |
| Q25 | Model compression and quantization for edge | **No topic file exists** |
| Q26 | Automated machine learning workflows | **No topic file exists** |
| Q27 | Medical imaging and healthcare applications | **No topic file exists** |
| Q28 | Combining FastAI with other frameworks | **No topic file exists** |
| Q29 | Custom data transformations | **No topic file exists** |
| Q30 | Model security and privacy in production | **No topic file exists** |
| Q31 | Feature engineering and selection pipelines | **No topic file exists** |
| Q32 | Mixed precision vs full precision training | **No topic file exists** |
| Q33 | Model monitoring and performance tracking | **No topic file exists** |
| Q34 | Complexity management in large-scale projects | **No topic file exists** |
| Q35 | Data versioning and pipeline orchestration | **No topic file exists** |
| Q36 | Callback system vs custom training loops | **No topic file exists** |
| Q37 | Model testing and validation procedures | **No topic file exists** |
| Q38 | Domain-specific optimization (finance, retail, mfg) | **No topic file exists** |
| Q39 | Reinforcement learning with FastAI | **No topic file exists** |
| Q40 | Custom optimizers vs built-in algorithms | **No topic file exists** |
| Q41 | Business intelligence and reporting integration | **No topic file exists** |
| Q42 | Code organization and project structure | **No topic file exists** |
| Q43 | Accessibility and inclusive AI development | **No topic file exists** |
| Q45 | Collaborative development and team workflows | **No topic file exists** |
| Q46 | Licensing and intellectual property | **No topic file exists** |
| Q47 | Custom evaluation metrics and model selection | **No topic file exists** |
| Q48 | Experimental features vs stable APIs | **No topic file exists** |
| Q49 | Educational purposes and curriculum development | **No topic file exists** |
| Q50 | Continuous learning and model updating | **No topic file exists** |

> **Note:** FastAI Q44 appears to be missing/skipped in the file.

**⚠️ DUPLICATE STUBS: Questions Q51–Q85 (35 questions) — "Answer to be added"**

These are exact title duplicates of theory Q1–Q40 with empty placeholder answers.

| Stub Q# | Duplicate Of | PythonMl Q# |
|---------|-------------|-------------|
| Q51 | theory Q1 | Q1 |
| Q52 | theory Q2 | Q2 |
| Q53 | theory Q3 | Q3 |
| Q54 | theory Q4 | Q5 |
| Q55 | theory Q5 | Q6 |
| Q56 | theory Q6 | Q7 |
| Q57 | (new) *args/**kwargs | Q8 |
| Q58 | theory Q7 | Q9 |
| Q59 | theory Q8 | Q10 |
| Q60 | theory Q9 | Q12 |
| Q61 | theory Q10 | Q14 |
| Q62 | theory Q11 | Q15 |
| Q63 | theory Q12 | Q18 |
| Q64 | theory Q13 | Q19 |
| Q65 | theory Q14 | Q20 |
| Q66 | theory Q15 | Q22 |
| Q67 | theory Q16 | Q23 |
| Q68 | theory Q17 | Q24 |
| Q69 | theory Q18 | Q27 |
| Q70 | theory Q19 | Q29 |
| Q71 | theory Q20 | Q30 |
| Q72 | theory Q21 | Q31 |
| Q73 | theory Q22 | Q32 |
| Q74 | theory Q23 | Q36 |
| Q75 | theory Q24 | Q38 |
| Q76 | theory Q25 | Q41 |
| Q77 | theory Q29 | Q59 |
| Q78 | theory Q30 | Q64 |
| Q79 | theory Q31 | Q65 |
| Q80 | theory Q32 | Q75 |
| Q81 | theory Q33 | Q76 |
| Q82 | theory Q34 | Q78 |
| Q83 | theory Q35 | Q79 |
| Q84 | theory Q36 | Q82 |
| Q85 | theory Q40 | Q97 |

---

### 1B. coding_questions.md (3,050 lines)

**Legitimate PythonML Questions (Q1–Q32) — 32 questions, ALL CORRECT ✅**

| coding Q# | Question Title | PythonMl Q# |
|-----------|---------------|-------------|
| Q1 | Gradient descent algorithm | Q34 |
| Q2 | K-Means clustering from scratch | Q44 |
| Q3 | Train-test split from scratch | Q49 |
| Q4 | Standardization (Z-score) from scratch | Q22 (implementation) |
| Q5 | Decision tree classifier from scratch | Q48 |
| Q6 | One-Hot Encoding from scratch | Q23 (implementation) |
| Q7 | Accuracy, precision, recall, F1 from scratch | Q35/Q37 (implementation) |
| Q8 | Normalize array to [0,1] | Q42 |
| Q9 | Perceptron model | Q43 |
| Q10 | Linear regression with NumPy | Q45 |
| Q11 | Optimize cost function using gradient descent | Q46 |
| Q12 | Pandas CSV clean data | Q47 |
| Q13 | Hyperparameter tuning grid search | Q50 |
| Q14 | Neural network implementation | Q60 |
| Q15 | Reinforcement learning | Q61 |
| Q16 | Transfer learning | Q63 |
| Q17 | Recommendation system | Q66 |
| Q18 | Spam detection system | Q67 |
| Q19 | House prices prediction | Q68 |
| Q20 | Sentiment analysis model | Q69 |
| Q21 | Customer churn prediction | Q70 |
| Q22 | Image classification system | Q71 |
| Q23 | Fraud detection | Q72 |
| Q24 | Data batch generator | Q88 |
| Q25 | CNN implementation (PyTorch) | Q89 |
| Q26 | Genetic algorithms optimization | Q90 |
| Q27 | Optimization techniques comparison | Q91 |
| Q28 | Decision boundaries visualization | Q92 |
| Q29 | A* search algorithm | Q93 |
| Q30 | RL agent for basic game | Q94 |
| Q31 | Time-series forecasting | Q95 |
| Q32 | Federated learning implementation | Q100 |

**⚠️ DUPLICATE STUBS: Questions Q33–Q35 (3 questions) — "Answer to be added"**

| Stub Q# | Duplicate Of | PythonMl Q# |
|---------|-------------|-------------|
| Q33 | coding Q2 (K-Means) | Q44 |
| Q34 | coding Q5 (Decision tree) | Q48 |
| Q35 | coding Q3 (Train-test split) | Q49 |

---

### 1C. general_questions.md (895 lines)

**15 questions — ALL CORRECT ✅, NO ISSUES**

| general Q# | Question Title | PythonMl Q# |
|------------|---------------|-------------|
| Q1 | Name key Python libraries for ML | Q11 |
| Q2 | Describe Pandas overview | Q13 |
| Q3 | Compare SciPy vs NumPy | Q16 |
| Q4 | Handle missing/corrupted data | Q21 |
| Q5 | Handle categorical data | Q26 |
| Q6 | Ensure model not overfitting | Q28 |
| Q7 | Precision and recall | Q37 |
| Q8 | Learning curve diagnosis | Q40 |
| Q9 | Parallelize computations in Python | Q56 |
| Q10 | Logistic regression coefficients | Q58 |
| Q11 | GANs | Q62 |
| Q12 | Python scopes (global/nonlocal/local) | Q73 |
| Q13 | Docker containerization for ML | Q81 |
| Q14 | Exception handling in deployment | Q87 |
| Q15 | Deep learning advancements in NLP | Q98 |

---

### 1D. scenario_based_questions.md (1,172 lines)

**Legitimate PythonML Questions (Q1–Q16) — 16 questions, ALL CORRECT ✅**

| scenario Q# | Question Title | PythonMl Q# |
|-------------|---------------|-------------|
| Q1 | Lists, tuples, sets (when to use which) | Q4 |
| Q2 | *args and **kwargs usage | Q8 |
| Q3 | Jupyter Notebooks benefits for ML | Q17 |
| Q4 | Scikit-learn pipelines for preprocessing | Q25 |
| Q5 | Ensemble methods | Q33 |
| Q6 | Assess model performance (metrics) | Q35 |
| Q7 | Supervised vs unsupervised evaluation | Q39 |
| Q8 | Feature selection techniques | Q52 |
| Q9 | Handle imbalanced datasets | Q53 |
| Q10 | Model persistence/serialization | Q57 |
| Q11 | GIL impact on ML | Q74 |
| Q12 | Collections module usage | Q77 |
| Q13 | ML model deployment options | Q80 |
| Q14 | Logging and monitoring in ML | Q84 |
| Q15 | Quantum computing and ML | Q96 |
| Q16 | Big data technologies for ML | Q99 |

**⚠️ DUPLICATE STUBS: Questions Q17–Q22 (6 questions) — "Answer to be added"**

| Stub Q# | Duplicate Of | PythonMl Q# |
|---------|-------------|-------------|
| Q17 | theory Q37 (model failure) | Q83 |
| Q18 | theory Q27 (regularization) | Q54 |
| Q19 | theory Q28 (SGD advantages) | Q55 |
| Q20 | theory Q37 (model failure, again) | Q83 |
| Q21 | theory Q38 (profiling tools) | Q85 |
| Q22 | theory Q39 (unit/integration tests) | Q86 |

---

## 2. PythonMl.md Coverage Map

Every PythonMl.md question and which category file covers it:

| PythonMl Q# | Question Title | Category File | Category Q# |
|-------------|---------------|---------------|-------------|
| Q1 | Python 2 vs Python 3 | theory | Q1 |
| Q2 | Python memory management | theory | Q2 |
| Q3 | PEP 8 importance | theory | Q3 |
| Q4 | Lists, tuples, sets | scenario | Q1 |
| Q5 | Dictionary (keys/values) | theory | Q4 |
| Q6 | List comprehension | theory | Q5 |
| Q7 | Generators | theory | Q6 |
| Q8 | *args and **kwargs | scenario | Q2 |
| Q9 | Garbage collection | theory | Q7 |
| Q10 | Decorators | theory | Q8 |
| Q11 | Key Python libraries for ML | general | Q1 |
| Q12 | NumPy for ML | theory | Q9 |
| Q13 | Pandas overview | general | Q2 |
| Q14 | Scikit-learn in ML workflow | theory | Q10 |
| Q15 | Matplotlib / Seaborn | theory | Q11 |
| Q16 | SciPy vs NumPy | general | Q3 |
| Q17 | Jupyter Notebooks | scenario | Q3 |
| Q18 | TensorFlow and Keras | theory | Q12 |
| Q19 | Data cleaning | theory | Q13 |
| Q20 | Data preprocessing steps | theory | Q14 |
| Q21 | Handle missing/corrupted data | general | Q4 |
| Q22 | Feature scaling | theory | Q15 |
| Q23 | Label vs one-hot encoding | theory | Q16 |
| Q24 | Train/val/test splitting | theory | Q17 |
| Q25 | Scikit-learn pipelines | scenario | Q4 |
| Q26 | Handle categorical data | general | Q5 |
| Q27 | Building a ML model | theory | Q18 |
| Q28 | Prevent overfitting | general | Q6 |
| Q29 | Cross-validation | theory | Q19 |
| Q30 | Bias-variance tradeoff | theory | Q20 |
| Q31 | Improve model accuracy | theory | Q21 |
| Q32 | Hyperparameters and tuning | theory | Q22 |
| Q33 | Ensemble methods | scenario | Q5 |
| Q34 | Gradient descent | coding | Q1 |
| Q35 | Model performance metrics | scenario | Q6 |
| Q36 | Confusion matrix | theory | Q23 |
| Q37 | Precision and recall | general | Q7 |
| Q38 | ROC curve and AUC | theory | Q24 |
| Q39 | Supervised vs unsupervised eval | scenario | Q7 |
| Q40 | Learning curves | general | Q8 |
| Q41 | Validation strategies | theory | Q25 |
| Q42 | Normalize array [0,1] | coding | Q8 |
| Q43 | Perceptron implementation | coding | Q9 |
| Q44 | K-Means from scratch | coding | Q2 |
| Q45 | Linear regression (NumPy) | coding | Q10 |
| Q46 | Gradient descent cost function | coding | Q11 |
| Q47 | Pandas CSV clean data | coding | Q12 |
| Q48 | Decision tree from scratch | coding | Q5 |
| Q49 | Train-test split from scratch | coding | Q3 |
| Q50 | Grid search hyperparameter tuning | coding | Q13 |
| Q51 | Overfitting on new data | theory | Q26 |
| Q52 | Feature selection | scenario | Q8 |
| Q53 | Imbalanced datasets | scenario | Q9 |
| Q54 | Regularization | theory | Q27 |
| Q55 | SGD vs standard GD | theory | Q28 |
| Q56 | Parallelize computations | general | Q9 |
| Q57 | Model persistence | scenario | Q10 |
| Q58 | Logistic regression coefficients | general | Q10 |
| Q59 | Dimensionality reduction | theory | Q29 |
| Q60 | Neural network implementation | coding | Q14 |
| Q61 | Reinforcement learning | coding | Q15 |
| Q62 | GANs | general | Q11 |
| Q63 | Transfer learning | coding | Q16 |
| Q64 | Batch vs online learning | theory | Q30 |
| Q65 | Attention mechanisms in NLP | theory | Q31 |
| Q66 | Recommendation system | coding | Q17 |
| Q67 | Spam detection system | coding | Q18 |
| Q68 | House prices prediction | coding | Q19 |
| Q69 | Sentiment analysis | coding | Q20 |
| Q70 | Customer churn prediction | coding | Q21 |
| Q71 | Image classification | coding | Q22 |
| Q72 | Fraud detection | coding | Q23 |
| Q73 | Python scopes | general | Q12 |
| Q74 | GIL impact | scenario | Q11 |
| Q75 | Context managers | theory | Q32 |
| Q76 | Slots in Python classes | theory | Q33 |
| Q77 | Collections module | scenario | Q12 |
| Q78 | Microservices architecture | theory | Q34 |
| Q79 | Scaling ML application | theory | Q35 |
| Q80 | ML model deployment options | scenario | Q13 |
| Q81 | Docker containerization | general | Q13 |
| Q82 | Model versioning | theory | Q36 |
| Q83 | ML model failure investigation | theory | Q37 |
| Q84 | Logging and monitoring | scenario | Q14 |
| Q85 | Python profiling tools | theory | Q38 |
| Q86 | Unit/integration tests | theory | Q39 |
| Q87 | Exception handling in deployment | general | Q14 |
| Q88 | Data batch generator | coding | Q24 |
| Q89 | CNN implementation | coding | Q25 |
| Q90 | Genetic algorithms | coding | Q26 |
| Q91 | Optimization comparison | coding | Q27 |
| Q92 | Decision boundaries visualization | coding | Q28 |
| Q93 | A* search algorithm | coding | Q29 |
| Q94 | RL agent for basic game | coding | Q30 |
| Q95 | Time-series forecasting | coding | Q31 |
| Q96 | Quantum computing and ML | scenario | Q15 |
| Q97 | Explainable AI (XAI) | theory | Q40 |
| Q98 | Deep learning in NLP | general | Q15 |
| Q99 | Big data technologies | scenario | Q16 |
| Q100 | Federated learning | coding | Q32 |

**Result: 100/100 questions covered ✅**

---

## 3. Issues & Recommended Actions

### Issue 1 — CRITICAL: 50 FastAI questions in theory_questions.md

**Location:** theory_questions.md, after the Q40 (XAI) answer block  
**Problem:** 50 fully-answered FastAI-specific questions are embedded in the PythonML theory file  
**Impact:** theory_questions.md is ~50% contaminated with off-topic content  
**Root cause:** No FastAI topic folder exists in the workspace; these questions have no home  

**Recommended action:**
1. Create a new folder: `Part 2-ML & DL Stack/XX_fastai/`
2. Create a `FastAI.md` topic file listing all 50 questions
3. Move the 50 FastAI questions into appropriate category files under that folder
4. Remove them from `01_python_ml/theory_questions.md`

---

### Issue 2 — MODERATE: 35 duplicate stubs in theory_questions.md

**Location:** theory_questions.md, Q51–Q85  
**Problem:** Exact title duplicates of Q1–Q40 with "Answer to be added" placeholders  
**Impact:** Inflates question count; confusing for readers  

**Recommended action:** Delete Q51–Q85 entirely (they duplicate already-answered questions)

---

### Issue 3 — MINOR: 3 duplicate stubs in coding_questions.md

**Location:** coding_questions.md, Q33–Q35  
**Problem:** Duplicates of Q2 (K-Means), Q5 (Decision tree), Q3 (Train-test split)  

**Recommended action:** Delete Q33–Q35

---

### Issue 4 — MINOR: 6 duplicate stubs in scenario_based_questions.md

**Location:** scenario_based_questions.md, Q17–Q22  
**Problem:** Duplicates of questions already answered in theory_questions.md  

**Recommended action:** Delete Q17–Q22

---

## 4. Cross-Reference Against Other Topics

The misplaced FastAI questions were checked against all other topic files:

| Topic File | Questions | FastAI Match? |
|-----------|-----------|---------------|
| SQL in ML.md (02_sql_ml) | 55 SQL questions | No — completely different domain |
| NumPy.md (03_numpy) | 70 NumPy questions | No — different library |
| Pandas.md (04_pandas) | 45 Pandas questions | No — different library |
| MATLAB.md (11_matlab) | 70 MATLAB questions | No — different language |
| Scikit-learn (05) | Not in workspace scope | — |
| TensorFlow (06) | Not in workspace scope | — |
| Keras (07) | Not in workspace scope | — |
| PyTorch (08) | Not in workspace scope | — |

**Conclusion:** The 50 FastAI questions belong to a topic that does not yet exist. A new `FastAI` topic folder needs to be created.

---

## 5. Statistics Summary

| File | Total Questions | Correctly Placed | Misplaced (FastAI) | Duplicate Stubs |
|------|----------------|-----------------|-------------------|----------------|
| theory_questions.md | ~125 | 40 | **50** | **35** |
| coding_questions.md | ~35 | 32 | 0 | **3** |
| general_questions.md | 15 | 15 | 0 | 0 |
| scenario_based_questions.md | ~22 | 16 | 0 | **6** |
| **TOTAL** | **~197** | **103** | **50** | **44** |

### Distribution of Correct Questions by Category

| Category | Count | PythonMl Sections Covered |
|----------|-------|--------------------------|
| theory | 40 | Basics, Libraries, Data Prep, Model Dev, Eval, Practical, Advanced Topics, Advanced Python, Scalability, Debugging, Trends |
| coding | 32 | Coding Challenges, Case Studies, Advanced Coding, some from other sections |
| general | 15 | Libraries, Data Prep, Model Dev, Eval, Practical, Advanced Topics, Advanced Python, Scalability, Debugging, Trends |
| scenario | 16 | Basics, Libraries, Data Prep, Model Dev, Eval, Practical, Advanced Python, Scalability, Debugging, Trends |
