# 🔍 AUDIT REPORT: 03_numpy Folder

**Generated:** 2026-02-20
**Source File:** NumPy.md (70 questions)
**Category Files Audited:** theory_questions.md, coding_questions.md, general_questions.md, scenario_based_questions.md
**Cross-Referenced Against:** PythonMl.md, SQL in ML.md, Pandas.md, MATLAB.md

---

## Executive Summary

| Metric | Count |
|--------|-------|
| Total questions in NumPy.md | 70 |
| Total questions across category files | 92 (28 theory + 38 coding + 11 general + 15 scenario) |
| Correctly mapped to NumPy.md | 82 |
| Extra questions (not in NumPy.md but valid NumPy) | 8 |
| WRONG topic questions (belong elsewhere) | 0 |
| Content mismatch (title ≠ content) | 1 |
| Stub answers ("Answer to be added") | 9 |
| Duplicate questions (across/within files) | 10 instances |
| Missing from NumPy.md (not in any file) | 0 (all 70 covered) |

---

## 1. theory_questions.md — 28 Questions

| # | Question Title | Status | NumPy.md # | Notes |
|---|---------------|--------|------------|-------|
| 1 | What is NumPy, and why is it important in Machine Learning? | ✅ CORRECT | #1 | Full answer |
| 2 | Explain how NumPy arrays are different from Python lists | ✅ CORRECT | #2 | Full answer |
| 3 | What are the main attributes of a NumPy ndarray? | ✅ CORRECT | #3 | Full answer |
| 4 | Explain the concept of broadcasting in NumPy | ✅ CORRECT | #5 | Full answer |
| 5 | What are the data types supported by NumPy arrays? | ✅ CORRECT | #6 | Full answer |
| 6 | What is the difference between a deep copy and a shallow copy in NumPy? | ✅ CORRECT | #8 | Full answer |
| 7 | What are universal functions (ufuncs) in NumPy? | ✅ CORRECT | #10 | Full answer |
| 8 | What is the use of the axis parameter in NumPy functions? | ✅ CORRECT | #14 | Full answer |
| 9 | Explain the use of slicing and indexing with NumPy arrays | ✅ CORRECT | #18 | Full answer |
| 10 | What is the purpose of the NumPy histogram function? | ✅ CORRECT | #23 | Full answer |
| 11 | What is the difference between np.var() and np.std()? | ✅ CORRECT | #26 | Full answer |
| 12 | What is the concept of vectorization in NumPy? | ✅ CORRECT | #30 | Full answer |
| 13 | Explain the term "stride" in the context of NumPy arrays | ✅ CORRECT | #31 | Full answer |
| 14 | How does NumPy handle data types to optimize memory use? | ✅ CORRECT | #34 | Full answer |
| 15 | What are NumPy strides, and how do they affect array manipulation? | ✅ CORRECT | #35 | Full answer |
| 16 | Explain the concept and use of masked arrays in NumPy | ✅ CORRECT | #37 | Full answer |
| 17 | What are the functions available for padding arrays in NumPy? | ✅ CORRECT | #41 | Full answer |
| 18 | Describe how you can use NumPy for simulating Monte Carlo experiments | ✅ CORRECT | #54 | Full answer |
| 19 | Explain how to resolve the MemoryError when working with very large arrays in NumPy | ✅ CORRECT | #62 | Full answer |
| 20 | What are NumPy "polynomial" objects and how are they used? | ✅ CORRECT | #63 | Full answer |
| 21 | How does the internal C-API contribute to NumPy's performance? | ✅ CORRECT | #64 | Full answer |
| 22 | Explain the concept of a stride trick in NumPy | ✅ CORRECT | #65 | Full answer |
| 23 | What is the role of the NumPy nditer object? | ✅ CORRECT | #67 | Full answer |
| 24 | Explain how NumPy integrates with other Python libraries like Pandas and Matplotlib | ✅ CORRECT | #68 | Full answer |
| 25 | Describe how NumPy can be used with JAX for accelerated ML computation | ✅ CORRECT | #70 | Full answer |
| 26 | Why is NumPy more efficient for numerical computations than pure Python? | ⚠️ STUB | #29 | "Answer to be added" — duplicate of general Q4 (which has full answer) |
| 27 | Discuss the performance benefits of using NumPy's in-place operations | ⚠️ STUB | #33 | "Answer to be added" — duplicate of scenario Q11 (which has full answer) |
| 28 | Discuss the use of NumPy for operations on polynomials | ⚠️ STUB | #66 | "Answer to be added" — duplicate of scenario Q15 (which has full answer) |

**Summary:** 25 fully answered, 3 stubs. All 28 correctly belong to NumPy.md. No wrong-topic questions.

---

## 2. coding_questions.md — 38 Questions

| # | Question Title | Status | NumPy.md # | Notes |
|---|---------------|--------|------------|-------|
| 1 | How do you create a NumPy array from a regular Python list? | ✅ CORRECT | #4 | Full answer |
| 2 | How do you reshape a NumPy array? | ➕ EXTRA | — | Valid NumPy topic, but NOT a question in NumPy.md |
| 3 | How do you slice and index NumPy arrays? | ✅ CORRECT | #18 | Coding version of theory Q9; overlaps |
| 4 | How do you perform matrix multiplication using NumPy? | ✅ CORRECT | #11 | Full answer |
| 5 | How do you concatenate and stack arrays? | ✅ CORRECT | #15 + #38 | Combines two NumPy.md questions into one |
| 6 | How do you generate random numbers with NumPy? | ✅ CORRECT | #22 | Full answer |
| 7 | How do you find unique values and their counts? | ✅ CORRECT | #60 | Full answer; **duplicate** of general Q10 |
| 8 | How do you normalize an array in NumPy? | 🔴 CONTENT MISMATCH | #24 (title) | **Title says "normalize" but content is about SORTING** (np.sort, argsort, axis sorting). The actual normalization content was split into Q9 and Q10. |
| 9 | How do you implement standardization (Z-score normalization)? | ➕ EXTRA | — | Valid NumPy topic, specific sub-topic of #24. Not an explicit NumPy.md question title. |
| 10 | How do you implement Min-Max normalization? | ➕ EXTRA | — | Valid NumPy topic, specific sub-topic of #24. Not an explicit NumPy.md question title. |
| 11 | How do you compute the dot product and cross product? | ➕ EXTRA | — | Valid NumPy topic, but NOT a question in NumPy.md |
| 12 | How do you compute eigenvalues and eigenvectors? | ✅ CORRECT | #17 | Full answer |
| 13 | Explain how to invert a matrix in NumPy | ✅ CORRECT | #12 | Full answer |
| 14 | How do you calculate the determinant of a matrix? | ✅ CORRECT | #13 | Full answer |
| 15 | Describe how you would flatten a multi-dimensional array | ✅ CORRECT | #16 | Full answer |
| 16 | How can you reverse an array in NumPy? | ✅ CORRECT | #19 | Full answer |
| 17 | How do you apply a conditional filter to a NumPy array? | ✅ CORRECT | #20 | Full answer |
| 18 | How can you compute percentiles with NumPy? | ✅ CORRECT | #25 | Full answer |
| 19 | How do you calculate the correlation coefficient using NumPy? | ✅ CORRECT | #27 | Full answer |
| 20 | Explain the use of the np.cumsum() and np.cumprod() functions | ✅ CORRECT | #28 | Full answer |
| 21 | Describe the process for creating a structured array in NumPy | ✅ CORRECT | #39 | Full answer |
| 22 | How do you save and load NumPy arrays to and from disk? | ✅ CORRECT | #40 | Full answer |
| 23 | Write a NumPy code to create a 3x3 identity matrix | ✅ CORRECT | #42 | Full answer |
| 24 | Code a function in NumPy to compute the moving average of a 1D array | ✅ CORRECT | #43 | Full answer |
| 25 | Generate a 2D NumPy array of random integers and normalize it between 0 and 1 | ✅ CORRECT | #44 | Full answer |
| 26 | Create a NumPy code snippet to extract all odd numbers from an array | ✅ CORRECT | #45 | Full answer |
| 27 | Implement a routine to calculate the outer product of two vectors in NumPy | ✅ CORRECT | #46 | Full answer |
| 28 | Write a NumPy program to create a checkerboard 8x8 matrix using the tile function | ✅ CORRECT | #47 | Full answer |
| 29 | Code a NumPy snippet to create a border around an existing array | ✅ CORRECT | #48 | Full answer |
| 30 | Write a function to compute the convolution of two matrices in NumPy | ✅ CORRECT | #49 | Full answer |
| 31 | Implement a script that computes the Fibonacci sequence using a NumPy matrix | ✅ CORRECT | #50 | Full answer |
| 32 | Write a code to replace all elements greater than a certain threshold | ✅ CORRECT | #51 | Full answer |
| 33 | Implement an efficient rolling window calculation for a 1D array using NumPy | ✅ CORRECT | #52 | Full answer |
| 34 | Explain how you would implement gradient descent optimization with NumPy | ✅ CORRECT | #56 | Full answer |
| 35 | How do you concatenate two arrays in NumPy? | ⚠️ STUB | #15 | "Answer to be added" — overlaps with Q5 |
| 36 | How do you calculate the eigenvalues and eigenvectors of a matrix in NumPy? | ⚠️ STUB | #17 | "Answer to be added" — **duplicate** of Q12 (which has full answer) |
| 37 | Explain how to generate random data with NumPy | ⚠️ STUB | #22 | "Answer to be added" — **duplicate** of Q6 (which has full answer) |
| 38 | How do you stack multiple arrays vertically and horizontally? | ⚠️ STUB | #38 | "Answer to be added" — overlaps with Q5 |

**Summary:** 29 fully answered, 4 stubs, 4 extra (valid NumPy), 1 content mismatch. No wrong-topic questions.

---

## 3. general_questions.md — 11 Questions

| # | Question Title | Status | NumPy.md # | Notes |
|---|---------------|--------|------------|-------|
| 1 | How do you inspect the shape and size of a NumPy array? | ✅ CORRECT | #7 | Full answer |
| 2 | How do you perform element-wise operations in NumPy? | ✅ CORRECT | #9 | Full answer |
| 3 | How do you compute the mean, median, and standard deviation with NumPy? | ✅ CORRECT | #21 | Full answer |
| 4 | Why is NumPy more efficient than pure Python for numerical computations? | ✅ CORRECT | #29 | Full answer |
| 5 | How do you check the memory size of a NumPy array? | ✅ CORRECT | #32 | Full answer |
| 6 | How do you handle NaN or infinite values in a NumPy array? | ✅ CORRECT | #58 | Full answer |
| 7 | How do you create a record array in NumPy? | ✅ CORRECT | #36 | Full answer |
| 8 | How can NumPy be used for audio signal processing? | ✅ CORRECT | #55 | Full answer |
| 9 | What methods are there in NumPy to deal with missing data? | ✅ CORRECT | #59 | Full answer |
| 10 | How do you find unique values and their counts in a NumPy array? | ✅ CORRECT | #60 | Full answer; **duplicate** of coding Q7 |
| 11 | How can you use NumPy arrays with Cython for performance optimization? | ✅ CORRECT | #69 | Full answer |

**Summary:** 11/11 fully answered. All correctly belong to NumPy.md. No issues.

---

## 4. scenario_based_questions.md — 15 Scenarios + 3 Stubs

| # | Question Title | Status | NumPy.md # | Notes |
|---|---------------|--------|------------|-------|
| S1 | Z-score standardization on a feature matrix for neural network | ➕ EXTRA | ~#24 | Original scenario; valid applied NumPy |
| S2 | Cap extreme values at 5th and 95th percentiles | ➕ EXTRA | ~#25 | Original scenario; valid applied NumPy |
| S3 | Implement simple linear regression (closed-form solution) | ➕ EXTRA | — | Original scenario; valid applied NumPy |
| S4 | Implement one-hot encoding using only NumPy | ➕ EXTRA | — | Original scenario; valid applied NumPy |
| S5 | Implement softmax function using NumPy | ➕ EXTRA | — | Original scenario; valid applied NumPy |
| S6 | Calculate cosine similarity between two vectors | ➕ EXTRA | — | Original scenario; valid applied NumPy |
| S7 | K-means clustering: calculate distances to centroids | ➕ EXTRA | — | Original scenario; valid applied NumPy |
| S8 | Implement train/test split using only NumPy | ➕ EXTRA | — | Original scenario; valid applied NumPy |
| S9 | Implement batch gradient descent for linear regression | ✅ CORRECT | #56 | Full answer; overlaps with coding Q34 |
| S10 | Calculate precision, recall, F1-score using NumPy | ➕ EXTRA | — | Original scenario; valid applied NumPy |
| S11 | Discuss performance benefits of NumPy's in-place operations | ✅ CORRECT | #33 | Full answer |
| S12 | How would you use NumPy to process image data for a CNN? | ✅ CORRECT | #53 | Full answer |
| S13 | Discuss the role of NumPy in managing data for training an ML model | ✅ CORRECT | #57 | Full answer |
| S14 | Discuss potential issues when importing large datasets into NumPy arrays | ✅ CORRECT | #61 | Full answer |
| S15 | Discuss the use of NumPy for operations on polynomials | ✅ CORRECT | #66 | Full answer |
| Stub Q1 | How would you use NumPy to process image data for a CNN? | ⚠️ STUB | #53 | **Duplicate** of S12 (which has full answer) |
| Stub Q2 | Discuss the role of NumPy in managing data for training an ML model | ⚠️ STUB | #57 | **Duplicate** of S13 (which has full answer) |
| Stub Q3 | Discuss potential issues when importing large datasets into NumPy arrays | ⚠️ STUB | #61 | **Duplicate** of S14 (which has full answer) |

**Summary:** 6 directly mapped + 9 original applied scenarios (all valid NumPy). 3 stubs that are duplicates of already-answered scenarios. No wrong-topic questions.

---

## 5. NumPy.md Coverage — All 70 Questions

Every NumPy.md question is covered by at least one category file:

| NumPy.md # | Question | Covered In |
|-----------|----------|-----------|
| 1 | What is NumPy, and why is it important in ML? | theory Q1 |
| 2 | NumPy arrays vs Python lists | theory Q2 |
| 3 | Main attributes of a NumPy ndarray | theory Q3 |
| 4 | Create a NumPy array from a Python list | coding Q1 |
| 5 | Broadcasting in NumPy | theory Q4 |
| 6 | Data types supported by NumPy arrays | theory Q5 |
| 7 | Inspect shape and size of a NumPy array | general Q1 |
| 8 | Deep copy vs shallow copy in NumPy | theory Q6 |
| 9 | Element-wise operations in NumPy | general Q2 |
| 10 | Universal functions (ufuncs) | theory Q7 |
| 11 | Matrix multiplication using NumPy | coding Q4 |
| 12 | Invert a matrix in NumPy | coding Q13 |
| 13 | Calculate the determinant of a matrix | coding Q14 |
| 14 | Use of the axis parameter | theory Q8 |
| 15 | Concatenate two arrays | coding Q5 (combined) + coding Q35 (stub) |
| 16 | Flatten a multi-dimensional array | coding Q15 |
| 17 | Eigenvalues and eigenvectors | coding Q12 + coding Q36 (stub) |
| 18 | Slicing and indexing with NumPy arrays | theory Q9 + coding Q3 |
| 19 | Reverse an array in NumPy | coding Q16 |
| 20 | Apply a conditional filter | coding Q17 |
| 21 | Mean, median, standard deviation | general Q3 |
| 22 | Generate random data | coding Q6 + coding Q37 (stub) |
| 23 | NumPy histogram function | theory Q10 |
| 24 | Normalize an array | coding Q8 (title only!), Q9, Q10 |
| 25 | Compute percentiles | coding Q18 |
| 26 | np.var() vs np.std() | theory Q11 |
| 27 | Correlation coefficient | coding Q19 |
| 28 | np.cumsum() and np.cumprod() | coding Q20 |
| 29 | Why NumPy more efficient than pure Python | general Q4 + theory Q26 (stub) |
| 30 | Vectorization in NumPy | theory Q12 |
| 31 | Stride in context of NumPy arrays | theory Q13 |
| 32 | Check memory size of a NumPy array | general Q5 |
| 33 | In-place operations performance benefits | scenario S11 + theory Q27 (stub) |
| 34 | NumPy handle data types to optimize memory | theory Q14 |
| 35 | NumPy strides and array manipulation | theory Q15 |
| 36 | Record array in NumPy | general Q7 |
| 37 | Masked arrays in NumPy | theory Q16 |
| 38 | Stack arrays vertically and horizontally | coding Q5 (combined) + coding Q38 (stub) |
| 39 | Structured array in NumPy | coding Q21 |
| 40 | Save and load NumPy arrays to/from disk | coding Q22 |
| 41 | Padding arrays in NumPy | theory Q17 |
| 42 | Create a 3x3 identity matrix | coding Q23 |
| 43 | Moving average of a 1D array | coding Q24 |
| 44 | 2D random integers, normalize 0-1 | coding Q25 |
| 45 | Extract all odd numbers from an array | coding Q26 |
| 46 | Outer product of two vectors | coding Q27 |
| 47 | Checkerboard 8x8 matrix using tile | coding Q28 |
| 48 | Create a border around an existing array | coding Q29 |
| 49 | Convolution of two matrices | coding Q30 |
| 50 | Fibonacci sequence using a NumPy matrix | coding Q31 |
| 51 | Replace elements greater than threshold | coding Q32 |
| 52 | Efficient rolling window calculation | coding Q33 |
| 53 | Process image data for a CNN | scenario S12 + scenario stub Q1 |
| 54 | Simulating Monte Carlo experiments | theory Q18 |
| 55 | Audio signal processing | general Q8 |
| 56 | Gradient descent optimization | coding Q34 + scenario S9 |
| 57 | Managing data for training an ML model | scenario S13 + scenario stub Q2 |
| 58 | Handle NaN or infinite values | general Q6 |
| 59 | Deal with missing data | general Q9 |
| 60 | Find unique values and their counts | general Q10 + coding Q7 |
| 61 | Issues importing large datasets | scenario S14 + scenario stub Q3 |
| 62 | Resolve MemoryError with very large arrays | theory Q19 |
| 63 | NumPy "polynomial" objects | theory Q20 |
| 64 | Internal C-API performance | theory Q21 |
| 65 | Stride trick in NumPy | theory Q22 |
| 66 | Operations on polynomials | scenario S15 + theory Q28 (stub) |
| 67 | NumPy nditer object | theory Q23 |
| 68 | NumPy integrates with Pandas and Matplotlib | theory Q24 |
| 69 | NumPy arrays with Cython | general Q11 |
| 70 | NumPy with JAX for accelerated ML | theory Q25 |

**Result: 0 missing questions. All 70 NumPy.md questions are covered.**

---

## 6. Issues Found

### 🔴 CRITICAL: Content Mismatch (1)

| File | Question | Issue |
|------|----------|-------|
| coding_questions.md Q8 | Title: "How do you normalize an array in NumPy?" | **Content is about SORTING** (np.sort, argsort, axis-based sorting). Has nothing to do with normalization. The actual normalization content was split into Q9 (Z-score) and Q10 (Min-Max). Title should be changed to "How do you sort a NumPy array?" or content should be replaced with normalization code. |

### ⚠️ Stub Questions — "Answer to be added" (9)

| File | Question # | Title | Has Full Answer Elsewhere? |
|------|-----------|-------|---------------------------|
| theory_questions.md | Q26 | Why is NumPy more efficient than pure Python? | ✅ Yes → general Q4 |
| theory_questions.md | Q27 | Performance benefits of in-place operations | ✅ Yes → scenario S11 |
| theory_questions.md | Q28 | Use of NumPy for operations on polynomials | ✅ Yes → scenario S15 |
| coding_questions.md | Q35 | How do you concatenate two arrays in NumPy? | ✅ Partially → coding Q5 (combined) |
| coding_questions.md | Q36 | Eigenvalues and eigenvectors | ✅ Yes → coding Q12 |
| coding_questions.md | Q37 | Generate random data with NumPy | ✅ Yes → coding Q6 |
| coding_questions.md | Q38 | Stack arrays vertically and horizontally | ✅ Partially → coding Q5 (combined) |
| scenario_based_questions.md | Stub Q1 | Image data for CNN | ✅ Yes → scenario S12 |
| scenario_based_questions.md | Stub Q2 | Data management for ML model | ✅ Yes → scenario S13 |
| scenario_based_questions.md | Stub Q3 | Issues with large datasets | ✅ Yes → scenario S14 |

**All 9 stubs have full answers elsewhere — they are duplicate placeholders that should be removed.**

### 🔄 Duplicate Questions (10 instances)

| Question | Appears In | Recommendation |
|----------|-----------|----------------|
| Find unique values and counts (#60) | coding Q7 + general Q10 | Remove from one file |
| Eigenvalues and eigenvectors (#17) | coding Q12 (full) + coding Q36 (stub) | Remove stub Q36 |
| Generate random data (#22) | coding Q6 (full) + coding Q37 (stub) | Remove stub Q37 |
| Concatenate arrays (#15) | coding Q5 (combined) + coding Q35 (stub) | Remove stub Q35 |
| Stack arrays (#38) | coding Q5 (combined) + coding Q38 (stub) | Remove stub Q38 |
| Image data for CNN (#53) | scenario S12 (full) + scenario stub Q1 | Remove stub |
| Data management for ML (#57) | scenario S13 (full) + scenario stub Q2 | Remove stub |
| Large datasets issues (#61) | scenario S14 (full) + scenario stub Q3 | Remove stub |
| NumPy efficiency (#29) | general Q4 (full) + theory Q26 (stub) | Remove stub |
| In-place operations (#33) | scenario S11 (full) + theory Q27 (stub) | Remove stub |

### ➕ Extra Questions Not in NumPy.md (valid NumPy topics)

| File | Question | Assessment |
|------|----------|-----------|
| coding Q2 | How do you reshape a NumPy array? | Valid, common NumPy question — good addition |
| coding Q9 | Z-score normalization | Valid sub-topic of #24 |
| coding Q10 | Min-Max normalization | Valid sub-topic of #24 |
| coding Q11 | Dot product and cross product | Valid NumPy linear algebra topic — good addition |
| scenario S1 | Z-score standardization scenario | Valid applied scenario |
| scenario S2 | Cap outliers with percentiles | Valid applied scenario |
| scenario S3 | Linear regression closed-form | Valid applied scenario |
| scenario S4 | One-hot encoding | Valid applied scenario |
| scenario S5 | Softmax function | Valid applied scenario |
| scenario S6 | Cosine similarity | Valid applied scenario |
| scenario S7 | K-means distances | Valid applied scenario |
| scenario S8 | Train/test split | Valid applied scenario |
| scenario S10 | Precision/recall/F1 | Valid applied scenario |

**All extra questions are legitimate NumPy topics. None belong to other topics (Python, SQL, Pandas, MATLAB).**

---

## 7. Wrong-Topic Check

Cross-referenced all 92 questions against PythonMl.md, SQL in ML.md, Pandas.md, and MATLAB.md:

**Result: 0 wrong-topic questions found.** Every question in the NumPy category files is genuinely about NumPy. No questions accidentally belong to Python, SQL, Pandas, or MATLAB topics.

---

## 8. Recommended Fixes (Priority Order)

### Priority 1 — Fix Content Mismatch
1. **coding_questions.md Q8**: Rename title from "How do you normalize an array in NumPy?" to **"How do you sort a NumPy array?"** (the content is about sorting). Alternatively, replace the sorting content with actual normalization content (but Q9 and Q10 already cover that).

### Priority 2 — Remove Duplicate Stubs
2. Remove **theory Q26** (stub) — full answer exists in general Q4
3. Remove **theory Q27** (stub) — full answer exists in scenario S11
4. Remove **theory Q28** (stub) — full answer exists in scenario S15
5. Remove **coding Q36** (stub) — duplicate of coding Q12
6. Remove **coding Q37** (stub) — duplicate of coding Q6
7. Remove **scenario stub Q1, Q2, Q3** — duplicates of scenarios S12, S13, S14

### Priority 3 — Consolidate Duplicates
8. **coding Q5** covers both concatenation and stacking. Either:
   - Fill in coding Q35 (concatenate) and Q38 (stack) as separate focused answers, OR
   - Remove Q35 and Q38 stubs since Q5 already covers both
9. **coding Q7** and **general Q10** both cover "find unique values" — remove from one file

### Priority 4 — Optional Improvements
10. Coding Q8 (after rename to "sorting") does not correspond to any NumPy.md question — add as a bonus or note it as extra content
11. Consider adding a "How do you sort a NumPy array?" question to NumPy.md since it's a fundamental topic

---

## 9. File Quality Summary

| File | Questions | Full Answers | Stubs | Issues |
|------|-----------|-------------|-------|--------|
| theory_questions.md | 28 | 25 | 3 | 3 stubs are duplicates |
| coding_questions.md | 38 | 30 | 4 | 1 content mismatch, 3 stubs are duplicates, 4 extra topics |
| general_questions.md | 11 | 11 | 0 | 1 duplicate with coding, otherwise clean ✨ |
| scenario_based_questions.md | 15+3 | 15 | 3 | 3 stubs are duplicates, 9 original scenarios (good) |
| **TOTAL** | **92+3** | **81** | **10** | |

**Overall Assessment:** The NumPy folder is in **good shape**. All 70 source questions are covered. The main issues are the content mismatch in coding Q8, 9 removable stub duplicates, and 1 cross-file duplicate. No wrong-topic contamination.
