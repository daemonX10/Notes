# AUDIT REPORT: 02_sql_ml Folder

**Audit Date:** 2026-02-20  
**Master Topic File:** SQL in ML.md (55 questions)  
**Category Files Audited:** theory_questions.md, coding_questions.md, general_questions.md, scenario_based_questions.md

---

## Summary

| Category File | Total Questions | Correct | Wrong (Not in Master) | Stubs | Duplicates |
|---|---|---|---|---|---|
| theory_questions.md | 16 | 16 | 0 | 0 | 0 |
| coding_questions.md | 24 | 18 | 6 | 2 | 2 |
| general_questions.md | 26 | 26 | 0 | 10 | 4 |
| scenario_based_questions.md | 8 | 2 | 6 | 0 | 0 |
| **TOTALS** | **74** | **62** | **12** | **12** | **6** |

- **All 55 master questions are covered** in at least one category file.
- **0 questions belong to other topics** (Python, NumPy, Pandas, MATLAB).
- **12 questions are not from the master list** — they are supplementary SQL questions (valid SQL content, just not from the original 55).
- **12 stub questions** (title only, no answer content).
- **6 duplicate questions** appear in multiple locations.

---

## 1. theory_questions.md — 16 Questions

| # | Question Title | Status | Master Q# |
|---|---|---|---|
| 1 | What are the different types of JOIN operations in SQL? | ✅ CORRECT | Q1 |
| 2 | Explain the difference between WHERE and HAVING clauses. | ✅ CORRECT | Q2 |
| 3 | Describe a subquery and its typical use case. | ✅ CORRECT | Q6 |
| 4 | What are SQL Window Functions and how can they be used for Machine Learning feature engineering? | ✅ CORRECT | Q15 |
| 5 | Explain how to discretize a continuous variable in SQL. | ✅ CORRECT | Q19 |
| 6 | Describe SQL techniques to perform data sampling. | ✅ CORRECT | Q25 |
| 7 | Explain how to perform binning of categorical variables in SQL for use in a Machine Learning model. | ✅ CORRECT | Q24 |
| 8 | How does SQL play a role in ML model deployment? | ✅ CORRECT | Q29 |
| 9 | What is the significance of in-database analytics for Machine Learning? | ✅ CORRECT | Q30 |
| 10 | Explain recursive SQL queries and how they can be used to prepare data for hierarchical Machine Learning algorithms. | ✅ CORRECT | Q31 |
| 11 | Describe how graph-based features can be generated from SQL data. | ✅ CORRECT | Q32 |
| 12 | What are SQL Common Table Expressions (CTEs) and how can they be used for feature generation? | ✅ CORRECT | Q33 |
| 13 | Explain the role of partitioning in large-scale SQL databases. | ✅ CORRECT | Q42 |
| 14 | Describe how you could use SQL to report the performance metrics of a Machine Learning model. | ✅ CORRECT | Q48 |
| 15 | Describe how you would version control the datasets used for building Machine Learning models in SQL. | ✅ CORRECT | Q51 |
| 16 | What is Data Lineage, and how can you track it using SQL? | ✅ CORRECT | Q52 |

**Verdict: 16/16 correct. No issues.**

---

## 2. coding_questions.md — 24 Questions

| # | Question Title | Status | Master Q# | Notes |
|---|---|---|---|---|
| 1 | Write a SQL query that joins two tables and retrieves only the rows with matching keys | ✅ CORRECT | Q10 | |
| 2 | Write a SQL query to pivot rows into columns. | ✅ CORRECT | Q12 | |
| 3 | Write a SQL query to find duplicate records. | ✅ CORRECT | Q21 | Answer includes find + delete |
| 4 | Write a SQL query to calculate cumulative sum. | ❌ WRONG | — | Not in master list. Supplementary SQL question. |
| 5 | Write a SQL query to find the second highest salary. | ❌ WRONG | — | Not in master list. Generic SQL interview question. |
| 6 | Write a SQL query to calculate year-over-year growth. | ❌ WRONG | — | Not in master list. Supplementary SQL question. |
| 7 | Write a SQL query to create ML features from transaction data. | ❌ WRONG | — | Not in master list. Similar to Q20 concept but different question. |
| 8 | Write a SQL query to detect outliers using IQR method. | ✅ CORRECT | Q22 | Coding implementation of Q22 |
| 9 | Write a SQL query to create a train/test split. | ❌ WRONG | — | Not in master list. Supplementary SQL question. |
| 10 | Write a SQL query to one-hot encode a categorical column. | ❌ WRONG | — | Not in master list. Supplementary SQL question. |
| 11 | Write a SQL query to calculate moving averages. | ✅ CORRECT | Q16 | |
| 12 | How can you create lagged features in SQL? | ✅ CORRECT | Q17 | |
| 13 | Describe how to compute a ratio feature within groups using SQL. | ✅ CORRECT | Q18 | |
| 14 | In SQL, how would you format strings or concatenate columns for text-based Machine Learning features? | ✅ CORRECT | Q23 | |
| 15 | Write a SQL stored procedure that calls a Machine Learning scoring function. | ✅ CORRECT | Q27 | |
| 16 | How would you construct a complex SQL query to extract time series features for a Machine Learning model? | ✅ CORRECT | Q34 | |
| 17 | Discuss ways to implement regular expressions in SQL for natural language processing tasks. | ✅ CORRECT | Q35 | |
| 18 | Write a SQL script to identify and replace missing values with the column mean. | ✅ CORRECT | Q36 | |
| 19 | Create a SQL query that normalizes a column (scales between 0 and 1). | ✅ CORRECT | Q37 | |
| 20 | Generate a feature that is a count over a rolling time window using SQL. | ✅ CORRECT | Q38 | |
| 21 | Code an SQL function that categorizes continuous variables into bins. | ✅ CORRECT | Q39 | |
| 22 | Implement a SQL solution to compute the TF-IDF score for text data. | ✅ CORRECT | Q40 | |
| 23 | Create a SQL query to pivot a table transforming rows into columns | ⚠️ STUB + DUPLICATE | Q12 | Duplicate of Q2; answer says "to be added" |
| 24 | Write a SQL query that identifies and removes duplicate records from a dataset | ⚠️ STUB + DUPLICATE | Q21 | Duplicate of Q3; answer says "to be added" |

### Wrong Questions — Where They Belong

| Coding Q# | Question | Belongs To |
|---|---|---|
| 4 | Calculate cumulative sum | **No other topic** — supplementary SQL question (not in any master file) |
| 5 | Find second highest salary | **No other topic** — classic SQL interview question (not in any master file) |
| 6 | Calculate year-over-year growth | **No other topic** — supplementary SQL question (not in any master file) |
| 7 | Create ML features from transaction data | **No other topic** — supplementary SQL/ML question (not in any master file) |
| 9 | Create a train/test split | **No other topic** — supplementary SQL/ML question (not in any master file) |
| 10 | One-hot encode a categorical column | **No other topic** — supplementary SQL/ML question (not in any master file) |

> **Note:** All 6 "wrong" questions are valid SQL-for-ML content. They are not misplaced from another topic — they are **bonus/supplementary** coding questions that don't correspond to any of the 55 master questions.

---

## 3. general_questions.md — 26 Questions

| # | Question Title | Status | Master Q# | Notes |
|---|---|---|---|---|
| 1 | What does GROUP BY do in a SQL query? | ✅ CORRECT | Q4 | Full answer |
| 2 | Explain indexes and their importance for ML pipelines. | ✅ CORRECT | Q7 | Full answer |
| 3 | How do you handle NULL values in SQL? | ✅ CORRECT | Q9 | Full answer |
| 4 | What is a Common Table Expression (CTE)? | ✅ CORRECT | Q33 | Full answer; **DUPLICATE** with theory Q12 |
| 5 | What is the difference between UNION and UNION ALL? | ✅ CORRECT | Q11 | Full answer |
| 6 | How do you calculate running totals and moving averages in SQL? | ✅ CORRECT | Q16 | Full answer; **DUPLICATE** with coding Q11 |
| 7 | How can you aggregate data in SQL (e.g., COUNT, AVG, SUM, MAX, MIN)? | ✅ CORRECT | Q5 | Full answer |
| 8 | How can you extract time-based features from a SQL datetime field for use in a Machine Learning model? | ✅ CORRECT | Q14 | Full answer |
| 9 | How do you join transactional data to a dimension table in such a way that features for Machine Learning can be extracted? | ✅ CORRECT | Q20 | Full answer |
| 10 | How can you deal with outliers in a SQL database before passing data to Machine Learning algorithms? | ✅ CORRECT | Q22 | Full answer; **DUPLICATE** with coding Q8 |
| 11 | How can you execute a Machine Learning model stored in a database? | ✅ CORRECT | Q26 | Full answer |
| 12 | Can you update a Machine Learning model directly from SQL? If so, how might you do it? | ✅ CORRECT | Q28 | Full answer |
| 13 | What strategies can be used to efficiently update a large SQL-based Machine Learning model? | ✅ CORRECT | Q43 | Full answer |
| 14 | How do you ensure the consistency and reliability of SQL data used for Machine Learning? | ✅ CORRECT | Q44 | Full answer |
| 15 | What SQL features are there for report generation that might be useful for analyzing ML model performance? | ✅ CORRECT | Q46 | Full answer |
| 16 | How can you use SQL to visualize the distribution of data points before feeding them into an ML algorithm? | ✅ CORRECT | Q47 | Full answer |
| 17 | Can SQL be used to visualize false positives and false negatives in classification models? If so, how? | ✅ CORRECT | Q49 | Full answer |
| 18 | What strategies might you use to automate the retraining and evaluation of Machine Learning models from within SQL? | ✅ CORRECT | Q55 | Full answer |
| 19 | Can you explain the use of indexes in databases and how they relate to Machine Learning? | ⚠️ STUB + DUPLICATE | Q7 | Duplicate of Q2; answer says "[To be filled]" |
| 20 | Explain the importance of data normalization in SQL and how it affects Machine Learning models. | ⚠️ STUB | Q13 | Answer says "[To be filled]" |
| 21 | How would you optimize a SQL query that seems to be running slowly? | ⚠️ STUB | Q8 | Answer says "to be added" |
| 22 | How do you handle missing values in a SQL dataset? | ⚠️ STUB + DUPLICATE | Q9 | Duplicate of Q3; answer says "to be added" |
| 23 | How would you merge multiple result sets in SQL without duplicates? | ⚠️ STUB + DUPLICATE | Q11 | Duplicate of Q5; answer says "to be added" |
| 24 | How would you handle very large datasets in SQL for Machine Learning purposes? | ⚠️ STUB | Q41 | Answer says "to be added" |
| 25 | Discuss how you would design a system to regularly feed a Machine Learning model with SQL data | ⚠️ STUB | Q45 | Answer says "to be added" |
| 26 | How would you extract and prepare a confusion matrix for a classification problem using SQL? | ⚠️ STUB | Q50 | Answer says "to be added" |
| 27 | How would you log and track predictions made by a Machine Learning model within a SQL environment? | ⚠️ STUB | Q53 | Answer says "to be added" |
| 28 | Discuss how to manage the entire lifecycle of a Machine Learning model using SQL tools | ⚠️ STUB | Q54 | Answer says "to be added" |

**Verdict: 26/26 correct (all match master). But 10 are stubs (no answer), and 4 are duplicates of earlier answered questions in the same file.**

---

## 4. scenario_based_questions.md — 8 Questions

| # | Question Title | Status | Master Q# | Notes |
|---|---|---|---|---|
| 1 | How would you write a SQL query to select distinct values from a column? | ✅ CORRECT | Q3 | |
| 2 | Scenario: You have a customer table with duplicate emails. How would you identify and handle them? | ✅ CORRECT | Q21 | Scenario application of Q21 |
| 3 | Scenario: Build a churn prediction feature set for customers who haven't purchased in 30 days. | ❌ WRONG | — | Not in master list. Supplementary scenario. |
| 4 | Scenario: Calculate customer RFM (Recency, Frequency, Monetary) scores. | ❌ WRONG | — | Not in master list. Supplementary scenario. |
| 5 | Scenario: Find products frequently bought together for recommendation system. | ❌ WRONG | — | Not in master list. Supplementary scenario. |
| 6 | Scenario: Detect anomalous transactions (potential fraud). | ❌ WRONG | — | Not in master list. Related to Q22 outliers but distinct. |
| 7 | Scenario: Create time-based features for time series forecasting. | ❌ WRONG | — | Not in master list. Related to Q14/Q34 but distinct. |
| 8 | Scenario: Segment customers into tiers based on spending. | ❌ WRONG | — | Not in master list. Supplementary scenario. |

### Wrong Questions — Where They Belong

All 6 "wrong" scenario questions are **valid SQL-for-ML content**. None belong to Python, NumPy, Pandas, or MATLAB. They are **supplementary practical scenarios** that demonstrate SQL skills for ML applications.

---

## 5. Missing Questions from Master List

**All 55 questions from SQL in ML.md are covered.** Here's the complete coverage map:

| Master Q# | Question (abbreviated) | Covered In |
|---|---|---|
| 1 | JOIN operations | theory Q1 |
| 2 | WHERE vs HAVING | theory Q2 |
| 3 | Select distinct | scenario Q1 |
| 4 | GROUP BY | general Q1 |
| 5 | Aggregate functions | general Q7 |
| 6 | Subquery | theory Q3 |
| 7 | Indexes | general Q2 + general Q19 (stub) |
| 8 | Optimize slow query | general Q21 (stub only) |
| 9 | Missing values | general Q3 + general Q22 (stub) |
| 10 | JOIN two tables (coding) | coding Q1 |
| 11 | Merge result sets | general Q5 + general Q23 (stub) |
| 12 | Pivot table (coding) | coding Q2 + coding Q23 (stub) |
| 13 | Data normalization | general Q20 (stub only) |
| 14 | Time-based features | general Q8 |
| 15 | Window functions | theory Q4 |
| 16 | Moving averages (coding) | coding Q11 + general Q6 |
| 17 | Lagged features (coding) | coding Q12 |
| 18 | Ratio feature (coding) | coding Q13 |
| 19 | Discretize continuous variable | theory Q5 |
| 20 | Join transactional to dimension | general Q9 |
| 21 | Remove duplicates (coding) | coding Q3 + coding Q24 (stub) + scenario Q2 |
| 22 | Outliers | general Q10 + coding Q8 |
| 23 | String formatting (coding) | coding Q14 |
| 24 | Binning categorical | theory Q7 |
| 25 | Data sampling | theory Q6 |
| 26 | Execute ML model | general Q11 |
| 27 | Stored procedure scoring (coding) | coding Q15 |
| 28 | Update ML model | general Q12 |
| 29 | SQL role in deployment | theory Q8 |
| 30 | In-database analytics | theory Q9 |
| 31 | Recursive queries | theory Q10 |
| 32 | Graph-based features | theory Q11 |
| 33 | CTEs | theory Q12 + general Q4 |
| 34 | Time series features (coding) | coding Q16 |
| 35 | Regex for NLP | coding Q17 |
| 36 | Replace missing with mean (coding) | coding Q18 |
| 37 | Normalize column (coding) | coding Q19 |
| 38 | Rolling time window (coding) | coding Q20 |
| 39 | Categorize into bins (coding) | coding Q21 |
| 40 | TF-IDF (coding) | coding Q22 |
| 41 | Handle large datasets | general Q24 (stub only) |
| 42 | Partitioning | theory Q13 |
| 43 | Efficiently update model | general Q13 |
| 44 | Consistency/reliability | general Q14 |
| 45 | Design system to feed ML | general Q25 (stub only) |
| 46 | Report generation | general Q15 |
| 47 | Visualize distribution | general Q16 |
| 48 | Report performance metrics | theory Q14 |
| 49 | Visualize FP/FN | general Q17 |
| 50 | Confusion matrix | general Q26 (stub only) |
| 51 | Version control datasets | theory Q15 |
| 52 | Data lineage | theory Q16 |
| 53 | Log/track predictions | general Q27 (stub only) |
| 54 | ML lifecycle management | general Q28 (stub only) |
| 55 | Automate retraining | general Q18 |

### Questions with STUB-ONLY coverage (no full answer anywhere):

| Master Q# | Question | Location (stub) |
|---|---|---|
| 8 | How would you optimize a SQL query that seems to be running slowly? | general Q21 |
| 13 | Explain the importance of data normalization in SQL and how it affects ML models. | general Q20 |
| 41 | How would you handle very large datasets in SQL for ML purposes? | general Q24 |
| 45 | Discuss how you would design a system to regularly feed a ML model with SQL data. | general Q25 |
| 50 | How would you extract and prepare a confusion matrix for a classification problem using SQL? | general Q26 |
| 53 | How would you log and track predictions made by a ML model within a SQL environment? | general Q27 |
| 54 | Discuss how to manage the entire lifecycle of a ML model using SQL tools. | general Q28 |

> **7 master questions exist ONLY as stubs** — they have titles but no answer content yet.

---

## 6. Duplicates Across Files

| Question | Locations | Notes |
|---|---|---|
| CTEs / feature generation (Q33) | theory Q12 + general Q4 | Both have full answers |
| Moving averages (Q16) | coding Q11 + general Q6 | Both have full answers |
| Outliers (Q22) | coding Q8 + general Q10 | Both have full answers |
| Indexes (Q7) | general Q2 (full) + general Q19 (stub) | Stub duplicate within same file |
| Missing values (Q9) | general Q3 (full) + general Q22 (stub) | Stub duplicate within same file |
| Merge result sets (Q11) | general Q5 (full) + general Q23 (stub) | Stub duplicate within same file |
| Pivot table (Q12) | coding Q2 (full) + coding Q23 (stub) | Stub duplicate within same file |
| Remove duplicates (Q21) | coding Q3 (full) + coding Q24 (stub) + scenario Q2 | Stub duplicate + cross-file |

---

## 7. Recommendations

1. **Remove 6 supplementary coding questions** (coding Q4-Q7, Q9-Q10) or explicitly label them as "Bonus" questions not from the master list.
2. **Remove 6 supplementary scenario questions** (scenario Q3-Q8) or label as "Bonus" scenarios.
3. **Delete 4 intra-file stub duplicates** in general_questions.md (Q19, Q22, Q23) and coding_questions.md (Q23, Q24).
4. **Write answers for 7 stub-only questions** (master Q8, Q13, Q41, Q45, Q50, Q53, Q54) that currently have no full answer anywhere.
5. **Resolve 3 cross-file duplicates** (CTEs, moving averages, outliers) — keep in one file and reference from the other.
