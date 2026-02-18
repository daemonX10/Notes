# Plan: Restore Missing Questions to Organized Folders

## TL;DR
During folder restructuring, many general, scenario, and coding questions were deleted from organized topic files. The "All questions/" folder contains the original versions. The task is to systematically compare each topic's files between "All questions/" (source) and the main organized folders (target), then add back any missing questions as stubs.

## Scope
- **Source**: Only organized .md files in "All questions/" subfolders (NOT raw .txt files)
- **Format**: Add missing questions as stubs (`**Answer:** _[To be filled]_`)
- **Focus**: General, scenario, coding questions (brief theory check)
- **No duplicate cleanup**: Leave existing appended stub batches as-is

---

## Phase 1: 01_foundations_mathematics — SKIP (verified matching)
Both topics (linear_algebra, time_series) have identical counts. No action needed.

---

## Phase 2: 02_programming_tools — HIGH PRIORITY (~130+ missing questions)

For each topic, compare `All questions/02_programming_tools/{topic}/{type}_questions.md` vs `02_programming_tools/{topic}/{type}_questions.md`:

### Step 2.1: 01_python_ml
- general_questions.md: 15 → 1 (**add 14 missing**)
- scenario_based_questions.md: 16 → 6 (**add 10 missing**)
- coding_questions.md: 29 → 7 (**add 22 missing**)

### Step 2.2: 02_sql_ml
- theory_questions.md: 18 → 16 (**add 2 missing**)
- general_questions.md: 14 → 6 (**add 8 missing**)
- coding_questions.md: 15 → 10 (**add 5 missing**)

### Step 2.3: 03_numpy
- general_questions.md: 11 → 6 (**add 5 missing**)
- scenario_based_questions.md: 5 → 0 (**create file with 5 questions**)
- coding_questions.md: 28 → 12 (**add 16 missing**)

### Step 2.4: 04_pandas
- general_questions.md: 15 → 11 (**add 4 missing**)
- scenario_based_questions.md: 5 → 0 (**create file with 5 questions**)

### Step 2.5: 05_scikit_learn
- general_questions.md: 13 → 10 (**add 3 missing**)
- coding_questions.md: 13 → 10 (**add 3 missing**)

### Step 2.6: 06_tensorflow
- theory_questions.md: 34 → 32 (**add 2 missing**)
- general_questions.md: 18 → 10 (**add 8 missing**)
- coding_questions.md: 12 → 10 (**add 2 missing**)

### Step 2.7: 07_keras
- general_questions.md: 17 → 10 (**add 7 missing**)
- coding_questions.md: 19 → 10 (**add 9 missing**)

### Step 2.8: 08_pytorch
- coding_questions.md: 12 → 10 (**add 2 missing**)

### Step 2.9: 09_hadoop — OK (matched or main expanded). Skip.
### Step 2.10: 10_apache_spark — OK (matched or main expanded). Skip.

### Step 2.11: 11_matlab
- general_questions.md: 19 → 5 (**add 14 missing**)
- scenario_based_questions.md: 14 → 5 (**add 9 missing**)

---

## Phase 3: 03_data_science — MEDIUM PRIORITY

### To skip (verified matching):
- 01_probability — SKIP
- 08_model_evaluation — SKIP
- 09_ml_design_patterns — SKIP
- 11_llmops — SKIP

### To investigate and restore:
- **02_statistics** — incomplete data; check general/scenario/coding counts in both source and main
- **03_data_processing** — flagged as possibly having NO content in main; check and create all 4 types if empty
- **04_data_mining** — incomplete data; verify counts
- **05_data_analyst** — main has general only; check if source has scenario + coding (source showed NO scenario/coding)
- **06_data_engineer** — main has theory only; check if source has general/scenario/coding (source showed NONE)
- **07_data_scientist** — main has theory + coding; check if source has general + scenario (source showed NONE for general/scenario)
- **10_mlops** — verify general/scenario/coding counts between source and main

---

## Phase 4: 04_machine_learning — MEDIUM PRIORITY (28 core topics)

Only topic 01 (supervised_learning) confirmed matching. Remaining 27 topics (02–28) need systematic comparison of general/scenario/coding counts.

### Step 4.1: Check each topic individually
For each topic from `02_unsupervised_learning` to `28_reinforcement_learning`:
1. Read source file from `All questions/04_machine_learning/{topic}/{type}_questions.md`
2. Read target file from `04_machine_learning/{topic}/{type}_questions.md`
3. Identify questions in source but NOT in target (by question text, not number)
4. Append missing questions to end of target, continuing numbering

---

## Phase 5: 06_algorithms_optimization — SKIP
No "All questions/" counterpart exists. Main folder has complete set.

---

## Phase 6: 07_computer_vision — MEDIUM PRIORITY (missing file types)

Main has 3 bundled subfolders with `theory_questions.md` only.
Source: `All questions/06_ai_nlp/02_computer_vision/`
- general_questions.md (13 Q)
- scenario_based_questions.md (10 Q)
- coding_questions.md (5 Q)

### Step 6.1: Create general_questions.md in `07_computer_vision/01_cnn_architectures_and_detection/`
### Step 6.2: Create scenario_based_questions.md in same subfolder
### Step 6.3: Create coding_questions.md in same subfolder
(Questions are broad CV questions — placing in subfolder 01 as the primary/most visited)

---

## Phase 7: 08_natural_language_processing — MEDIUM PRIORITY (missing file types)

Source: `All questions/06_ai_nlp/01_nlp/`
- general_questions.md (13 Q)
- scenario_based_questions.md (8 Q)
- coding_questions.md (5 Q)

### Step 7.1–7.3: Create all three files in `08_natural_language_processing/01_nlp_fundamentals/`

---

## Phase 8: 09_large_language_models_genai — MEDIUM PRIORITY (missing file types)

Sources:
- `All questions/06_ai_nlp/03_llms/` → general (15 Q), scenario (15 Q), coding (10 Q)
- `All questions/06_ai_nlp/04_chatgpt/` → general (13 Q), scenario (11 Q), coding (7 Q)

### Step 8.1: Merge LLMs + ChatGPT general → create `09_large_language_models_genai/01_llm_architectures_models/general_questions.md` (28 Q)
### Step 8.2: Merge LLMs + ChatGPT scenario → create `09_large_language_models_genai/01_llm_architectures_models/scenario_based_questions.md` (26 Q)
### Step 8.3: Merge LLMs + ChatGPT coding → create `09_large_language_models_genai/01_llm_architectures_models/coding_questions.md` (17 Q)

---

## Phase 9: 10_explainable_ai — MEDIUM PRIORITY (missing file types)

Source: `All questions/06_ai_nlp/05_explainable_ai/`
- general_questions.md (6 Q)
- scenario_based_questions.md (4 Q)
- coding_questions.md (8 Q)

### Step 9.1–9.3: Create all three files in `10_explainable_ai/` root (same level as existing `explainable_ai_questions.md`)

---

## Phase 10: 11_model_evaluation_metrics — SKIP
No general/scenario/coding source files exist in "All questions/". Only theory exists.

---

## Phase 11: Brief Theory Check

Spot-check theory question counts across all sections to catch any significant losses:
- 02_programming_tools: 02_sql_ml (−2), 06_tensorflow (−2) → already accounted for in Phase 2
- 03_data_science: spot-check 02_statistics, 03_data_processing, 04_data_mining, 05_data_analyst
- 04_machine_learning: spot-check 5–6 representative topics
- No action needed for 07–11 (theory files were confirmed matching or expanded)

---

## Implementation Approach (per topic)

1. Read the "All questions/" source file completely
2. Read the main organized target file completely
3. Identify questions in source but NOT in target (by question text matching, not number)
4. Append missing questions to end of target file, continuing the existing question numbering
5. If target file doesn't exist, create it from the source file content

## Format for appended questions

```markdown
---

## Question N: [Title / brief label]

**[Question text as written in source]**

**Answer:** _[To be filled]_
```

## Verification (after each phase)

1. Re-count `## Question` headers in modified files to confirm counts match or exceed source
2. Check that appended questions follow the same format as existing content
3. Spot-check for accidental duplicates (scan question text against existing entries)

---

## Decisions

- Questions added as stubs only — no answers generated
- No cleanup of existing appended stub batches in theory files
- Where main already has MORE questions than source (main was expanded), no action needed
- `06_algorithms_optimization` and `11_model_evaluation_metrics` skipped — no source material available
- For consolidated subfolders (07–09), broad-topic source questions placed in the first/most relevant subfolder
- `03_data_science` roles (data analyst/engineer/scientist) depend on source availability — if source has no general/scenario/coding, nothing to restore

---

## File Path Quick Reference

| Section | Source root | Target root |
|---------|-------------|-------------|
| 02_programming_tools | `All questions/02_programming_tools/` | `02_programming_tools/` |
| 03_data_science | `All questions/03_data_science/` | `03_data_science/` |
| 04_machine_learning | `All questions/04_machine_learning/` | `04_machine_learning/` |
| 07_computer_vision | `All questions/06_ai_nlp/02_computer_vision/` | `07_computer_vision/01_cnn_architectures_and_detection/` |
| 08_NLP | `All questions/06_ai_nlp/01_nlp/` | `08_natural_language_processing/01_nlp_fundamentals/` |
| 09_LLM | `All questions/06_ai_nlp/03_llms/` + `04_chatgpt/` | `09_large_language_models_genai/01_llm_architectures_models/` |
| 10_XAI | `All questions/06_ai_nlp/05_explainable_ai/` | `10_explainable_ai/` |
