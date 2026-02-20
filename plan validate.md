# Plan: Validate & Fix Organized Questions Against Source

## TL;DR
Compare all ~67 organized topic folders against their freshly-downloaded source files in `ALL Questions/AI/`. Fix reworded questions to match verbatim, add missing source questions (question text only), keep all destination-only extra content, and create 4 missing topic folders with questions only.

## Known Issues (from sampling 10 of 67 topics)

| Issue | Affected Topics (confirmed) | Likely Scope |
|-------|----------------------------|--------------|
| **Hallucinated/extra questions** not in source | Linear Regression (21 extra), Statistics (14 extra), RL (17+ SARSA) | Likely affects many more topics |
| **Missing questions** from source | NumPy (~17), CNN (~4) | Likely affects more topics |
| **Reworded questions** (paraphrased, not verbatim) | NumPy, PyTorch, Supervised Learning | Several topics |
| **Formatting defects** (broken bold/spaces) | CNN general_questions.md | Possibly others |
| **Empty answers** (placeholder `_To be filled_`) | RL (SARSA block) | Check all topics |
| **Missing organized folders** for source files | ChatGPT, Julia, R, Scala | 4 topics |
| **Miscategorization** (theory Q in coding file) | CNN | Possibly others |

## Steps

### Phase 1: Automated Audit (build a comparison script)

1. **Create a Python audit script** that programmatically:
   - Reads each source file from `ALL Questions/AI/` and extracts all question texts (H3 headers)
   - Reads each organized folder's question files (theory, general, scenario, coding .md files) and extracts all question texts (H2 headers)
   - Performs fuzzy matching (to catch minor space/punctuation differences) between source and organized questions
   - Reports per topic:
     - Questions in organized NOT in source (hallucinated/extra)
     - Questions in source NOT in organized (missing)
     - Questions that are reworded (fuzzy match but not exact)
     - Total count comparison
   - Outputs a structured report (JSON or markdown)
   - *Depends on: nothing. Can start immediately.*

2. **Define the source→folder mapping** as a JSON/dict in the script covering all 67 topics (from the mapping table above)
   - *Parallel with step 1*

### Phase 2: Fix Existing Topics (sequential, topic by topic based on audit results)

3. **For each topic with hallucinated/extra questions**: Keep destination-only extra content as is (don't remove). Only fix questions that are reworded versions of source questions.
   - Known: Linear Regression theory Q31-51 (keep — not in source), Statistics general (keep), RL SARSA block (keep)
   - *Depends on: Phase 1 audit results*

4. **For each topic with missing questions**: Add the missing source questions into the appropriate organized file (theory/general/scenario/coding based on the source's section classification + question type tag `📝`/`💻`). Add question text only, no generated answers needed for newly added questions.
   - Known: NumPy (~17 missing), CNN (~4 missing)
   - *Depends on: Phase 1 audit results*

5. **For each topic with reworded questions**: Replace the reworded question text with the exact verbatim source question text. Keep the existing answer.
   - Known: NumPy, PyTorch, Supervised Learning
   - *Depends on: Phase 1 audit results*

6. **Fix formatting defects**: Repair broken bold markers/spacing in affected files.
   - Known: CNN general_questions.md
   - *Depends on: Phase 1 audit results*

7. **Fix miscategorizations**: Move questions to correct category file (e.g., CNN theory Q in coding file).
   - Known: CNN Source #29
   - *Depends on: Phase 1 audit results*

8. **Update README.md** in each fixed folder to reflect corrected question counts.
   - *Depends on: Steps 3-7*

### Phase 3: Create Missing Topic Folders

9. **Create organized folders** for the 4 missing topics: ChatGPT, Julia, R, Scala
   - Each needs: README.md + a single questions file with all questions from source
   - **Questions only** — no generated answers
   - Follow the same folder/file naming conventions as existing organized folders
   - *Depends on: nothing, parallel with Phase 2*

### Phase 4: Verification

10. **Re-run the audit script** to confirm all source questions are now present in organized folders with exact text matches, no extras, and correct counts.
    - *Depends on: Phases 2 & 3*

11. **Spot-check 5 random topics manually** — read source and organized side-by-side to validate.
    - *Depends on: Step 10*

## Relevant Files

**Source (read-only reference):**
- `ALL Questions/AI/*.md` — 67 source question files (~3,946 questions total)
- `ALL Questions/AI/README.md` — topic list with question counts

**Organized (to be modified):**
- `04_machine_learning/05_linear_regression/theory_questions.md` — keep extra Q31-Q51, fix any reworded source questions
- `03_data_science/02_statistics/general_questions.md` — keep extra questions
- `03_data_science/02_statistics/scenario_based_questions.md` — keep extra Q76-Q77
- `04_machine_learning/28_reinforcement_learning/theory_questions.md` — keep SARSA block
- `02_programming_tools/03_numpy/` — add ~17 missing questions, fix reworded texts
- `04_machine_learning/23_cnn/` — add ~4 missing questions, fix formatting, fix miscategorization
- `02_programming_tools/08_pytorch/theory_questions.md` — fix reworded questions
- `04_machine_learning/01_supervised_learning/theory_questions.md` — fix reworded questions
- All other organized topic folders — audit TBD
- New folders to create: ChatGPT, Julia, R, Scala

**Script to create:**
- `audit_questions.py` — comparison/audit script (root of workspace or in `ALL Questions/`)

## Verification
1. Run `audit_questions.py` — should report 0 missing questions, 0 reworded questions across all 67 topics
2. Verify total organized question count ≥ total source count (3,946)
3. Manually spot-check 5 topics: compare source H3 headers vs organized H2 headers line by line
4. Check all new folders (ChatGPT, Julia, R, Scala) have correct README counts and complete question sets
5. Grep for `_To be filled_` across all organized files — should return 0 results

## Decisions
- **Source is authoritative** — if a topic exists in both source and destination, match questions to source
- **Question text must be verbatim** from source (only trimming trailing spaces is acceptable)
- **Destination-only content is kept** — if destination has extra questions/content NOT in source, keep as is (don't remove)
- **Entire destination-only topics are kept** — if a topic folder exists in destination but has NO source file, keep as is
- **Question categorization** follows source section headers + type tags: `📝 Question` → theory/general/scenario, `💻 Coding Challenge` → coding
- **Model Evaluation**: `Model Evaluation.md` maps to `03_data_science/08_model_evaluation/` only; `11_model_evaluation_metrics/` is kept as is (separate detailed content)
- **Processing order**: sequential, one topic at a time (not batch)
- **4 new topic folders** (ChatGPT, Julia, R, Scala): add questions only, no generated answers
