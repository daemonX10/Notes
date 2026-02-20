"""
Audit script: Compare source questions (ALL Questions/AI/) against organized topic folders.
Reports: missing questions, reworded questions, extra questions (destination-only).
"""

import os
import re
import json
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).parent

SOURCE_DIR = ROOT / "ALL Questions" / "AI"

# ── Mapping: source filename → organized folder path(s) ──
# For multi-folder topics, use a list of folder paths.
SOURCE_TO_ORGANIZED = {
    # 01_foundations_mathematics
    "Linear Algebra.md":           ["01_foundations_mathematics/01_linear_algebra/"],
    "Time Series.md":              ["01_foundations_mathematics/02_time_series/"],

    # 02_programming_tools
    "PythonMl.md":                 ["02_programming_tools/01_python_ml/"],
    "SQL in ML.md":                ["02_programming_tools/02_sql_ml/"],
    "NumPy.md":                    ["02_programming_tools/03_numpy/"],
    "Pandas.md":                   ["02_programming_tools/04_pandas/"],
    "Scikit-Learn.md":             ["02_programming_tools/05_scikit_learn/"],
    "TensorFlow.md":               ["02_programming_tools/06_tensorflow/"],
    "Keras.md":                    ["02_programming_tools/07_keras/"],
    "PyTorch.md":                  ["02_programming_tools/08_pytorch/"],
    "Hadoop.md":                   ["02_programming_tools/09_hadoop/"],
    "Apache Spark.md":             ["02_programming_tools/10_apache_spark/"],
    "MATLAB.md":                   ["02_programming_tools/11_matlab/"],

    # 03_data_science
    "Probability.md":              ["03_data_science/01_probability/"],
    "Statistics.md":               ["03_data_science/02_statistics/"],
    "Data Processing.md":          ["03_data_science/03_data_processing/"],
    "Data Mining.md":              ["03_data_science/04_data_mining/"],
    "Data Analyst.md":             ["03_data_science/05_data_analyst/"],
    "Data Engineer.md":            ["03_data_science/06_data_engineer/"],
    "Data Scientist.md":           ["03_data_science/07_data_scientist/"],
    "Model Evaluation.md":         ["03_data_science/08_model_evaluation/"],
    "ML Design Patterns.md":       ["03_data_science/09_ml_design_patterns/"],
    "MLOps.md":                    ["03_data_science/10_mlops/"],
    "LLMOps.md":                   ["03_data_science/11_llmops/"],

    # 04_machine_learning
    "Supervised Learning.md":      ["04_machine_learning/01_supervised_learning/"],
    "Unsupervised Learning.md":    ["04_machine_learning/02_unsupervised_learning/"],
    "Bias & Variance.md":          ["04_machine_learning/03_bias_and_variance/"],
    "Feature Engineering.md":      ["04_machine_learning/04_feature_engineering/"],
    "Linear Regression.md":        ["04_machine_learning/05_linear_regression/"],
    "Logistic Regression.md":      ["04_machine_learning/06_logistic_regression/"],
    "K-Nearest Neighbors.md":      ["04_machine_learning/07_k_nearest_neighbors/"],
    "Naive Bayes.md":              ["04_machine_learning/08_naive_bayes/"],
    "Decision Trees.md":           ["04_machine_learning/09_decision_tree/"],
    "SVM.md":                      ["04_machine_learning/10_svm/"],
    "Classification Algorithms.md": ["04_machine_learning/11_classification_algorithms/"],
    "Gradient Descent.md":         ["04_machine_learning/12_gradient_descent/"],
    "K-Means Clustering.md":       ["04_machine_learning/13_k_means_clustering/"],
    "Cluster Analysis.md":         ["04_machine_learning/14_cluster_analysis/"],
    "PCA.md":                      ["04_machine_learning/15_pca/"],
    "Dimensionality Reduction.md": ["04_machine_learning/16_dimensionality_reduction/"],
    "Ensemble Learning.md":        ["04_machine_learning/17_ensemble_learning/"],
    "Random Forest.md":            ["04_machine_learning/18_random_forest/"],
    "XGBoost.md":                  ["04_machine_learning/19_xgboost/"],
    "Anomaly Detection.md":        ["04_machine_learning/20_anomaly_detection/"],
    "Neural Networks.md":          ["04_machine_learning/21_neural_networks/"],
    "Deep Learning.md":            ["04_machine_learning/22_deep_learning/"],
    "CNN.md":                      ["04_machine_learning/23_cnn/"],
    "RNN.md":                      ["04_machine_learning/24_rnn/"],
    "Autoencoders.md":             ["04_machine_learning/25_autoencoders/"],
    "GANs.md":                     ["04_machine_learning/26_gans/"],
    "Transfer Learning.md":        ["04_machine_learning/27_transfer_learning/"],
    "Reinforcement Learning.md":   ["04_machine_learning/28_reinforcement_learning/"],

    # 06_algorithms_optimization
    "Cost Function.md":            ["06_algorithms_optimization/01_cost_function/"],
    "Optimization.md":             ["06_algorithms_optimization/02_optimization/"],
    "Curse of Dimensionality.md":  ["06_algorithms_optimization/03_curse_of_dimensionality/"],
    "Genetic Algorithms.md":       ["06_algorithms_optimization/04_genetic_algorithms/"],
    "Q-Learning.md":               ["06_algorithms_optimization/05_q_learning/"],
    "LightGBM.md":                 ["06_algorithms_optimization/06_light_gbm/"],
    "Recommendation Systems.md":   ["06_algorithms_optimization/07_recommendation_systems/"],

    # Multi-folder topics
    "Computer Vision.md":          [
        "07_computer_vision/00_core_questions.md",
        "07_computer_vision/01_cnn_architectures_and_detection/",
        "07_computer_vision/02_segmentation_and_transformers/",
        "07_computer_vision/03_generative_models_and_applications/",
    ],
    "NLP.md":                      [
        "08_natural_language_processing/01_nlp_fundamentals/",
        "08_natural_language_processing/02_text_understanding/",
        "08_natural_language_processing/03_text_generation/",
    ],
    "LLMs.md":                     [
        "09_large_language_models_genai/01_llm_architectures_models/",
        "09_large_language_models_genai/02_embeddings_vector_systems/",
        "09_large_language_models_genai/03_llm_applications_engineering/",
    ],
    "Explainable AI.md":           ["10_explainable_ai/"],

    # No organized folder
    "ChatGPT.md":                  None,
    "Julia.md":                    None,
    "R.md":                        None,
    "Scala.md":                    None,
}


def normalize(text: str) -> str:
    """Normalize question text for comparison: lowercase, collapse spaces, strip punctuation edges."""
    text = text.strip()
    # Remove trailing ? if present
    text = text.rstrip("?").rstrip()
    # Collapse multiple spaces to one
    text = re.sub(r"\s+", " ", text)
    # Lowercase
    text = text.lower()
    # Remove leading/trailing punctuation and spaces
    text = text.strip(" .,;:!?")
    return text


def parse_source_questions(filepath: Path) -> list[dict]:
    """Parse questions from a source file (ALL Questions/AI/*.md).
    Returns list of {num, text, type, section, raw_text}
    """
    content = filepath.read_text(encoding="utf-8")
    questions = []
    current_section = ""

    for line in content.split("\n"):
        # Track section headers (## Section Name)
        sec_match = re.match(r"^## (.+)$", line.strip())
        if sec_match:
            sec_name = sec_match.group(1).strip()
            if sec_name != "Table of Contents":
                current_section = sec_name

        # Match question headers: ### N. Question text
        q_match = re.match(r"^### (\d+)\.\s+(.+)$", line.strip())
        if q_match:
            num = int(q_match.group(1))
            raw_text = q_match.group(2).strip()
            questions.append({
                "num": num,
                "raw_text": raw_text,
                "text": normalize(raw_text),
                "type": None,  # filled next
                "section": current_section,
            })

    # Second pass: find types
    lines = content.split("\n")
    q_idx = 0
    for i, line in enumerate(lines):
        if re.match(r"^### \d+\.\s+", line.strip()):
            # Find the type line nearby (within next 5 lines)
            for j in range(i + 1, min(i + 6, len(lines))):
                if "**Type:**" in lines[j]:
                    if "💻" in lines[j] or "Coding" in lines[j]:
                        questions[q_idx]["type"] = "coding"
                    else:
                        questions[q_idx]["type"] = "theory"
                    break
            q_idx += 1
            if q_idx >= len(questions):
                break

    return questions


def parse_organized_questions_from_file(filepath: Path) -> list[dict]:
    """Parse questions from an organized .md file.
    Returns list of {num, text, raw_text, file}
    """
    if not filepath.exists():
        return []

    content = filepath.read_text(encoding="utf-8")
    questions = []

    lines = content.split("\n")
    for i, line in enumerate(lines):
        # Match: ## Question N
        q_match = re.match(r"^## Question\s+(\d+)\s*$", line.strip())
        if q_match:
            num = int(q_match.group(1))
            # Look for bold question text in next few lines
            raw_text = ""
            for j in range(i + 1, min(i + 5, len(lines))):
                bold_match = re.match(r"^\*\*(.+)\*\*\s*$", lines[j].strip())
                if bold_match:
                    raw_text = bold_match.group(1).strip()
                    break
            if raw_text:
                questions.append({
                    "num": num,
                    "raw_text": raw_text,
                    "text": normalize(raw_text),
                    "file": str(filepath.relative_to(ROOT)),
                })

    return questions


def parse_core_questions_file(filepath: Path) -> list[dict]:
    """Parse the special 00_core_questions.md format (### Question N under ## Section)."""
    if not filepath.exists():
        return []

    content = filepath.read_text(encoding="utf-8")
    questions = []

    lines = content.split("\n")
    for i, line in enumerate(lines):
        q_match = re.match(r"^### Question\s+(\d+)\s*$", line.strip())
        if q_match:
            num = int(q_match.group(1))
            raw_text = ""
            for j in range(i + 1, min(i + 5, len(lines))):
                bold_match = re.match(r"^\*\*(.+)\*\*\s*$", lines[j].strip())
                if bold_match:
                    raw_text = bold_match.group(1).strip()
                    break
            if raw_text:
                questions.append({
                    "num": num,
                    "raw_text": raw_text,
                    "text": normalize(raw_text),
                    "file": str(filepath.relative_to(ROOT)),
                })

    return questions


def collect_organized_questions(paths: list[str]) -> list[dict]:
    """Collect all questions from a list of organized folder paths or file paths."""
    all_questions = []
    question_files = ["theory_questions.md", "general_questions.md",
                      "coding_questions.md", "scenario_based_questions.md"]

    for p in paths:
        full_path = ROOT / p

        if full_path.suffix == ".md":
            # It's a direct file (like 00_core_questions.md)
            if "core_questions" in p:
                all_questions.extend(parse_core_questions_file(full_path))
            else:
                all_questions.extend(parse_organized_questions_from_file(full_path))
        elif full_path.is_dir():
            # It's a folder — look for question files
            for qf in question_files:
                qf_path = full_path / qf
                if qf_path.exists():
                    all_questions.extend(parse_organized_questions_from_file(qf_path))
            # Also check for explainable_ai_questions.md pattern (flat folders)
            for f in full_path.glob("*_questions.md"):
                if f.name not in question_files:
                    all_questions.extend(parse_organized_questions_from_file(f))

    return all_questions


def fuzzy_match(text1: str, text2: str) -> float:
    """Return similarity ratio between two normalized texts."""
    return SequenceMatcher(None, text1, text2).ratio()


def audit_topic(source_file: str, organized_paths: list[str] | None) -> dict:
    """Audit a single topic. Returns a report dict."""
    source_path = SOURCE_DIR / source_file
    if not source_path.exists():
        return {"source_file": source_file, "error": "Source file not found"}

    source_qs = parse_source_questions(source_path)

    if organized_paths is None:
        return {
            "source_file": source_file,
            "source_count": len(source_qs),
            "organized_count": 0,
            "status": "NO_ORGANIZED_FOLDER",
            "missing": [{"num": q["num"], "text": q["raw_text"], "type": q["type"]} for q in source_qs],
            "reworded": [],
            "extra": [],
            "exact_matches": 0,
        }

    organized_qs = collect_organized_questions(organized_paths)

    # Match source questions to organized questions
    EXACT_THRESHOLD = 0.95
    FUZZY_THRESHOLD = 0.65

    exact_matches = []
    reworded = []
    missing = []

    for sq in source_qs:
        best_ratio = 0
        best_match = None
        for oq in organized_qs:
            ratio = fuzzy_match(sq["text"], oq["text"])
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = oq

        if best_ratio >= EXACT_THRESHOLD:
            exact_matches.append({
                "source_num": sq["num"],
                "source_text": sq["raw_text"],
                "organized_text": best_match["raw_text"],
                "file": best_match["file"],
                "ratio": round(best_ratio, 3),
            })
        elif best_ratio >= FUZZY_THRESHOLD:
            reworded.append({
                "source_num": sq["num"],
                "source_text": sq["raw_text"],
                "organized_text": best_match["raw_text"],
                "file": best_match["file"],
                "ratio": round(best_ratio, 3),
                "type": sq["type"],
            })
        else:
            missing.append({
                "source_num": sq["num"],
                "source_text": sq["raw_text"],
                "type": sq["type"],
                "section": sq["section"],
                "best_ratio": round(best_ratio, 3),
                "best_match": best_match["raw_text"] if best_match else None,
            })

    # Find extra organized questions (not matched to any source)
    matched_org_texts = set()
    for sq in source_qs:
        for oq in organized_qs:
            ratio = fuzzy_match(sq["text"], oq["text"])
            if ratio >= FUZZY_THRESHOLD:
                matched_org_texts.add(oq["text"])

    extra = []
    for oq in organized_qs:
        if oq["text"] not in matched_org_texts:
            extra.append({
                "organized_num": oq["num"],
                "organized_text": oq["raw_text"],
                "file": oq["file"],
            })

    return {
        "source_file": source_file,
        "source_count": len(source_qs),
        "organized_count": len(organized_qs),
        "exact_matches": len(exact_matches),
        "reworded_count": len(reworded),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "reworded": reworded,
        "missing": missing,
        "extra": extra,
    }


def main():
    print("=" * 80)
    print("AUDIT: Source Questions vs Organized Questions")
    print("=" * 80)

    all_reports = []
    total_source = 0
    total_organized = 0
    total_exact = 0
    total_reworded = 0
    total_missing = 0
    total_extra = 0

    for source_file, organized_paths in SOURCE_TO_ORGANIZED.items():
        report = audit_topic(source_file, organized_paths)
        all_reports.append(report)

        sc = report.get("source_count", 0)
        oc = report.get("organized_count", 0)
        ex = report.get("exact_matches", 0)
        rw = report.get("reworded_count", 0)
        mi = report.get("missing_count", 0)
        ext = report.get("extra_count", 0)

        total_source += sc
        total_organized += oc
        total_exact += ex
        total_reworded += rw
        total_missing += mi
        total_extra += ext

        status = ""
        if report.get("status") == "NO_ORGANIZED_FOLDER":
            status = " [NO FOLDER]"
        elif mi > 0 or rw > 0:
            status = " [NEEDS FIX]"
        else:
            status = " [OK]"

        print(f"\n{source_file}{status}")
        print(f"  Source: {sc} | Organized: {oc} | Exact: {ex} | Reworded: {rw} | Missing: {mi} | Extra: {ext}")

        if rw > 0:
            print(f"  --- Reworded ({rw}) ---")
            for r in report["reworded"]:
                print(f"    S#{r['source_num']}: \"{r['source_text'][:80]}\"")
                print(f"      → \"{r['organized_text'][:80]}\" ({r['ratio']}) in {r['file']}")

        if mi > 0:
            print(f"  --- Missing ({mi}) ---")
            for m in report["missing"]:
                print(f"    S#{m['source_num']}: \"{m['source_text'][:80]}\" [type={m['type']}] (best={m['best_ratio']})")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total source questions:    {total_source}")
    print(f"Total organized questions:  {total_organized}")
    print(f"Exact matches:             {total_exact}")
    print(f"Reworded (need fix):       {total_reworded}")
    print(f"Missing (need add):        {total_missing}")
    print(f"Extra (destination-only):  {total_extra}")

    # Save detailed report as JSON
    report_path = ROOT / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
