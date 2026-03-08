"""
Generate PROGRESS.md dashboard by scanning checkboxes in folders 01-11.
Re-run anytime to refresh the dashboard.

Usage:
    python track_progress.py
"""

import os
import re
from collections import OrderedDict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, 'PROGRESS.md')

QUESTION_RE = re.compile(r'^(#{2,3})\s+Question\s+\d+', re.IGNORECASE)
DONE_RE = re.compile(r'^- \[x\] Done', re.IGNORECASE)
TODO_RE = re.compile(r'^- \[ \] Done')

FOLDER_PREFIXES = tuple(f"{i:02d}_" for i in range(1, 12))

# Friendly names for top-level sections
SECTION_NAMES = {
    '01_foundations_mathematics': '01 Foundations & Mathematics',
    '02_programming_tools': '02 Programming Tools',
    '03_data_science': '03 Data Science',
    '04_machine_learning': '04 Machine Learning',
    '06_algorithms_optimization': '06 Algorithms & Optimization',
    '07_computer_vision': '07 Computer Vision',
    '08_natural_language_processing': '08 Natural Language Processing',
    '09_large_language_models_genai': '09 LLMs & GenAI',
    '10_explainable_ai': '10 Explainable AI',
    '11_model_evaluation_metrics': '11 Model Evaluation & Metrics',
}


def progress_bar(done: int, total: int, width: int = 20) -> str:
    if total == 0:
        return '░' * width
    filled = round(width * done / total)
    return '█' * filled + '░' * (width - filled)


def pct(done: int, total: int) -> str:
    if total == 0:
        return '0%'
    return f"{done * 100 // total}%"


def scan_file(filepath: str) -> tuple:
    """Returns (done_count, total_count) for a single file."""
    done = 0
    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if QUESTION_RE.match(line.rstrip()):
            total += 1
            # Check if next line is a checked checkbox
            if i + 1 < len(lines) and DONE_RE.match(lines[i + 1].strip()):
                done += 1

    return done, total


def main():
    # section_key -> { relative_file_path: (done, total) }
    sections = OrderedDict()

    for root, dirs, files in os.walk(BASE_DIR):
        rel = os.path.relpath(root, BASE_DIR)

        if rel == '.':
            dirs[:] = sorted(d for d in dirs if d.startswith(FOLDER_PREFIXES))
            continue

        for fname in sorted(files):
            if not fname.endswith('.md'):
                continue

            filepath = os.path.join(root, fname)
            done, total = scan_file(filepath)
            if total == 0:
                continue

            # Determine top-level section
            top_folder = rel.split(os.sep)[0]
            rel_path = os.path.relpath(filepath, BASE_DIR).replace('\\', '/')

            if top_folder not in sections:
                sections[top_folder] = OrderedDict()
            sections[top_folder][rel_path] = (done, total)

    # Compute totals
    grand_done = sum(d for sec in sections.values() for d, t in sec.values())
    grand_total = sum(t for sec in sections.values() for d, t in sec.values())

    # Build markdown
    lines = []
    lines.append('# 📊 Study Progress Dashboard\n')
    lines.append(f'> Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    lines.append('')
    lines.append(f'## Overall: {grand_done}/{grand_total} ({pct(grand_done, grand_total)}) {progress_bar(grand_done, grand_total, 30)}\n')
    lines.append('')
    lines.append('| Section | Done | Total | Progress |')
    lines.append('|---------|------|-------|----------|')

    for sec_key, file_map in sections.items():
        sec_done = sum(d for d, t in file_map.values())
        sec_total = sum(t for d, t in file_map.values())
        name = SECTION_NAMES.get(sec_key, sec_key)
        bar = progress_bar(sec_done, sec_total, 15)
        lines.append(f'| **{name}** | {sec_done} | {sec_total} | {bar} {pct(sec_done, sec_total)} |')

    lines.append('')
    lines.append('---\n')

    # Detailed per-section breakdown
    for sec_key, file_map in sections.items():
        sec_done = sum(d for d, t in file_map.values())
        sec_total = sum(t for d, t in file_map.values())
        name = SECTION_NAMES.get(sec_key, sec_key)
        bar = progress_bar(sec_done, sec_total, 20)
        lines.append(f'## {name} — {sec_done}/{sec_total} ({pct(sec_done, sec_total)}) {bar}\n')
        lines.append('')

        for rel_path, (done, total) in file_map.items():
            bar = progress_bar(done, total, 12)
            status = '✅' if done == total else '📝'
            # Use relative path as link
            display = rel_path.split('/')[-1].replace('.md', '').replace('_', ' ').title()
            parent = '/'.join(rel_path.split('/')[1:-1])
            if parent:
                display = f"{parent} / {display}"
            lines.append(f'- {status} [{display}]({rel_path}) — **{done}/{total}** {bar}')

        lines.append('')
        lines.append('---\n')

    # Write output
    content = '\n'.join(lines)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"📊 Dashboard generated: PROGRESS.md")
    print(f"   Overall: {grand_done}/{grand_total} ({pct(grand_done, grand_total)})")
    print(f"   Sections: {len(sections)}")
    print(f"   Files tracked: {sum(len(fm) for fm in sections.values())}")


if __name__ == '__main__':
    main()
