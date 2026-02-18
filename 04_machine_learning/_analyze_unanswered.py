import os
import re

base = '.'

def extract_unanswered(filepath):
    """Extract all unanswered questions (marked with 'To be filled') from a markdown file."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    results = []
    
    for i, line in enumerate(lines):
        if 'To be filled' in line:
            # Found an unanswered marker. Walk backwards to find the question text and heading.
            question_text = None
            question_heading = None
            question_line = None
            
            for j in range(i - 1, max(i - 10, -1), -1):
                stripped = lines[j].strip()
                # Look for the bold question text: **Question text?**
                if stripped.startswith('**') and stripped.endswith('**') and '?' in stripped:
                    question_text = stripped.strip('*').strip()
                    question_line = j + 1
                elif stripped.startswith('**') and stripped.endswith('**') and len(stripped) > 10:
                    question_text = stripped.strip('*').strip()
                    question_line = j + 1
                # Look for the ## heading
                if stripped.startswith('## '):
                    question_heading = stripped[3:].strip()
                    if question_line is None:
                        question_line = j + 1
                    break
            
            if question_text is None:
                question_text = question_heading or "(could not extract)"
            
            results.append({
                'answer_line': i + 1,
                'question_line': question_line or i + 1,
                'heading': question_heading,
                'question': question_text,
            })
    
    return results

# Process all files
all_results = {}
for dirpath, _, filenames in sorted(os.walk(base)):
    for fname in sorted(filenames):
        if fname.endswith('.md') and not fname.startswith('_'):
            fpath = os.path.join(dirpath, fname)
            results = extract_unanswered(fpath)
            if results:
                all_results[fpath] = results

# Print detailed results grouped by file
grand_total = 0
for fpath, results in sorted(all_results.items()):
    print(f'\n{"="*80}')
    print(f'FILE: {fpath}')
    print(f'UNANSWERED: {len(results)}')
    print(f'{"="*80}')
    for r in results:
        q = r['question'][:130]
        heading = r['heading'] or ''
        print(f'  L{r["question_line"]}: [{heading}] {q}')
    grand_total += len(results)

# Summary by file
print(f'\n\n{"="*80}')
print(f'SUMMARY')
print(f'{"="*80}')
for fpath, results in sorted(all_results.items()):
    print(f'  {fpath}: {len(results)} unanswered')
print(f'\n  GRAND TOTAL: {grand_total} unanswered questions')
