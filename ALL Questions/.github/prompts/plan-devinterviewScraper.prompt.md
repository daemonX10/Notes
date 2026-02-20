# Plan: DevInterview.io Questions Scraper

**TL;DR:** Build a Python scraper using `requests` + `BeautifulSoup` that crawls all 3 categories (DSA, System Design, ML & Data Science) from devinterview.io, extracts every question preserving the exact website structure (sub-categories, question types, numbering, premium badges), and saves each topic as a Markdown file in the corresponding existing folder (`DSA/`, `system design/`, `AI/`).

The site serves all question content in the initial HTML (SSR), so no browser automation is needed for questions-only scraping. ~80+ topic pages across 3 categories will be scraped.

## Steps

### 1. Create `scraper.py` at `e:\Interview questions\scraper.py`
- Dependencies: `requests`, `beautifulsoup4`, `time`, `os`, `re`
- Install: `pip install requests beautifulsoup4`

### 2. Define category-to-folder mapping
- `data-structures-and-algorithms` → `DSA/`
- `software-architecture-and-system-design` → `system design/`
- `machine-learning-and-data-science` → `AI/`

### 3. Step 1: Scrape category pages to discover all topic URLs
- GET each of the 3 category pages (e.g., `https://devinterview.io/questions/data-structures-and-algorithms`)
- Parse all `<a>` links matching pattern `*-interview-questions/` to get every subtopic URL
- This auto-discovers all topics (Arrays, Backtracking, ... for DSA; API Design, Caching, ... for System Design; Anomaly Detection, Apache Spark, ... for ML), so no hardcoding of topic lists

### 4. Step 2: Scrape each topic page
- GET each topic URL (e.g., `.../array-data-structure-interview-questions/`)
- Extract:
  - **Page title** from `<h1>`: e.g., "60 Arrays interview questions"
  - **Sub-category sections** from `<h2>` headings: e.g., "Fundamental Array Concepts", "Array-based Algorithms"
  - **Questions** from `<h3>` elements under each `<h2>`:
    - Question number (sequential)
    - Question text
    - Question type: "Question" (theory) or "Coding Challenge"
    - Premium status: detected by presence of lock icon SVG (`img[alt*="Lock"]`)
- Add polite delay (1-2 seconds) between requests to avoid rate-limiting

### 5. Step 3: Save as Markdown files
- One `.md` file per topic, saved in the mapped folder
- Filename: topic name cleaned (e.g., `Arrays.md`, `API Design.md`, `Anomaly Detection.md`)
- Markdown structure per file:
  ```
  # {N} {Topic} Interview Questions
  
  ## {Sub-category 1 name}
  
  ### 1. {Question text}
  **Type:** Question | Coding Challenge
  **Premium:** Yes/No
  
  ---
  
  ### 2. {Next question text}
  ...
  
  ## {Sub-category 2 name}
  
  ### 11. {Question text}
  ...
  ```
- This preserves the exact website hierarchy: Category → Topic → Sub-category → Questions

### 6. Add progress tracking and error handling
- Print progress: `[3/23] DSA > Arrays - 60 questions saved`
- Retry failed requests (up to 3 retries with exponential backoff)
- Save a `scrape_log.txt` with summary: total questions scraped per category, any failed pages
- If a page fails after retries, log it and continue (don't crash)

### 7. Add a summary index file per folder
- Generate `DSA/README.md`, `system design/README.md`, `AI/README.md`
- Contains table of all topics in that category with question counts and links to the `.md` files

## Expected Output Structure

```
e:\Interview questions\
├── scraper.py
├── scrape_log.txt
├── DSA/
│   ├── README.md
│   ├── Arrays.md
│   ├── Backtracking.md
│   ├── Big-O Notation.md
│   ├── Binary Tree.md
│   ├── ... (23 files)
│   └── Trie Data Structure.md
├── system design/
│   ├── README.md
│   ├── API Design.md
│   ├── Availability & Reliability.md
│   ├── ... (18 files)
│   └── XML.md
└── AI/
    ├── README.md
    ├── Anomaly Detection.md
    ├── Apache Spark.md
    ├── ... (65 files)
    └── XGBoost.md
```

## Verification

- Run `python scraper.py` from `e:\Interview questions\`
- Check console output for progress (should process ~106 topic pages)
- Verify file count: ~23 in DSA/, ~18 in system design/, ~65 in AI/
- Open a few `.md` files and compare question count with website page title (e.g., "60 Arrays interview questions" → 60 questions in `Arrays.md`)
- Check `scrape_log.txt` for any failed pages

## Decisions

- **Python requests + BeautifulSoup** chosen over Selenium since all question content is server-side rendered (no JS needed for questions-only)
- **Auto-discover topics** from category pages rather than hardcoding the ~106 topic URLs — this ensures no topic is missed even if the website adds new ones
- **1-2 second delay** between requests to be respectful to the server
- **Premium questions included** — they'll be scraped with a `Premium: Yes` badge (the question text is visible, only answers are gated)
