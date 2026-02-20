"""
DevInterview.io Questions Scraper
=================================
Scrapes all interview questions from 3 categories:
  - Data Structures & Algorithms → DSA/
  - Software Architecture & System Design → system design/
  - Machine Learning & Data Science → AI/

Preserves the exact website structure:
  Category → Topic → Sub-category → Questions (with type + premium status)

Usage:
  pip install requests beautifulsoup4 lxml
  python scraper.py
"""

import os
import re
import sys
import time
import logging
import requests
from datetime import datetime
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag

# ─── Configuration ──────────────────────────────────────────────────────────────

BASE_URL = "https://devinterview.io"
QUESTIONS_BASE = f"{BASE_URL}/questions"

# Category slug → local folder mapping
CATEGORIES = {
    "data-structures-and-algorithms": "DSA",
    "software-architecture-and-system-design": "system design",
    "machine-learning-and-data-science": "AI",
}

# Where to save files (directory where this script lives)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Request settings
REQUEST_DELAY = 1.5          # seconds between requests (be polite)
REQUEST_TIMEOUT = 30         # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2            # exponential backoff multiplier

# HTTP headers to mimic a real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# ─── Logging ────────────────────────────────────────────────────────────────────

LOG_FILE = os.path.join(OUTPUT_DIR, "scrape_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("scraper")

# ─── HTTP Session ───────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)


def fetch(url: str) -> BeautifulSoup | None:
    """Fetch a URL with retries and return parsed BeautifulSoup, or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            wait = RETRY_BACKOFF ** attempt
            log.warning(f"  Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt < MAX_RETRIES:
                log.info(f"  Retrying in {wait}s...")
                time.sleep(wait)
    log.error(f"  FAILED after {MAX_RETRIES} attempts: {url}")
    return None


# ─── Step 1: Discover all topic URLs from category pages ────────────────────────

def discover_topics(category_slug: str) -> list[dict]:
    """
    Fetch a category page and extract all topic links.
    Returns list of dicts: { 'name': 'Arrays', 'url': 'https://...', 'slug': '...' }
    """
    url = f"{QUESTIONS_BASE}/{category_slug}"
    log.info(f"Discovering topics from: {url}")
    
    soup = fetch(url)
    if not soup:
        return []
    
    topics = []
    seen_urls = set()
    
    # Find all links that match the topic URL pattern: *-interview-questions/
    pattern = re.compile(
        rf"/questions/{re.escape(category_slug)}/([a-z0-9-]+)-interview-questions/?$"
    )
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Normalize to absolute URL
        full_url = urljoin(BASE_URL, href)
        
        match = pattern.search(full_url)
        if match and full_url not in seen_urls:
            seen_urls.add(full_url)
            topic_slug = match.group(1)
            
            # Extract clean topic name from the link text
            link_text = a_tag.get_text(strip=True)
            # Remove "topic icon" prefix if present
            name = re.sub(r"^topic\s*icon\s*", "", link_text, flags=re.IGNORECASE).strip()
            
            if not name:
                # Fallback: derive name from slug
                name = topic_slug.replace("-", " ").title()
            
            # Ensure URL has trailing slash
            if not full_url.endswith("/"):
                full_url += "/"
            
            topics.append({
                "name": name,
                "url": full_url,
                "slug": topic_slug,
            })
    
    log.info(f"  Found {len(topics)} topics in '{category_slug}'")
    return topics


# ─── Step 2: Scrape a single topic page ─────────────────────────────────────────

def scrape_topic(topic_url: str) -> dict | None:
    """
    Scrape a topic page and return structured data.

    DOM structure (verified):
      main > section > div (one per subcategory group)
        ├── header.GroupTitle > h2 (subcategory name)
        └── ol/ul > li > article.Question
              ├── .QuestionNumber       → "1."
              ├── h3 (inside .QuestionTitle > .QuestionTitleContent)
              ├── .QuestionAttributes   → "Question" | "Coding Challenge"
              ├── .QuestionLocked       → present if premium
              └── .categoryQuestion | .categoryChallenge  (CSS class on article)
    """
    soup = fetch(topic_url)
    if not soup:
        return None
    
    # ── Extract page title ──
    # The correct H1 is inside div.TopicTitleMain (not the hero section)
    title_div = soup.find("div", class_="TopicTitleMain")
    if title_div:
        h1 = title_div.find("h1")
    else:
        # Fallback: find H1 containing "interview questions"
        h1 = None
        for candidate in soup.find_all("h1"):
            if "interview questions" in candidate.get_text(strip=True).lower():
                h1 = candidate
                break
    
    title = h1.get_text(separator=" ", strip=True) if h1 else "Interview Questions"
    # Clean up whitespace artifacts from SSR
    title = re.sub(r"\s+", " ", title).strip()
    
    # Try to extract total question count from title like "60 Arrays interview questions"
    count_match = re.match(r"(\d+)\s+", title)
    expected_count = int(count_match.group(1)) if count_match else None
    
    # ── Parse subcategories and questions ──
    # Find main > section, then iterate its direct child divs (one per group)
    main_tag = soup.find("main")
    if not main_tag:
        log.warning(f"  No <main> tag found on {topic_url}")
        return None
    
    section = main_tag.find("section")
    if not section:
        log.warning(f"  No <section> in <main> on {topic_url}")
        return None
    
    subcategories = []
    
    # Each direct child div of section = one subcategory group
    group_divs = [c for c in section.children if isinstance(c, Tag) and c.name == "div"]
    
    for group_div in group_divs:
        # Get subcategory name from header.GroupTitle > h2
        group_header = group_div.find("header", class_="GroupTitle")
        if not group_header:
            continue
        h2 = group_header.find("h2")
        if not h2:
            continue
        
        subcat_name = re.sub(r"\s+", " ", h2.get_text(separator=" ", strip=True)).strip()
        if not subcat_name:
            continue
        
        # Get all questions (article.Question) in this group
        articles = group_div.find_all("article", class_="Question")
        
        questions = []
        for article in articles:
            # ── Question number ──
            num_el = article.find(class_="QuestionNumber")
            q_number = None
            if num_el:
                num_text = num_el.get_text(strip=True)
                num_match = re.match(r"(\d+)", num_text)
                if num_match:
                    q_number = int(num_match.group(1))
            
            # ── Question text ──
            h3 = article.find("h3")
            if not h3:
                continue
            q_text = re.sub(r"\s+", " ", h3.get_text(separator=" ", strip=True)).strip()
            if not q_text or len(q_text) < 3:
                continue
            
            # ── Question type ──
            # Check .QuestionAttributes text, or article CSS classes
            q_type = "Question"  # default
            
            type_el = article.find(class_="QuestionAttributes")
            if type_el:
                type_text = type_el.get_text(strip=True)
                if "Coding Challenge" in type_text:
                    q_type = "Coding Challenge"
                elif "Question" in type_text:
                    q_type = "Question"
            else:
                # Fallback: check article CSS classes
                art_classes = article.get("class", [])
                if "categoryChallenge" in art_classes:
                    q_type = "Coding Challenge"
                elif "categoryQuestion" in art_classes:
                    q_type = "Question"
            
            # ── Premium status ──
            # Premium questions have a .QuestionLocked element
            is_premium = article.find(class_="QuestionLocked") is not None
            
            # Fallback: check for lock icon image
            if not is_premium:
                lock_img = article.find("img", alt=lambda x: x and "lock" in x.lower())
                is_premium = lock_img is not None
            
            questions.append({
                "number": q_number,
                "text": q_text,
                "type": q_type,
                "premium": is_premium,
            })
        
        if questions:
            subcategories.append({
                "name": subcat_name,
                "questions": questions,
            })
    
    # Fix any missing question numbers (sequential fallback)
    counter = 0
    for sc in subcategories:
        for q in sc["questions"]:
            counter += 1
            if q["number"] is None:
                q["number"] = counter
            else:
                counter = q["number"]
    
    # Calculate actual total
    actual_total = sum(len(sc["questions"]) for sc in subcategories)
    
    return {
        "title": title,
        "expected_count": expected_count,
        "actual_count": actual_total,
        "subcategories": subcategories,
    }


# ─── Step 3: Save topic as Markdown ─────────────────────────────────────────────

def save_topic_markdown(
    topic_name: str,
    topic_url: str,
    data: dict,
    folder: str,
) -> str:
    """Save scraped topic data as a Markdown file. Returns the file path."""
    
    # Clean filename (remove characters invalid on Windows)
    safe_name = re.sub(r'[<>:"/\\|?*]', '-', topic_name)
    safe_name = safe_name.strip(". ")
    filename = f"{safe_name}.md"
    filepath = os.path.join(folder, filename)
    
    lines = []
    
    # Header
    lines.append(f"# {data['title']}")
    lines.append("")
    lines.append(f"> Source: [{topic_url}]({topic_url})")
    lines.append(f"> Scraped: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> Total Questions: {data['actual_count']}")
    if data['expected_count'] and data['expected_count'] != data['actual_count']:
        lines.append(f"> Expected: {data['expected_count']} (some may be hidden)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    for i, subcat in enumerate(data["subcategories"], 1):
        anchor = re.sub(r"[^\w\s-]", "", subcat["name"].lower())
        anchor = re.sub(r"\s+", "-", anchor).strip("-")
        q_count = len(subcat["questions"])
        lines.append(f"{i}. [{subcat['name']}](#{anchor}) ({q_count} questions)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Subcategories and questions
    for subcat in data["subcategories"]:
        lines.append(f"## {subcat['name']}")
        lines.append("")
        
        for q in subcat["questions"]:
            # Question header with number
            premium_badge = " 🔒" if q["premium"] else ""
            lines.append(f"### {q['number']}. {q['text']}{premium_badge}")
            lines.append("")
            
            # Type badge
            if q["type"] == "Coding Challenge":
                lines.append(f"**Type:** 💻 Coding Challenge")
            else:
                lines.append(f"**Type:** 📝 Question")
            
            if q["premium"]:
                lines.append(f"**Access:** 🔒 Premium")
            
            lines.append("")
            lines.append("---")
            lines.append("")
    
    # Write file
    os.makedirs(folder, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return filepath


# ─── Step 4: Generate README index for each category ────────────────────────────

def generate_readme(
    category_name: str,
    category_slug: str,
    folder: str,
    topics_data: list[dict],
):
    """Generate a README.md index file for a category folder."""
    filepath = os.path.join(folder, "README.md")
    
    lines = []
    lines.append(f"# {category_name} - Interview Questions")
    lines.append("")
    lines.append(f"> Source: [devinterview.io/questions/{category_slug}]"
                 f"(https://devinterview.io/questions/{category_slug})")
    lines.append(f"> Scraped: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Summary
    total_topics = len(topics_data)
    total_questions = sum(t.get("actual_count", 0) for t in topics_data)
    total_coding = sum(
        sum(1 for sc in t.get("subcategories", []) for q in sc["questions"] if q["type"] == "Coding Challenge")
        for t in topics_data
    )
    total_theory = total_questions - total_coding
    
    lines.append(f"**Total Topics:** {total_topics}")
    lines.append(f"**Total Questions:** {total_questions}")
    lines.append(f"**Theory Questions:** {total_theory}")
    lines.append(f"**Coding Challenges:** {total_coding}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Topic table
    lines.append("| # | Topic | Questions | Coding | Theory | Premium |")
    lines.append("|---|-------|-----------|--------|--------|---------|")
    
    for i, t in enumerate(topics_data, 1):
        name = t["name"]
        safe_name = re.sub(r'[<>:"/\\|?*]', '-', name).strip(". ")
        link = f"[{name}]({safe_name}.md)"
        total = t.get("actual_count", 0)
        coding = sum(
            1 for sc in t.get("subcategories", [])
            for q in sc["questions"]
            if q["type"] == "Coding Challenge"
        )
        theory = total - coding
        premium = sum(
            1 for sc in t.get("subcategories", [])
            for q in sc["questions"]
            if q["premium"]
        )
        lines.append(f"| {i} | {link} | {total} | {coding} | {theory} | {premium} |")
    
    lines.append("")
    
    # Subcategory breakdown per topic
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Breakdown")
    lines.append("")
    
    for t in topics_data:
        name = t["name"]
        lines.append(f"### {name}")
        lines.append("")
        for sc in t.get("subcategories", []):
            q_count = len(sc["questions"])
            lines.append(f"- **{sc['name']}** ({q_count} questions)")
        lines.append("")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    log.info(f"  README saved: {filepath}")


# ─── Main ────────────────────────────────────────────────────────────────────────

CATEGORY_DISPLAY_NAMES = {
    "data-structures-and-algorithms": "Data Structures & Algorithms",
    "software-architecture-and-system-design": "Software Architecture & System Design",
    "machine-learning-and-data-science": "Machine Learning & Data Science",
}


def main():
    start_time = time.time()
    
    log.info("=" * 70)
    log.info("DevInterview.io Questions Scraper")
    log.info("=" * 70)
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info(f"Categories: {len(CATEGORIES)}")
    log.info("")
    
    grand_total_topics = 0
    grand_total_questions = 0
    failed_pages = []
    
    for cat_slug, cat_folder_name in CATEGORIES.items():
        cat_display = CATEGORY_DISPLAY_NAMES.get(cat_slug, cat_slug)
        cat_folder = os.path.join(OUTPUT_DIR, cat_folder_name)
        os.makedirs(cat_folder, exist_ok=True)
        
        log.info("─" * 70)
        log.info(f"CATEGORY: {cat_display}")
        log.info(f"  Folder: {cat_folder}")
        log.info("─" * 70)
        
        # Step 1: Discover topics
        topics = discover_topics(cat_slug)
        time.sleep(REQUEST_DELAY)
        
        if not topics:
            log.error(f"  No topics found for {cat_slug}! Skipping.")
            failed_pages.append(f"Category: {cat_slug}")
            continue
        
        cat_total_questions = 0
        cat_topics_data = []
        
        # Step 2: Scrape each topic
        for idx, topic in enumerate(topics, 1):
            progress = f"[{idx}/{len(topics)}]"
            log.info(f"  {progress} Scraping: {topic['name']} ...")
            
            data = scrape_topic(topic["url"])
            time.sleep(REQUEST_DELAY)
            
            if data is None:
                log.error(f"  {progress} FAILED: {topic['name']}")
                failed_pages.append(f"{cat_display} > {topic['name']} ({topic['url']})")
                continue
            
            # Add topic name to data for README generation
            data["name"] = topic["name"]
            data["url"] = topic["url"]
            
            # Save as markdown
            filepath = save_topic_markdown(
                topic_name=topic["name"],
                topic_url=topic["url"],
                data=data,
                folder=cat_folder,
            )
            
            q_count = data["actual_count"]
            expected = data.get("expected_count")
            
            # Log with match status
            if expected and expected != q_count:
                log.info(
                    f"  {progress} ✓ {topic['name']} - {q_count} questions "
                    f"(expected {expected}) → {os.path.basename(filepath)}"
                )
            else:
                log.info(
                    f"  {progress} ✓ {topic['name']} - {q_count} questions "
                    f"→ {os.path.basename(filepath)}"
                )
            
            # Log subcategory breakdown
            for sc in data["subcategories"]:
                sc_q = len(sc["questions"])
                log.info(f"           └─ {sc['name']}: {sc_q} questions")
            
            cat_total_questions += q_count
            cat_topics_data.append(data)
        
        # Step 3: Generate README index
        generate_readme(
            category_name=cat_display,
            category_slug=cat_slug,
            folder=cat_folder,
            topics_data=cat_topics_data,
        )
        
        grand_total_topics += len(cat_topics_data)
        grand_total_questions += cat_total_questions
        
        log.info("")
        log.info(f"  Category Summary: {len(cat_topics_data)} topics, {cat_total_questions} questions")
        log.info("")
    
    # ── Final Summary ──
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    log.info("=" * 70)
    log.info("SCRAPING COMPLETE")
    log.info("=" * 70)
    log.info(f"  Total Topics:    {grand_total_topics}")
    log.info(f"  Total Questions: {grand_total_questions}")
    log.info(f"  Failed Pages:    {len(failed_pages)}")
    log.info(f"  Time Elapsed:    {minutes}m {seconds}s")
    log.info(f"  Log File:        {LOG_FILE}")
    
    if failed_pages:
        log.info("")
        log.info("FAILED PAGES:")
        for fp in failed_pages:
            log.info(f"  ✗ {fp}")
    
    log.info("")
    log.info("Files saved to:")
    for cat_slug, cat_folder_name in CATEGORIES.items():
        cat_folder = os.path.join(OUTPUT_DIR, cat_folder_name)
        if os.path.exists(cat_folder):
            files = [f for f in os.listdir(cat_folder) if f.endswith(".md")]
            log.info(f"  {cat_folder_name}/  →  {len(files)} files")
    
    log.info("=" * 70)


if __name__ == "__main__":
    main()
