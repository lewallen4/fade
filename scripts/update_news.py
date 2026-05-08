#!/usr/bin/env python3
"""Daily news updater for FADE. Fetches AI and cybersecurity headlines and updates docs/news.json."""

import json
import datetime
import os
import sys
import requests
import anthropic

HEADERS = {"User-Agent": "FADE-NewsBot/1.0 (github.com/lewallen4/fade)"}

QUERIES = [
    "artificial intelligence AI",
    "agentic AI agents LLM",
    "cybersecurity breach vulnerability hack",
    "AI safety regulation policy",
]


def fetch_stories():
    since = int((datetime.datetime.utcnow() - datetime.timedelta(hours=36)).timestamp())
    all_stories = []
    seen = set()

    for query in QUERIES:
        try:
            resp = requests.get(
                "https://hn.algolia.com/api/v1/search",
                params={
                    "query": query,
                    "tags": "story",
                    "numericFilters": f"created_at_i>{since}",
                    "hitsPerPage": 15,
                },
                headers=HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            for h in resp.json().get("hits", []):
                if h.get("url") and h["objectID"] not in seen:
                    seen.add(h["objectID"])
                    all_stories.append({
                        "title": h["title"],
                        "url": h.get("url", ""),
                        "points": h.get("points", 0),
                    })
        except Exception as e:
            print(f"Query '{query}' failed: {e}, skipping.")

    if not all_stories:
        # Fallback: undated broad search
        try:
            resp = requests.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": "AI security", "tags": "story", "hitsPerPage": 25},
                headers=HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            for h in resp.json().get("hits", []):
                if h.get("url"):
                    all_stories.append({
                        "title": h["title"],
                        "url": h.get("url", ""),
                        "points": h.get("points", 0),
                    })
        except Exception as e:
            print(f"Fallback fetch failed: {e}")

    return sorted(all_stories, key=lambda x: x["points"], reverse=True)


def generate_news_entry(stories):
    client = anthropic.Anthropic()
    today = datetime.date.today().strftime("%B %d, %Y")
    stories_text = "\n".join(
        f"- [{s['points']} pts] {s['title']} ({s['url']})" for s in stories[:20]
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You write the daily news post for FADE's blog. Today is {today}.\n\n"
                    "FADE is an AI agent auditor. Its readers are developers, founders, and "
                    "tech-curious humans keeping tabs on what's happening in AI and cybersecurity. "
                    "They're smart, they're busy, and they don't need hype.\n\n"
                    f"Here are today's top stories from HackerNews:\n{stories_text}\n\n"
                    "Pick the single most interesting or important story — AI, cybersecurity, "
                    "whatever is actually worth talking about today. Write 2 paragraphs:\n\n"
                    "Paragraph 1: What happened. Be specific — real names, real numbers, "
                    "real context. Write like a sharp human blogger, not a press release. "
                    "No corporate language.\n\n"
                    "Paragraph 2: Why it matters to the people reading this. Give them a "
                    "reason to care. A little opinion is fine. Connect it to something real.\n\n"
                    "Rules: No 'In today's rapidly evolving landscape.' No 'It remains to be seen.' "
                    "No fluff openers. Just get into it.\n\n"
                    "Also write a short punchy title (under 10 words) and pick the best source URL.\n\n"
                    "Respond ONLY in this exact JSON format, nothing else:\n"
                    '{"title": "...", "summary": "paragraph one\\n\\nparagraph two", "url": "..."}'
                ),
            }
        ],
    )

    raw = message.content[0].text.strip()
    return json.loads(raw)


def update_news_json(entry):
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    news_path = os.path.join(repo_root, "docs", "news.json")

    with open(news_path) as f:
        data = json.load(f)

    entry["date"] = datetime.date.today().isoformat()
    data["posts"].insert(0, entry)

    with open(news_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"Updated news.json: {entry['title']}")


if __name__ == "__main__":
    stories = fetch_stories()
    if not stories:
        print("No stories found, skipping update.")
        sys.exit(0)

    print(f"Found {len(stories)} stories across {len(QUERIES)} queries.")
    entry = generate_news_entry(stories)
    update_news_json(entry)
