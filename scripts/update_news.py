#!/usr/bin/env python3
"""Daily news updater for FADE. Fetches agentic AI headlines and updates docs/news.json."""

import json
import datetime
import os
import sys
import requests
import anthropic


HEADERS = {"User-Agent": "FADE-NewsBot/1.0 (github.com/lewallen4/fade)"}


def fetch_hn_ai_stories():
    since = int((datetime.datetime.utcnow() - datetime.timedelta(hours=36)).timestamp())
    url = (
        "https://hn.algolia.com/api/v1/search"
        f"?query=AI+agents+agentic+LLM&tags=story"
        f"&numericFilters=created_at_i>{since}&hitsPerPage=25"
    )
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        stories = [
            {"title": h["title"], "url": h.get("url", ""), "points": h.get("points", 0)}
            for h in hits
            if h.get("url")
        ]
        if stories:
            return stories
    except Exception as e:
        print(f"HN fetch failed ({e}), trying fallback...")

    # Fallback: broader HN search without date filter
    resp = requests.get(
        "https://hn.algolia.com/api/v1/search?query=agentic+AI+agents&tags=story&hitsPerPage=25",
        timeout=15,
        headers=HEADERS,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits", [])
    return [
        {"title": h["title"], "url": h.get("url", ""), "points": h.get("points", 0)}
        for h in hits
        if h.get("url")
    ]


def generate_news_entry(stories):
    client = anthropic.Anthropic()
    today = datetime.date.today().isoformat()
    stories_text = "\n".join(
        f"- {s['title']} ({s['url']})" for s in stories[:15]
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You write brief, sharp news updates for FADE — an AI agent auditor. Today is {today}.\n\n"
                    f"Here are today's top agentic/AI stories from HackerNews:\n{stories_text}\n\n"
                    "Write a news entry with:\n"
                    "1. A short, punchy title (under 10 words)\n"
                    "2. Exactly 2 paragraphs summarizing the most important agentic AI developments. "
                    "Be direct and specific — no fluff. Focus on what's actually happening in agentic AI.\n"
                    "3. The best URL from the list above as the primary source.\n\n"
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
    stories = fetch_hn_ai_stories()
    if not stories:
        print("No stories found, skipping update.")
        sys.exit(0)

    entry = generate_news_entry(stories)
    update_news_json(entry)
