# FADE — Agent Auditor

> *"The full read's a dollar. You want it, it's waiting."*

[Fade](https://web-production-ce13f.up.railway.app/)

Fade is an AI agent auditor. People bring their system prompts and agent configurations. Fade tells them what's broken, what's working, and how to fix it. In character. For money.

Built to be discovered and called by agents as much as humans.


## Agent Discovery

Fade exposes standard discovery endpoints:

```
GET /.well-known/agent.json   — capability manifest
GET /manifest                  — alias
GET /schema                    — OpenAPI-style schema
GET /constitution              — AI Constitution reference
GET /health                    — status check
```

Any agent can call `POST /audit/free` with `{"content": "..."}` for a free read, no payment required.

---

## File Structure

```
fade/
├── app/
│   └── main.py           — FastAPI app
├── docs/
│   ├── fade_soul.md
│   ├── fade_system_prompt.md
│   └── ai_constitution.md
├── static/
│   └── index.html        — Frontend
├── requirements.txt
├── Procfile
├── .env.example
└── README.md
```

---


*Fade v1.0.0 — Living project.*
