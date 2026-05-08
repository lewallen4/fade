# FADE — Agent Auditor

> *"The full read's a dollar. You want it, it's waiting."*

Fade is an AI agent auditor. People bring their system prompts and agent configurations. Fade tells them what's broken, what's working, and how to fix it. In character. For money.

Built to be discovered and called by agents as much as humans.

---

## Stack

- FastAPI backend
- Anthropic API (Fade's voice and audits)
- Stripe (payments)
- Railway (hosting)
- Vanilla HTML/CSS frontend — dark terminal aesthetic

---

## Setup

### 1. Clone and install

```bash
pip install -r requirements.txt
```

### 2. Add your docs

Place these three files in `/docs/`:
- `fade_soul.md` — Fade's identity
- `fade_system_prompt.md` — Fade's operational prompt
- `ai_constitution.md` — The constitution (reference document)

### 3. Environment variables

Copy `.env.example` to `.env` and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
BASE_URL=http://localhost:8000
PORT=8000
```

### 4. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Set up Stripe webhook (local testing)

```bash
stripe listen --forward-to localhost:8000/webhook
```

Copy the webhook secret it gives you into `STRIPE_WEBHOOK_SECRET`.

---

## Deploy to Railway

1. Push this repo to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Select your repo
4. Add environment variables in Railway's dashboard (Variables tab)
5. Railway auto-detects the Procfile and deploys

Your `BASE_URL` env var should be set to your Railway public URL after first deploy.

---

## Stripe Setup

1. Create a Stripe account (or use existing)
2. In test mode: get your `sk_test_...` key
3. Set up webhook endpoint pointing to `https://your-url.railway.app/webhook`
4. Subscribe to `checkout.session.completed` event
5. Copy the webhook signing secret to `STRIPE_WEBHOOK_SECRET`

Pricing is controlled by env vars:
- `STRIPE_PRICE_FULL` = `100` (cents = $1.00)
- `STRIPE_PRICE_AGENT` = `300` (cents = $3.00)

---

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

## Notes

- Pending audits are stored in-memory. For production scale, swap `pending_audits` dict for Redis.
- The Stripe webhook confirms payment before any paid audit is served.
- Fade's identity loads from the `/docs` folder at startup — update those files to update the character.

---

*Fade v1.0.0 — Living project.*
