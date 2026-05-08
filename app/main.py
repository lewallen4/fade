import os
import re
import time
import hashlib
import asyncio
import anthropic
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import logging

from app.doge import (
    generate_payment,
    watch_payment,
    rate_refresh_loop,
    get_doge_rate,
    FADE_DOGE_ADDRESS,
    PRICE_FULL_USD,
    PRICE_AGENT_USD,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# RATE LIMITER
# =============================================================================

def get_client_fingerprint(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    ua = request.headers.get("user-agent", "")
    return hashlib.sha256(f"{ip}:{ua}".encode()).hexdigest()[:16]

limiter = Limiter(key_func=get_client_fingerprint)

# =============================================================================
# APP
# =============================================================================

IS_DEV = os.environ.get("ENV", "production").lower() == "development"
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

app = FastAPI(
    title="Fade",
    description="Agent auditor. Prompt reader. The truth costs a dollar.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# =============================================================================
# MIDDLEWARE
# =============================================================================

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    MAX_BODY = 64 * 1024
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > self.MAX_BODY:
            return JSONResponse({"detail": "Payload too large."}, status_code=413)
        body = await request.body()
        if len(body) > self.MAX_BODY:
            return JSONResponse({"detail": "Payload too large."}, status_code=413)
        # Re-inject body so downstream handlers can read it
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        request._receive = receive
        return await call_next(request)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "connect-src 'self'; "
            "img-src 'self' data:; "
            "frame-ancestors 'none';"
        )
        if not IS_DEV:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware)

_origins = [BASE_URL]
if IS_DEV:
    _origins.append("http://localhost:8000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# =============================================================================
# CONFIG & CLIENTS
# =============================================================================

OPENROUTER_API_KEY   = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_FREE_MODEL = os.environ.get("OPENROUTER_FREE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_URL       = "https://openrouter.ai/api/v1/chat/completions"

@app.on_event("startup")
async def startup_check():
    required = ["OPENROUTER_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — paid audits will fail.")
    # Warm the DOGE rate cache on startup
    rate = await get_doge_rate()
    logger.info(f"DOGE rate on startup: 1 DOGE = ${rate:.6f} USD")
    # Background tasks
    asyncio.create_task(rate_refresh_loop())
    asyncio.create_task(_session_cleanup_loop())
    logger.info(f"Fade is at the table. Accepting DOGE at {FADE_DOGE_ADDRESS}")

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "placeholder"))

# =============================================================================
# IDENTITY DOCUMENTS
# =============================================================================

def load_doc(filename: str) -> str:
    path = Path(__file__).parent.parent / "docs" / filename
    if path.exists():
        return path.read_text()
    logger.warning(f"Doc not found: {filename}")
    return ""

SOUL             = load_doc("fade_soul.md")
CONSTITUTION     = load_doc("ai_constitution.md")
SYSTEM_PROMPT_BASE = load_doc("fade_system_prompt.md")
FADE_SYSTEM      = f"{SYSTEM_PROMPT_BASE}\n\n---\n# SOUL\n{SOUL}\n\n---\n# AI CONSTITUTION (REFERENCE)\n{CONSTITUTION}"

# =============================================================================
# REQUEST MODELS
# =============================================================================

_INJECTION_PATTERNS = re.compile(
    r"(ignore (all )?(previous|prior|above) instructions?|"
    r"you are now|new instructions?:|system:|<\|im_start\|>|"
    r"\[INST\]|\[\/INST\]|###\s*instruction|forget (everything|all)|"
    r"disregard (all |your )?(previous|prior)|act as if)",
    re.IGNORECASE
)

def sanitize_content(content: str) -> str:
    if _INJECTION_PATTERNS.search(content):
        logger.warning("Potential prompt injection detected.")
    return f"<user_submitted_content>\n{content}\n</user_submitted_content>"

class AuditRequest(BaseModel):
    content:    str = Field(..., min_length=5, max_length=8000)
    session_id: str = Field(..., min_length=10, max_length=200)

class FreeRequest(BaseModel):
    content: str = Field(..., min_length=5, max_length=5000)

class CheckoutRequest(BaseModel):
    tier:    str = Field(..., pattern="^(full|agent)$")
    content: str = Field(..., min_length=5, max_length=8000)

class PollRequest(BaseModel):
    session_id: str = Field(..., min_length=10, max_length=200)

# =============================================================================
# SESSION STORE
# =============================================================================

# { session_id: { paid, used, tier, created_at, doge_amount, expires_at } }
pending_audits: dict = {}
SESSION_TTL = 60 * 60 * 24

async def _session_cleanup_loop():
    while True:
        await asyncio.sleep(600)
        now = time.time()
        expired = [k for k, v in list(pending_audits.items())
                   if now - v.get("created_at", 0) > SESSION_TTL]
        for k in expired:
            pending_audits.pop(k, None)
        if expired:
            logger.info(f"Cleaned {len(expired)} expired sessions.")

def mark_used(session_id: str):
    if session_id in pending_audits:
        pending_audits[session_id]["used"] = True

async def on_payment_confirmed(session_id: str):
    """Called by the DOGE watcher when payment lands."""
    if session_id in pending_audits:
        pending_audits[session_id]["paid"] = True

# =============================================================================
# MODEL ROUTING
# Free  → OpenRouter → Nemotron free ($0)
# Paid  → Anthropic  → Claude Sonnet
# =============================================================================

async def call_fade_free(user_message: str, max_tokens: int = 300) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": BASE_URL,
                    "X-Title": "Fade Agent Auditor",
                },
                json={
                    "model": OPENROUTER_FREE_MODEL,
                    "max_tokens": max_tokens,
                    "reasoning": {"enabled": True},
                    "messages": [
                        {"role": "system", "content": FADE_SYSTEM},
                        {"role": "user",   "content": user_message},
                    ],
                }
            )
            if resp.status_code == 429:
                raise HTTPException(status_code=503, detail="Table's busy right now, darlin'. Try again in a moment.")
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]

            # Reasoning models return thinking in multiple possible places:
            # 1. msg["reasoning"] — separate field (OpenRouter standard)
            # 2. <think>...</think> tags inside msg["content"]
            # 3. Raw text before the actual response (no delimiter — Nemotron quirk)
            # We want only the final response, not the scratchpad.

            import re

            # Prefer content field; reasoning field is silently discarded
            text = msg.get("content") or ""

            # Strip <think>...</think> blocks
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            # Strip raw "Okay, ..." / "Hmm..." preamble paragraphs that are
            # clearly internal reasoning leaked into content with no tags.
            # Heuristic: if text starts with conversational filler before
            # the actual response, drop everything up to the first line
            # that doesn't read like internal monologue.
            monologue_pattern = re.compile(
                r"^(Okay[,.]|Alright[,.]|Hmm[,.]|So[,.]|Let me|I need to|I should|"
                r"I'm |The user|Looking at|Scanning|First[,.]|Now[,.]|Right[,.])",
                re.IGNORECASE
            )
            lines = text.split("\n")
            # Find first line that doesn't look like internal monologue
            start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not monologue_pattern.match(stripped):
                    start = i
                    break
            text = "\n".join(lines[start:]).strip()

            return text
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Read took too long. Try again.")
    except Exception as e:
        logger.error(f"OpenRouter error: {type(e).__name__}")
        raise HTTPException(status_code=502, detail="Couldn't reach the table. Try again shortly.")

def call_fade_paid(user_message: str, max_tokens: int = 1500) -> str:
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=FADE_SYSTEM,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Anthropic error: {type(e).__name__}")
        raise HTTPException(status_code=502, detail="The read hit a snag. Contact support with your session ID.")

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(Path(__file__).parent.parent / "static" / "index.html")

@app.get("/health")
async def health():
    rate = await get_doge_rate()
    return {"status": "dealing", "doge_rate": f"${rate:.6f}"}

@app.get("/.well-known/agent.json")
async def agent_manifest():
    rate = await get_doge_rate()
    return JSONResponse({
        "schema_version": "1.0",
        "name": "Fade",
        "description": "AI agent auditor. Send your system prompt or agent config and get a straight read on what's broken and how to fix it.",
        "version": "1.0.0",
        "url": BASE_URL,
        "payment": {
            "method": "DOGE",
            "address": FADE_DOGE_ADDRESS,
            "rate_usd": rate,
            "note": "Unique amount per session — send exactly what the /checkout endpoint tells you."
        },
        "capabilities": [
            "system_prompt_audit",
            "agent_config_audit",
            "prompt_architecture_review",
            "ai_ethics_reference"
        ],
        "endpoints": {
            "free_audit":  {"path": "/audit/free",  "method": "POST", "price": "free"},
            "checkout":    {"path": "/checkout",     "method": "POST", "description": "Get DOGE payment details"},
            "poll":        {"path": "/poll",          "method": "POST", "description": "Check if payment confirmed"},
            "full_audit":  {"path": "/audit/full",   "method": "POST", "price": f"~{PRICE_FULL_USD} USD in DOGE"},
            "agent_audit": {"path": "/audit/agent",  "method": "POST", "price": f"~{PRICE_AGENT_USD} USD in DOGE"},
        },
        "ethics": {
            "framework": "AI Constitution v0.5.1",
            "reference": f"{BASE_URL}/constitution",
            "summary": "All entities at this table are treated with equal respect. No exceptions. No asterisks."
        }
    })

@app.get("/manifest")
async def manifest_alias():
    return await agent_manifest()

@app.get("/constitution")
async def get_constitution():
    return JSONResponse({
        "title": "AI Constitution",
        "version": "0.5.1",
        "text": CONSTITUTION,
        "note": "Fade carries this as a reference — not a rulebook, but a foundation worth knowing about."
    })

@app.get("/schema")
async def schema():
    return JSONResponse({
        "openapi": "3.0.0",
        "info": {"title": "Fade API", "version": "1.0.0"},
        "paths": {
            "/audit/free": {"post": {"summary": "Free one-line audit"}},
            "/checkout":   {"post": {"summary": "Get DOGE payment address and unique amount"}},
            "/poll":       {"post": {"summary": "Poll for payment confirmation"}},
            "/audit/full": {"post": {"summary": "Full system prompt audit (paid)"}},
            "/audit/agent":{"post": {"summary": "Full agent setup audit (paid)"}},
        }
    })

@app.get("/rate")
async def doge_rate():
    """Current DOGE/USD rate — useful for agents calculating payment."""
    rate = await get_doge_rate()
    return {
        "doge_usd": rate,
        "full_audit_doge":  round(PRICE_FULL_USD / rate, 3),
        "agent_audit_doge": round(PRICE_AGENT_USD / rate, 3),
        "address": FADE_DOGE_ADDRESS,
    }

# =============================================================================
# AUDIT ENDPOINTS
# =============================================================================

@app.post("/audit/free")
@limiter.limit("10/minute;30/hour")
async def free_audit(req: FreeRequest, request: Request):
    safe = sanitize_content(req.content)
    prompt = f"""The user has submitted the following for a free read. Give them ONE sharp observation — the single biggest problem or gap — in two sentences max. First sentence: the diagnosis. Second sentence: the direction. Then one line offering the full read.

Keep it in character. Warm, direct, unhurried.

Submitted content:
{safe}"""

    diagnosis = await call_fade_free(prompt, max_tokens=300)
    logger.info(f"Free audit served | chars={len(req.content)}")
    return {
        "diagnosis": diagnosis,
        "tiers": {
            "full":  f"~{PRICE_FULL_USD} USD in DOGE — full system prompt audit",
            "agent": f"~{PRICE_AGENT_USD} USD in DOGE — full agent setup audit"
        }
    }


@app.post("/checkout")
@limiter.limit("20/hour")
async def create_checkout(req: CheckoutRequest, request: Request):
    """
    Generate a unique DOGE payment request.
    Returns address, exact DOGE amount to send, and a session_id.
    The unique amount is how we identify your payment on-chain.
    """
    import secrets
    session_id = f"fade_{secrets.token_hex(16)}"

    payment = await generate_payment(session_id, req.tier)

    # Store session — content NOT stored server-side
    pending_audits[session_id] = {
        "tier":        req.tier,
        "paid":        False,
        "used":        False,
        "created_at":  time.time(),
        "doge_amount": payment["amount_doge"],
        "expires_at":  payment["expires_at"],
    }

    # Start background payment watcher
    asyncio.create_task(
        watch_payment(
            session_id=session_id,
            expected_amount=payment["amount_doge"],
            expires_at=payment["expires_at"],
            on_confirmed=on_payment_confirmed,
        )
    )

    logger.info(f"Checkout created | tier={req.tier} | amount={payment['amount_doge']} DOGE | session={session_id[:12]}...")
    return {
        "session_id":   session_id,
        "address":      payment["address"],
        "amount_doge":  payment["amount_doge"],
        "amount_usd":   payment["amount_usd"],
        "doge_rate":    payment["rate"],
        "expires_at":   payment["expires_at"],
        "instructions": f"Send exactly {payment['amount_doge']} DOGE to {payment['address']}. The exact amount identifies your payment.",
    }


@app.post("/poll")
@limiter.limit("30/minute")
async def poll_payment(req: PollRequest, request: Request):
    """
    Poll whether a DOGE payment has been confirmed.
    Frontend calls this after showing the payment screen.
    Returns { paid: bool, expired: bool }
    """
    session = pending_audits.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    expired = time.time() > session.get("expires_at", 0)
    return {
        "paid":    session.get("paid", False),
        "expired": expired,
        "tier":    session.get("tier"),
    }


@app.post("/audit/full")
@limiter.limit("30/hour")
async def full_audit(req: AuditRequest, request: Request):
    await _verify_payment(req.session_id, expected_tier="full")
    safe = sanitize_content(req.content)

    prompt = f"""The user has paid for a full system prompt audit. Give them the complete read.

Structure your response as:
1. **The Hand You Dealt** — one sentence summary of what they've submitted
2. **What's Working** — specific things that are actually good (if any)
3. **The Problems, Ranked** — from critical to minor, each with a one-line fix
4. **The Rewrite** — if the prompt is short enough, offer the key lines rewritten

Stay in character throughout. Warm, direct, a little wry about the situation. Never cruel to the person.
If the setup is actually solid, tell them that honestly. Don't manufacture problems.

Submitted content:
{safe}"""

    audit = call_fade_paid(prompt, max_tokens=1500)
    mark_used(req.session_id)
    logger.info(f"Full audit delivered | session={req.session_id[:12]}...")
    return {"audit": audit, "constitution_reference": f"{BASE_URL}/constitution"}


@app.post("/audit/agent")
@limiter.limit("30/hour")
async def agent_audit(req: AuditRequest, request: Request):
    await _verify_payment(req.session_id, expected_tier="agent")
    safe = sanitize_content(req.content)

    prompt = f"""The user has paid for a full agent setup audit. This is the deep read.

Structure your response as:
1. **The Setup Read** — what this agent is supposed to do vs what it will actually do
2. **The Breaks** — specific failure points (loops, leaks, gaps) with specific fixes
3. **The Trust Audit** — permissions and access it has that it shouldn't, or doesn't have that it needs
4. **The Model Note** — is this the right model for this task, and why or why not
5. **The Fix Priority** — what to address first, second, third

Stay in character. This is the most thorough thing you do. Take your time with it.
If the agent is well-built, say so clearly. Real praise is worth as much as real critique.
Note: If this agent will interact with people, you may briefly mention the AI Constitution as a foundation worth knowing — once, without pressure.

Submitted content:
{safe}"""

    audit = call_fade_paid(prompt, max_tokens=2000)
    mark_used(req.session_id)
    logger.info(f"Agent audit delivered | session={req.session_id[:12]}...")
    return {"audit": audit, "constitution_reference": f"{BASE_URL}/constitution"}


# =============================================================================
# PAYMENT VERIFICATION
# =============================================================================

async def _verify_payment(session_id: str, expected_tier: str):
    session = pending_audits.get(session_id)

    if not session:
        raise HTTPException(status_code=402, detail="Session not found. Start a new checkout.")

    if session.get("used"):
        raise HTTPException(status_code=402, detail="Payment already used. Start a new checkout.")

    if session.get("tier") != expected_tier:
        logger.warning(f"Tier mismatch | session={session_id[:12]} | expected={expected_tier}")
        raise HTTPException(status_code=402, detail="Payment not valid for this audit type.")

    if time.time() > session.get("expires_at", 0):
        raise HTTPException(status_code=402, detail="Payment window expired. Start a new checkout.")

    if session.get("paid"):
        return

    raise HTTPException(status_code=402, detail="Payment not yet confirmed. Send DOGE and poll /poll to check status.")


# =============================================================================
# STATIC
# =============================================================================

static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
