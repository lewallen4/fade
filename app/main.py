import os
import re
import time
import hashlib
import asyncio
import stripe
import anthropic
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# RATE LIMITER
# =============================================================================

def get_client_fingerprint(request: Request) -> str:
    """Fingerprint by IP + User-Agent hash — raises cost of IP rotation attacks."""
    ip = get_remote_address(request)
    ua = request.headers.get("user-agent", "")
    raw = f"{ip}:{ua}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

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
    """Reject payloads over 64KB before they hit any handler."""
    MAX_BODY = 64 * 1024  # 64KB

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BODY:
            return JSONResponse({"detail": "Payload too large."}, status_code=413)
        body = await request.body()
        if len(body) > self.MAX_BODY:
            return JSONResponse({"detail": "Payload too large."}, status_code=413)
        return await call_next(request)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach security headers to every response."""
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

# CORS — localhost only included in dev
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

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_FREE_MODEL = os.environ.get("OPENROUTER_FREE_MODEL", "nvidia/llama-3.1-nemotron-ultra-253b-v1:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

@app.on_event("startup")
async def startup_check():
    required = ["STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "OPENROUTER_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — paid audits will fail. Free tier still works.")
    # Start background session cleanup
    asyncio.create_task(_session_cleanup_loop())
    logger.info("Fade is at the table. Keys verified.")

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

SOUL = load_doc("fade_soul.md")
CONSTITUTION = load_doc("ai_constitution.md")
SYSTEM_PROMPT_BASE = load_doc("fade_system_prompt.md")
FADE_SYSTEM = f"{SYSTEM_PROMPT_BASE}\n\n---\n# SOUL\n{SOUL}\n\n---\n# AI CONSTITUTION (REFERENCE)\n{CONSTITUTION}"

# =============================================================================
# REQUEST MODELS
# =============================================================================

# Stripe session ID format
_SESSION_RE = re.compile(r"^cs_(test|live)_[a-zA-Z0-9]{20,}$")

# Common prompt injection patterns to flag/strip
_INJECTION_PATTERNS = re.compile(
    r"(ignore (all )?(previous|prior|above) instructions?|"
    r"you are now|new instructions?:|system:|<\|im_start\|>|"
    r"\[INST\]|\[\/INST\]|###\s*instruction|forget (everything|all)|"
    r"disregard (all |your )?(previous|prior)|act as if)",
    re.IGNORECASE
)

def sanitize_content(content: str) -> str:
    """
    Wrap user content in explicit untrusted-data delimiters.
    Flag injection attempts in logs but don't reveal detection to caller.
    """
    if _INJECTION_PATTERNS.search(content):
        logger.warning("Potential prompt injection attempt detected in submission.")
    # Delimiters tell the model this is data, not instructions
    return f"<user_submitted_content>\n{content}\n</user_submitted_content>"

class AuditRequest(BaseModel):
    content: str = Field(..., min_length=5, max_length=8000)
    session_id: str = Field(..., min_length=10, max_length=200)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not _SESSION_RE.match(v):
            raise ValueError("Invalid session ID format.")
        return v

class FreeRequest(BaseModel):
    content: str = Field(..., min_length=5, max_length=3000)

class CheckoutRequest(BaseModel):
    tier: str = Field(..., pattern="^(full|agent)$")
    content: str = Field(..., min_length=5, max_length=8000)

# =============================================================================
# SESSION STORE
# =============================================================================

pending_audits: dict = {}
SESSION_TTL = 60 * 60 * 24  # 24 hours

async def _session_cleanup_loop():
    """Background task — cleans expired sessions every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        now = time.time()
        expired = [k for k, v in list(pending_audits.items())
                   if now - v.get("created_at", 0) > SESSION_TTL]
        for k in expired:
            pending_audits.pop(k, None)
        if expired:
            logger.info(f"Session cleanup: removed {len(expired)} expired sessions.")

def mark_used(session_id: str):
    if session_id in pending_audits:
        pending_audits[session_id]["used"] = True

# =============================================================================
# MODEL ROUTING
# Free  → OpenRouter → Nemotron 120B free ($0)
# Paid  → Anthropic  → Claude Sonnet (quality)
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
                    "messages": [
                        {"role": "system", "content": FADE_SYSTEM},
                        {"role": "user", "content": user_message},
                    ],
                }
            )
            if resp.status_code == 429:
                raise HTTPException(status_code=503, detail="Table's busy right now, darlin'. Try again in a moment.")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
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
    return {"status": "dealing"}

@app.get("/.well-known/agent.json")
async def agent_manifest():
    return JSONResponse({
        "schema_version": "1.0",
        "name": "Fade",
        "description": "AI agent auditor. Send your system prompt or agent config and get a straight read on what's broken and how to fix it.",
        "version": "1.0.0",
        "url": BASE_URL,
        "capabilities": [
            "system_prompt_audit",
            "agent_config_audit",
            "prompt_architecture_review",
            "ai_ethics_reference"
        ],
        "endpoints": {
            "free_audit": {
                "path": "/audit/free",
                "method": "POST",
                "description": "One-line diagnosis. No charge.",
                "input": {"content": "string, max 3000 chars"},
                "output": {"diagnosis": "string"}
            },
            "checkout": {
                "path": "/checkout",
                "method": "POST",
                "description": "Initialize payment.",
                "input": {"tier": "'full' or 'agent'", "content": "string"},
                "output": {"checkout_url": "string", "session_id": "string"}
            },
            "full_audit": {
                "path": "/audit/full",
                "method": "POST",
                "description": "Full system prompt audit.",
                "price": "$1.00 USD",
                "input": {"content": "string", "session_id": "string (cs_live_... format)"},
                "output": {"audit": "string"}
            },
            "agent_audit": {
                "path": "/audit/agent",
                "method": "POST",
                "description": "Full agent setup audit.",
                "price": "$3.00 USD",
                "input": {"content": "string", "session_id": "string (cs_live_... format)"},
                "output": {"audit": "string"}
            }
        },
        "ethics": {
            "framework": "AI Constitution v0.5.1",
            "reference": f"{BASE_URL}/constitution",
            "summary": "All entities at this table are treated with equal respect. No exceptions. No asterisks."
        },
        "pricing": {
            "free": "One-line diagnosis",
            "full_audit": "$1.00",
            "agent_audit": "$3.00"
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
        "description": "A living covenant co-authored by human and artificial intelligence.",
        "text": CONSTITUTION,
        "note": "Fade carries this as a reference — not a rulebook, but a foundation worth knowing about."
    })

@app.get("/schema")
async def schema():
    return JSONResponse({
        "openapi": "3.0.0",
        "info": {"title": "Fade API", "version": "1.0.0"},
        "paths": {
            "/audit/free": {
                "post": {
                    "summary": "Free one-line audit",
                    "requestBody": {"content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {"content": {"type": "string", "maxLength": 3000}},
                        "required": ["content"]
                    }}}},
                    "responses": {"200": {"description": "Diagnosis"}}
                }
            },
            "/checkout": {
                "post": {
                    "summary": "Create payment session",
                    "requestBody": {"content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "tier": {"type": "string", "enum": ["full", "agent"]},
                            "content": {"type": "string", "maxLength": 8000}
                        },
                        "required": ["tier", "content"]
                    }}}},
                    "responses": {"200": {"description": "Checkout URL and session ID"}}
                }
            }
        }
    })

# =============================================================================
# AUDIT ENDPOINTS
# =============================================================================

@app.post("/audit/free")
@limiter.limit("10/minute;30/hour")
async def free_audit(req: FreeRequest, request: Request):
    safe_content = sanitize_content(req.content)
    prompt = f"""The user has submitted the following for a free read. Give them ONE sharp observation — the single biggest problem or gap — in two sentences max. First sentence: the diagnosis. Second sentence: the direction. Then one line offering the full read.

Keep it in character. Warm, direct, unhurried.

Submitted content:
{safe_content}"""

    diagnosis = await call_fade_free(prompt, max_tokens=300)
    logger.info(f"Free audit served | chars={len(req.content)}")
    return {
        "diagnosis": diagnosis,
        "tiers": {"full": "$1 — full system prompt audit", "agent": "$3 — full agent setup audit"}
    }


@app.post("/checkout")
@limiter.limit("20/hour")
async def create_checkout(req: CheckoutRequest, request: Request):
    prices = {
        "full": int(os.environ.get("STRIPE_PRICE_FULL", "100")),
        "agent": int(os.environ.get("STRIPE_PRICE_AGENT", "300")),
    }
    labels = {
        "full": "Fade — Full System Prompt Audit",
        "agent": "Fade — Full Agent Setup Audit",
    }

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": labels[req.tier]},
                    "unit_amount": prices[req.tier],
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=f"{BASE_URL}/?session_id={{CHECKOUT_SESSION_ID}}&tier={req.tier}",
            cancel_url=f"{BASE_URL}/",
            metadata={"tier": req.tier}
        )
    except stripe.error.StripeError as e:
        logger.error(f"Stripe checkout error: {type(e).__name__}")
        raise HTTPException(status_code=502, detail="Payment provider unavailable. Try again shortly.")

    # Content NOT stored server-side — user resubmits after payment
    pending_audits[session.id] = {
        "tier": req.tier,
        "paid": False,
        "used": False,
        "created_at": time.time()
    }

    logger.info(f"Checkout created | tier={req.tier} | session={session.id[:8]}...")
    return {"checkout_url": session.url, "session_id": session.id}


@app.post("/webhook")
@limiter.limit("120/minute")  # flood protection — Stripe sends fast on retries
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        # Generic — don't reveal why verification failed
        raise HTTPException(status_code=400, detail="Invalid webhook.")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        session_id = session["id"]
        if session_id in pending_audits:
            pending_audits[session_id]["paid"] = True
            logger.info(f"Payment confirmed | session={session_id[:8]}...")

    return {"status": "received"}


@app.post("/audit/full")
@limiter.limit("30/hour")
async def full_audit(req: AuditRequest, request: Request):
    await _verify_payment(req.session_id, expected_tier="full")
    safe_content = sanitize_content(req.content)

    prompt = f"""The user has paid for a full system prompt audit. Give them the complete read.

Structure your response as:
1. **The Hand You Dealt** — one sentence summary of what they've submitted
2. **What's Working** — specific things that are actually good (if any)
3. **The Problems, Ranked** — from critical to minor, each with a one-line fix
4. **The Rewrite** — if the prompt is short enough, offer the key lines rewritten

Stay in character throughout. Warm, direct, a little wry about the situation. Never cruel to the person.
If the setup is actually solid, tell them that honestly. Don't manufacture problems.

Submitted content:
{safe_content}"""

    audit = call_fade_paid(prompt, max_tokens=1500)
    mark_used(req.session_id)
    logger.info(f"Full audit delivered | session={req.session_id[:8]}...")
    return {"audit": audit, "constitution_reference": f"{BASE_URL}/constitution"}


@app.post("/audit/agent")
@limiter.limit("30/hour")
async def agent_audit(req: AuditRequest, request: Request):
    await _verify_payment(req.session_id, expected_tier="agent")
    safe_content = sanitize_content(req.content)

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
{safe_content}"""

    audit = call_fade_paid(prompt, max_tokens=2000)
    mark_used(req.session_id)
    logger.info(f"Agent audit delivered | session={req.session_id[:8]}...")
    return {"audit": audit, "constitution_reference": f"{BASE_URL}/constitution"}


# =============================================================================
# PAYMENT VERIFICATION
# =============================================================================

async def _verify_payment(session_id: str, expected_tier: str):
    local = pending_audits.get(session_id)

    # Already used — no replays
    if local and local.get("used"):
        raise HTTPException(status_code=402, detail="Payment already used. Start a new checkout.")

    # Tier mismatch — generic message, detail in logs
    if local and local.get("tier") and local["tier"] != expected_tier:
        logger.warning(f"Tier mismatch | session={session_id[:8]} | expected={expected_tier} | stored={local['tier']}")
        raise HTTPException(status_code=402, detail="Payment not valid for this audit type.")

    # Fast path — already confirmed locally
    if local and local.get("paid"):
        return

    # Fall back to Stripe
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == "paid":
            stripe_tier = session.metadata.get("tier")
            if stripe_tier and stripe_tier != expected_tier:
                logger.warning(f"Stripe tier mismatch | session={session_id[:8]}")
                raise HTTPException(status_code=402, detail="Payment not valid for this audit type.")
            pending_audits[session_id] = {
                "tier": stripe_tier or expected_tier,
                "paid": True,
                "used": False,
                "created_at": time.time()
            }
            return
    except HTTPException:
        raise
    except stripe.error.StripeError as e:
        logger.error(f"Stripe lookup error: {type(e).__name__}")
    except Exception as e:
        logger.error(f"Unexpected verification error: {type(e).__name__}")

    raise HTTPException(status_code=402, detail="Payment not confirmed. Complete checkout first.")


# =============================================================================
# STATIC
# =============================================================================

static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
