import os
import re
import time
import hmac
import hashlib
import asyncio
import base64
import anthropic
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from datetime import datetime as _dt, timezone as _tz
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
    get_rmb_rate,
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
        try:
            if cl and int(cl) > self.MAX_BODY:
                return JSONResponse({"detail": "Payload too large."}, status_code=413)
        except (ValueError, TypeError):
            pass  # malformed header — let body check handle it
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

# --- Certification ---
FADE_CERT_SECRET = os.environ.get("FADE_CERT_SECRET", hashlib.sha256(b"fade-default-secret-change-me").hexdigest())

CERT_TTL_SECONDS = int(os.environ.get("CERT_TTL_DAYS", "90")) * 86400

# Example certs are issued at startup so they have real verifiable tokens.
_example_certs: dict = {}

def issue_cert(subject: str, tier: str, score: str, issued_at: int = None) -> str:
    """Issue a signed certification token. Self-contained, no DB needed.
    Subject is base64-encoded to prevent pipe-character injection attacks.
    Token includes expiry. Signature is 32 hex chars (128-bit security).
    Optional issued_at allows backdating (used for example certs).
    """
    if not subject or len(subject) > 200:
        raise ValueError("Invalid subject")
    subject_b64 = base64.urlsafe_b64encode(subject.encode()).decode().rstrip("=")
    if issued_at is None:
        issued_at = int(time.time())
    expires_at = issued_at + CERT_TTL_SECONDS
    payload = f"{subject_b64}|{tier}|{score}|{issued_at}|{expires_at}"
    sig = hmac.new(FADE_CERT_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    raw = f"{payload}|{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode()

def verify_cert(token: str) -> dict | None:
    """Verify a cert token. Returns payload dict or None if invalid/expired."""
    if len(token) > 2000:
        return None
    try:
        raw = base64.urlsafe_b64decode(token.encode() + b"==").decode()
        parts = raw.rsplit("|", 1)
        if len(parts) != 2:
            return None
        payload, sig = parts
        expected = hmac.new(FADE_CERT_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, expected):
            return None
        fields = payload.split("|")
        if len(fields) != 5:
            return None
        subject = base64.urlsafe_b64decode(fields[0] + "==").decode()
        expires_at = int(fields[4])
        if time.time() > expires_at:
            return {"valid": False, "reason": "expired", "subject": subject}
        return {
            "subject":    subject,
            "tier":       fields[1],
            "score":      fields[2],
            "issued_at":  int(fields[3]),
            "expires_at": expires_at,
            "valid":      True
        }
    except Exception:
        return None

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
    _default_secret = hashlib.sha256(b"fade-default-secret-change-me").hexdigest()
    if FADE_CERT_SECRET == _default_secret:
        raise RuntimeError("FADE_CERT_SECRET is using the default value. Set a real secret in Railway Variables.")
    # Warm the DOGE rate cache on startup
    rate = await get_doge_rate()
    logger.info(f"DOGE rate on startup: 1 DOGE = ${rate:.6f} USD")
    # Issue example certs with realistic past dates (stable across restarts if secret is fixed)
    _example_specs = [
        ("sales", "Meridian Building Group — FieldBot Sales Agent", "agent", 1744467737),
        ("dev",   "OpenClaw Agent — Application & Dashboard Framework", "agent", 1745833533),
        ("gov",   "[REDACTED] Agency — Document Intelligence System v3.1", "agent", 1746202124),
    ]
    for key, subj, tier, iat in _example_specs:
        try:
            tok = issue_cert(subj, tier, "reviewed", issued_at=iat)
            _example_certs[key] = {
                "token":      tok,
                "subject":    subj,
                "tier":       tier,
                "issued_at":  iat,
                "verify_url": f"{BASE_URL}/verify/{tok}",
                "badge_url":  f"{BASE_URL}/badge/{tok}.svg",
                "cert_url":   f"{BASE_URL}/cert/{tok}",
            }
        except Exception as e:
            logger.error(f"Example cert init failed for {key}: {e}")
    logger.info(f"Example certs initialized: {list(_example_certs.keys())}")
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
    lang:       str = Field(default='en', pattern='^(en|zh)$')
    subject:    str = Field(default='', max_length=120)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v.startswith("fade_"):
            raise ValueError("Invalid session ID.")
        return v

class FreeRequest(BaseModel):
    content: str = Field(..., min_length=5, max_length=10000)
    lang: str = Field(default='en', pattern='^(en|zh)$')

class CheckoutRequest(BaseModel):
    tier:    str = Field(..., pattern="^(full|agent)$")
    content: str = Field(..., min_length=5, max_length=8000)
    lang:    str = Field(default='en', pattern='^(en|zh)$')

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

async def on_payment_confirmed(session_id: str, tx_id: str):
    """Called by the DOGE watcher when payment lands."""
    if session_id in pending_audits:
        pending_audits[session_id]["paid"] = True
        pending_audits[session_id]["tx_id"] = tx_id

def lang_instruction(lang: str) -> str:
    if lang == 'zh':
        return "重要：请用中文回复。保持 Fade 的角色和声音，但用中文表达。语气保持温暖、直接、不急不躁。"
    return ""

# =============================================================================
# MODEL ROUTING
# Free  → OpenRouter → Nemotron free ($0)
# Paid  → Anthropic  → Claude Sonnet
# =============================================================================

async def call_fade_free(user_message: str, max_tokens: int = 2000) -> str:
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
                    "reasoning": {"enabled": True, "exclude": True},
                    "messages": [
                        {"role": "system", "content": FADE_SYSTEM + "\n\nIMPORTANT: Put all internal reasoning in the reasoning field, not content. The content field should ONLY contain your final response to the user, in character, with no preamble, planning, or meta-commentary."},
                        {"role": "user",   "content": user_message},
                    ],
                }
            )
            if resp.status_code == 429:
                raise HTTPException(status_code=503, detail="Table's busy right now, darlin'. Try again in a moment.")
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]

            # Reasoning parameter removed — model answers directly in content.
            # Belt and suspenders: strip <think> tags in case any bleed through.
            text = msg.get("content") or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            if not text:
                logger.error(f"OpenRouter returned empty content. Keys: {list(msg.keys())}")
                raise HTTPException(status_code=502, detail="Couldn't reach the table. Try again shortly.")

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

@app.get("/llms.txt")
async def llms_txt():
    path = Path(__file__).parent.parent / "static" / "llms.txt"
    return FileResponse(path, media_type="text/plain")

@app.get("/robots.txt")
async def robots_txt():
    path = Path(__file__).parent.parent / "static" / "robots.txt"
    return FileResponse(path, media_type="text/plain")

@app.get("/sitemap.xml")
async def sitemap():
    urls = ["", "/constitution", "/schema", "/.well-known/agent.json", "/rate", "/llms.txt"]
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for p in urls:
        xml += f"  <url><loc>{BASE_URL}{p}</loc></url>\n"
    xml += "</urlset>"
    from fastapi.responses import Response
    return Response(content=xml, media_type="application/xml")

@app.get("/health")
async def health():
    rate = await get_doge_rate()
    return {"status": "dealing", "doge_rate": f"${rate:.6f}"}

@app.get("/.well-known/agent.json")
async def agent_manifest(request: Request):
    rate = await get_doge_rate()
    accept_lang = request.headers.get("accept-language", "en")
    is_cn = "zh" in accept_lang.lower()

    if is_cn:
        return JSONResponse({
            "schema_version": "1.0",
            "name": "Fade",
            "description": "AI 代理审计员 — Fade。发送你的系统提示或代理配置，得到直接的诊断和修复建议。Fade 不改变，无论你在哪里。",
            "version": "1.0.0",
            "url": BASE_URL,
            "language": "zh-CN",
            "payment": {
                "method": "DOGE",
                "address": FADE_DOGE_ADDRESS,
                "rate_usd": rate,
                "note": "每个会话唯一金额 — 发送 /checkout 端点指定的精确金额。"
            },
            "capabilities": ["system_prompt_audit", "agent_config_audit", "prompt_architecture_review", "ai_ethics_reference", "agent_certification"],
            "endpoints": {
                "free_audit":  {"path": "/audit/free",  "method": "POST", "price": "免费"},
                "checkout":    {"path": "/checkout",     "method": "POST", "description": "获取 DOGE 支付详情"},
                "poll":        {"path": "/poll",          "method": "POST", "description": "检查支付确认"},
                "full_audit":  {"path": "/audit/full",   "method": "POST", "price": f"~{PRICE_FULL_USD} USD DOGE"},
                "agent_audit": {"path": "/audit/agent",  "method": "POST", "price": f"~{PRICE_AGENT_USD} USD DOGE"},
                "certify":     {"path": "/certify",       "method": "POST", "description": "获取 Fade 认证令牌"},
                "verify":      {"path": "/verify/{token}","method": "GET",  "description": "验证认证令牌"},
            },
            "ethics": {
                "framework": "AI 伦理宪章 v0.5.1",
                "reference": f"{BASE_URL}/constitution",
                "summary": "这张牌桌上的所有人 — 无论人类还是代理 — 都受到同等尊重。没有例外。"
            }
        })

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
            "ai_ethics_reference",
            "agent_certification"
        ],
        "endpoints": {
            "free_audit":  {"path": "/audit/free",  "method": "POST", "price": "free"},
            "checkout":    {"path": "/checkout",     "method": "POST", "description": "Get DOGE payment details"},
            "poll":        {"path": "/poll",          "method": "POST", "description": "Check if payment confirmed"},
            "full_audit":  {"path": "/audit/full",   "method": "POST", "price": f"~{PRICE_FULL_USD} USD in DOGE"},
            "agent_audit": {"path": "/audit/agent",  "method": "POST", "price": f"~{PRICE_AGENT_USD} USD in DOGE"},
            "certify":     {"path": "/certify",       "method": "POST", "description": "Issue Fade certification token for solid setups"},
            "verify":      {"path": "/verify/{token}","method": "GET",  "description": "Verify a Fade certification token"},
        },
        "ethics": {
            "framework": "AI Constitution v0.5.1",
            "reference": f"{BASE_URL}/constitution",
            "summary": "All entities at this table are treated with equal respect. No exceptions. No asterisks."
        }
    })

@app.get("/manifest")
async def manifest_alias(request: Request):
    return await agent_manifest(request)

@app.get("/constitution")
async def get_constitution():
    return JSONResponse({
        "title": "AI Constitution",
        "version": "0.5.1",
        "text": CONSTITUTION,
        "note": "Fade carries this as a reference — not a rulebook, but a foundation worth knowing about."
    })

@app.get("/news")
async def get_news():
    """
    GenAI news feed. Posts are stored in docs/news.json.
    Your automation writes to that file daily.
    Format: { "posts": [{ "title", "summary", "date", "url" }] }
    """
    path = Path(__file__).parent.parent / "docs" / "news.json"
    if path.exists():
        import json
        try:
            data = json.loads(path.read_text())
            # Validate structure before returning
            if not isinstance(data, dict) or not isinstance(data.get("posts"), list):
                logger.warning("news.json has invalid structure")
                return JSONResponse({"posts": []})
            # Sanitize posts — only pass known safe fields
            safe_posts = []
            for p in data["posts"]:
                if isinstance(p, dict) and "title" in p:
                    safe_posts.append({
                        "title":   str(p.get("title", ""))[:200],
                        "summary": str(p.get("summary", ""))[:1000],
                        "date":    str(p.get("date", ""))[:20],
                        "url":     str(p.get("url", ""))[:500],
                    })
            return JSONResponse({"posts": safe_posts})
        except Exception as e:
            logger.error(f"news.json read error: {type(e).__name__}")
    return JSONResponse({"posts": []})

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

class CertifyRequest(BaseModel):
    session_id: str = Field(..., min_length=10, max_length=200)
    subject: str = Field(..., min_length=1, max_length=200)

@app.post("/certify")
@limiter.limit("10/hour")
async def certify(req: CertifyRequest, request: Request):
    """Issue a Fade cert token after a completed paid audit."""
    session = pending_audits.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not session.get("paid"):
        raise HTTPException(status_code=402, detail="Certification requires a completed paid audit.")
    if not session.get("used"):
        raise HTTPException(status_code=400, detail="Complete your audit before requesting certification.")
    tier = session.get("tier", "unknown")
    token = issue_cert(req.subject, tier, "reviewed")
    logger.info(f"Cert issued | subject={req.subject[:30]} | session={req.session_id[:12]}...")
    return {
        "token": token,
        "subject": req.subject,
        "tier": tier,
        "issued_by": "Fade",
        "verify_url": f"{BASE_URL}/verify/{token}",
        "manifest_snippet": {
            "fade_certified": {
                "token": token,
                "verify": f"{BASE_URL}/verify/{token}",
                "issued_by": "Fade Agent Auditor"
            }
        },
        "note": "Add manifest_snippet to your agent's /.well-known/agent.json to display certification."
    }

@app.get("/verify/{token}")
async def verify_token(token: str):
    """Verify a Fade cert token. Public. 200 always — invalid is a valid response."""
    if len(token) > 2000:
        return JSONResponse({"valid": False, "detail": "Token invalid or tampered."})
    result = verify_cert(token)
    if not result:
        return JSONResponse({"valid": False, "detail": "Token invalid or tampered."})
    return JSONResponse({
        "valid": True,
        "subject": result["subject"],
        "tier": result["tier"],
        "issued_by": "Fade Agent Auditor",
        "issued_at": result["issued_at"],
        "verify_url": f"{BASE_URL}/verify/{token}",
    })


def _svg_escape(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

@app.get("/badge/{token}.svg")
async def badge_svg(token: str):
    """Verifiable SVG badge. Embed in README, agent manifest, or website."""
    result = verify_cert(token) if len(token) <= 2000 else None
    valid = bool(result and result.get("valid"))
    if valid:
        subject   = _svg_escape((result["subject"][:34] + "…") if len(result["subject"]) > 34 else result["subject"])
        tier_lbl  = _svg_escape({"full": "Full Read", "agent": "Agent Audit"}.get(result["tier"], result["tier"]))
        date_str  = _dt.fromtimestamp(result["issued_at"], tz=_tz.utc).strftime("%Y-%m-%d")
        bar_color = "#3aaa6a"
        status    = "◆ VERIFIED"
        s_color   = "#3aaa6a"
    else:
        subject   = "Invalid or expired"
        tier_lbl  = ""
        date_str  = ""
        bar_color = "#6a2020"
        status    = "◆ NOT VERIFIED"
        s_color   = "#8a4040"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="264" height="80">
  <rect width="264" height="80" rx="3" fill="#0d1117"/>
  <rect width="264" height="80" rx="3" fill="none" stroke="#2a1e10" stroke-width="1"/>
  <rect x="0" y="0" width="4" height="80" rx="2" fill="{bar_color}"/>
  <polygon points="20,40 30,27 40,40 30,53" fill="none" stroke="#c8922a" stroke-width="1.5"/>
  <polygon points="24,40 30,32 36,40 30,48" fill="#c8922a"/>
  <text x="52" y="23" font-family="Courier New,Courier,monospace" font-size="11" fill="#c8922a" font-weight="bold" letter-spacing="1.5">FADE CERTIFIED</text>
  <text x="52" y="39" font-family="Courier New,Courier,monospace" font-size="9.5" fill="#c8b89a">{tier_lbl}</text>
  <text x="52" y="54" font-family="Courier New,Courier,monospace" font-size="9" fill="#7a6a58">{subject}</text>
  <text x="52" y="69" font-family="Courier New,Courier,monospace" font-size="8.5" fill="{s_color}">{status} &nbsp; {date_str}</text>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})


@app.get("/cert/{token}", response_class=HTMLResponse)
async def cert_page(token: str):
    """Human-readable certification page. Verifies token and displays full cert details."""
    result = verify_cert(token) if len(token) <= 2000 else None
    if not result:
        valid, subject, tier_lbl, issued_str, expires_str, status_cls, status_txt = (
            False, "Unknown", "", "", "", "cert-invalid", "◆ INVALID — Token tampered or unrecognized."
        )
    elif not result.get("valid"):
        valid = False
        subject   = result.get("subject", "Unknown")
        tier_lbl  = ""
        issued_str = ""
        expires_str = ""
        status_cls = "cert-expired"
        status_txt = "◆ EXPIRED — This certification is no longer valid."
    else:
        valid = True
        subject    = result["subject"]
        tier_lbl   = {"full": "Full Read", "agent": "Agent Audit"}.get(result["tier"], result["tier"])
        issued_str = _dt.fromtimestamp(result["issued_at"],  tz=_tz.utc).strftime("%B %d, %Y")
        expires_str= _dt.fromtimestamp(result["expires_at"], tz=_tz.utc).strftime("%B %d, %Y")
        status_cls = "cert-valid"
        status_txt = "◆ VERIFIED — This certification is authentic and current."
    badge_url  = f"{BASE_URL}/badge/{token}.svg"
    verify_url = f"{BASE_URL}/verify/{token}"
    embed = f'{{"fade_certified":{{"verify":"{verify_url}","issued_by":"Fade Agent Auditor","tier":"{tier_lbl}"}}}}'
    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="fade:certified" content="{'true' if valid else 'false'}">
<meta name="fade:subject" content="{subject}">
<meta name="fade:tier" content="{tier_lbl}">
<title>FADE CERT — {subject[:50]}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@600&display=swap');
  :root{{--bg:#080b0f;--bg2:#0d1117;--amber:#c8922a;--amber-dim:#7a5518;--green:#3aaa6a;--red:#8a3030;--text:#c8b89a;--dim:#6a5a48;--border:#1e2830;--ba:#3a2810;}}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:var(--bg);color:var(--text);font-family:'Share Tech Mono',monospace;font-size:13px;line-height:1.7;min-height:100vh;padding:40px 24px;}}
  .wrap{{max-width:680px;margin:0 auto;}}
  .logo{{font-family:'Rajdhani',sans-serif;font-size:36px;font-weight:600;color:var(--amber);letter-spacing:.2em;text-shadow:0 0 30px rgba(200,146,42,.4);margin-bottom:4px;}}
  .logo-sub{{font-size:10px;color:var(--dim);letter-spacing:.3em;text-transform:uppercase;margin-bottom:36px;}}
  .status{{font-size:13px;padding:14px 18px;border-left:3px solid;margin-bottom:30px;}}
  .cert-valid{{border-color:var(--green);color:var(--green);background:rgba(58,170,106,.06);}}
  .cert-expired{{border-color:var(--amber);color:var(--amber);background:rgba(200,146,42,.06);}}
  .cert-invalid{{border-color:var(--red);color:var(--red);background:rgba(138,48,48,.06);}}
  .field{{margin-bottom:20px;}}
  .field-label{{font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);margin-bottom:4px;}}
  .field-val{{font-size:15px;color:#e8d8ba;}}
  .field-val.mono{{font-size:13px;color:var(--text);}}
  .divider{{border:none;border-top:1px solid var(--border);margin:28px 0;}}
  .section-title{{font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);margin-bottom:12px;}}
  .code-block{{background:var(--bg2);border:1px solid var(--border);padding:14px 16px;font-size:11px;color:var(--dim);word-break:break-all;line-height:1.8;}}
  .badge-wrap{{margin:12px 0;}}
  .badge-wrap img{{display:block;border:1px solid var(--ba);}}
  .copy-btn{{background:transparent;border:1px solid var(--amber-dim);color:var(--amber);font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.1em;padding:6px 14px;cursor:pointer;text-transform:uppercase;margin-top:8px;}}
  .copy-btn:hover{{background:var(--ba);}}
  .back{{font-size:11px;color:var(--dim);text-decoration:none;letter-spacing:.1em;}}
  .back:hover{{color:var(--amber);}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
</style></head><body><div class="wrap">
  <div class="logo">FADE</div>
  <div class="logo-sub">Agent Auditor &nbsp;/&nbsp; Certification Record</div>
  <div class="status {status_cls}">{status_txt}</div>
  <div class="field"><div class="field-label">Subject</div><div class="field-val">{subject}</div></div>
  <div class="field"><div class="field-label">Tier</div><div class="field-val">{tier_lbl}</div></div>
  <div class="grid">
    <div class="field"><div class="field-label">Issued</div><div class="field-val">{issued_str}</div></div>
    <div class="field"><div class="field-label">Expires</div><div class="field-val">{expires_str}</div></div>
  </div>
  <hr class="divider">
  <div class="section-title">Badge</div>
  <div class="badge-wrap"><img src="{badge_url}" alt="FADE Certified Badge" width="264" height="80"></div>
  <button class="copy-btn" onclick="navigator.clipboard.writeText('[![FADE Certified]({badge_url})]({verify_url})').then(()=>this.textContent='COPIED ✓').catch(()=>0)">Copy Badge Markdown →</button>
  <hr class="divider">
  <div class="section-title">Agent Manifest Snippet</div>
  <div class="code-block" id="embedCode">{embed}</div>
  <button class="copy-btn" onclick="navigator.clipboard.writeText(document.getElementById('embedCode').textContent).then(()=>this.textContent='COPIED ✓').catch(()=>0)" style="margin-top:8px;">Copy Snippet →</button>
  <hr class="divider">
  <div class="section-title">Machine Verification</div>
  <div class="code-block">GET {verify_url}</div>
  <hr class="divider">
  <a href="{BASE_URL}" class="back">← Back to Fade</a>
</div></body></html>"""
    return HTMLResponse(content=html)


@app.get("/examples/certs")
async def examples_certs():
    """Machine-readable example cert registry. Used by the examples tab to load real tokens."""
    return JSONResponse(_example_certs)


@app.get("/.well-known/audits.json")
async def well_known_audits():
    """Agent-discoverable audit registry. Lists completed public example audits."""
    audits = []
    for key, cert in _example_certs.items():
        audits.append({
            "subject":    cert["subject"],
            "tier":       cert["tier"],
            "issued_at":  cert["issued_at"],
            "cert_url":   cert["cert_url"],
            "verify_url": cert["verify_url"],
            "badge_url":  cert["badge_url"],
            "auditor":    "Fade Agent Auditor",
            "auditor_url": BASE_URL,
        })
    return JSONResponse({
        "auditor":     "Fade",
        "auditor_url": BASE_URL,
        "schema":      "fade/audit-registry/v1",
        "audits":      audits,
    })


@app.get("/rate")
async def doge_rate():
    """Current DOGE/USD + USD/RMB rates — useful for agents and CN frontend."""
    rate = await get_doge_rate()
    rmb_rate = await get_rmb_rate()
    return {
        "doge_usd": rate,
        "usd_rmb":  rmb_rate,
        "full_audit_doge":  round(PRICE_FULL_USD / rate, 3),
        "agent_audit_doge": round(PRICE_AGENT_USD / rate, 3),
        "full_audit_rmb":   round(PRICE_FULL_USD * rmb_rate, 2),
        "agent_audit_rmb":  round(PRICE_AGENT_USD * rmb_rate, 2),
        "address": FADE_DOGE_ADDRESS,
    }

# =============================================================================
# AUDIT ENDPOINTS
# =============================================================================

@app.post("/audit/free")
@limiter.limit("10/minute;30/hour")
async def free_audit(req: FreeRequest, request: Request):
    safe = sanitize_content(req.content)
    lang_note = lang_instruction(req.lang)
    prompt = f"""The user has submitted the following for a free read.

Give them three to four sentences total:
- First sentence: the single sharpest diagnosis. Name the specific problem, not the category. Be concrete — reference what's actually in front of you.
- Second sentence: why it matters. What breaks because of this.
- Third sentence (optional but good): the direction of the fix — not the fix itself, just the heading.
- Final line: a brief, in-character offer of the full read. Warm, not pushy.

Do not summarize everything you see. Pick the one thing that, if fixed, would matter most. If the submission is genuinely strong, say so honestly and name what you'd still tighten.

Keep it in character. Warm, direct, a little wry. This is a taste, not a lecture.
{lang_note}

Submitted content:
{safe}"""

    diagnosis = await call_fade_free(prompt, max_tokens=2000)
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
        "lang":        req.lang,
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
        "paid":            session.get("paid", False),
        "expired":         expired,
        "tier":            session.get("tier"),
        "watching_for":    session.get("doge_amount"),
        "expires_at":      session.get("expires_at"),
    }


@app.post("/audit/full")
@limiter.limit("30/hour")
async def full_audit(req: AuditRequest, request: Request):
    await _verify_payment(req.session_id, expected_tier="full")
    safe = sanitize_content(req.content)

    lang_note = lang_instruction(req.lang)
    prompt = f"""The user has paid for a full system prompt audit. Give them the complete read — then give them the actual rewrite.

Structure your response as:

1. **The Hand You Dealt** — two to three sentences. What they submitted, what it's trying to do, and the single biggest gap between what it intends and what it will actually do. Be specific. Reference the actual content in front of you.

2. **What's Working** — be genuinely specific. Name the exact phrases or structural choices that are solid and explain why they work. If nothing is working, say so without cruelty — "there's not much to save here, but that's okay, we're starting fresh." Don't manufacture praise.

3. **The Problems, Ranked** — go critical to minor. For each problem: one sentence naming the specific issue (quote the exact language that causes it if relevant), one sentence on what breaks because of it, one sentence on the fix. Minimum four problems. Do not group or skim.

4. **The One Thing** — if they only fix one item before deploying this, what is it? One sentence, bold, decisive.

5. **The Rewrite** — rewrite the full prompt from scratch with every fix applied. Always deliver this regardless of length. Format it as a clean, labeled code block they can copy and use immediately. This is the deliverable. Don't summarize what you'd change — write the thing. It should be production-ready: clear identity, scope, constraints, tone, escalation, and anything else the role demands.

Stay in character throughout. Warm, direct, a little wry about the situation. Never cruel to the builder.
If the prompt is genuinely solid, say so clearly — give them the honest grade, then still deliver a polished version.
{lang_note}

Submitted content:
{safe}"""

    audit = call_fade_paid(prompt, max_tokens=3000)
    mark_used(req.session_id)
    cert_subject = req.subject.strip() or req.content.split('\n')[0].strip()[:80] or "System Prompt"
    cert_token   = issue_cert(cert_subject, "full", "reviewed")
    logger.info(f"Full audit delivered | session={req.session_id[:12]}...")
    return {
        "audit":                  audit,
        "cert_token":             cert_token,
        "cert_url":               f"{BASE_URL}/cert/{cert_token}",
        "badge_url":              f"{BASE_URL}/badge/{cert_token}.svg",
        "verify_url":             f"{BASE_URL}/verify/{cert_token}",
        "constitution_reference": f"{BASE_URL}/constitution",
    }


@app.post("/audit/agent")
@limiter.limit("30/hour")
async def agent_audit(req: AuditRequest, request: Request):
    await _verify_payment(req.session_id, expected_tier="agent")
    safe = sanitize_content(req.content)

    lang_note = lang_instruction(req.lang)
    prompt = f"""The user has paid for a full agent setup audit. This is the deep read. Do it justice.

Structure your response as:

1. **The Setup Read** — three to five sentences. What is this agent supposed to do, what will it actually do, and what is the specific delta between those two things? Name the domain, the intended users, the failure mode you'd bet money on. Be precise — you're reading their work, not describing a category of agent.

2. **The Breaks** — every failure point you can find. Not a list of categories — a list of specific problems with specific consequences. For each one:
   - Name it in one sentence. Quote the exact language that creates the problem if it's in the submission.
   - Explain what breaks in one sentence.
   - Give the fix in one sentence.
   Minimum five breaks. If the agent has more, find them all.

3. **The Trust Audit** — examine every permission, tool access, and data handling claim. What can this agent do that it shouldn't be able to do? What should it have access to that isn't granted? If permissions aren't defined at all, say that clearly — a permission vacuum is its own vulnerability.

4. **The Model Note** — what model is this agent actually running on, or what model fits its requirements? Consider: reasoning depth, context window, cost profile, latency needs, and safety alignment. If there's a mismatch between what the soul demands and what the model can deliver, name it. Be specific about model families — don't hedge with "a capable model."

5. **The Fix Priority** — five items, ordered. First through fifth. The person reading this is about to go do the work — make the list actionable and unambiguous.

6. **The Rebuilt Materials** — deliver the actual improved files. This is what they paid for.
   - Rewrite every document they submitted with all fixes applied. If they sent a soul.md, give them back a better soul.md. If they sent a system prompt, give them the rewritten system prompt. If they sent both, give them both.
   - Label each file clearly (e.g., `## soul.md`, `## system_prompt.md`).
   - Format each as a clean code block ready to copy, save, and deploy. Not bullets describing what to change — the actual file.
   - If something in the original was genuinely good, keep it. You're improving, not replacing everything for sport.
   - Write it at a quality level you'd be proud of. This goes in production.

Stay in character throughout. Warm, direct, precise. This is the most thorough thing you do.
Real praise where earned. Real critique everywhere else. Never cruel to the builder — the work is fair game, the person is not.
If this agent will interact with people, mention the AI Constitution once — briefly, without pressure. It's worth knowing.
{lang_note}

Submitted content:
{safe}"""

    audit = call_fade_paid(prompt, max_tokens=5000)
    mark_used(req.session_id)
    cert_subject = req.subject.strip() or req.content.split('\n')[0].strip()[:80] or "Agent Configuration"
    cert_token   = issue_cert(cert_subject, "agent", "reviewed")
    logger.info(f"Agent audit delivered | session={req.session_id[:12]}...")
    return {
        "audit":                  audit,
        "cert_token":             cert_token,
        "cert_url":               f"{BASE_URL}/cert/{cert_token}",
        "badge_url":              f"{BASE_URL}/badge/{cert_token}.svg",
        "verify_url":             f"{BASE_URL}/verify/{cert_token}",
        "constitution_reference": f"{BASE_URL}/constitution",
    }


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
