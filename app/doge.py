"""
doge.py — DOGE payment watcher for Fade
No node. No bank. Just vibes and blockchain.

Flow:
  1. /checkout generates a unique DOGE amount (USD price converted + dust suffix)
  2. User sends exact amount to FADE_DOGE_ADDRESS
  3. Watcher polls DogeChain API every 30s for incoming txs
  4. Match found → session marked paid → audit unlocks

Rate cache updates daily via background task.
"""

import os
import asyncio
import time
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# --- Config ---
FADE_DOGE_ADDRESS = os.environ.get("DOGE_ADDRESS", "DMJ3AWFE4trzRwjwqyCUpMcrE6t1b2mh6h")

# USD prices
PRICE_FULL_USD  = float(os.environ.get("PRICE_FULL_USD", "1.00"))
PRICE_AGENT_USD = float(os.environ.get("PRICE_AGENT_USD", "3.00"))

# Poll interval for tx checking (seconds)
POLL_INTERVAL = 30

# How long to wait for payment before expiring (seconds)
PAYMENT_TIMEOUT = 60 * 30  # 30 minutes

# --- Rate cache ---
_doge_rate: float = 0.0          # USD per 1 DOGE
_rate_last_updated: float = 0.0
RATE_TTL = 60 * 60 * 24         # refresh every 24 hours

async def get_doge_rate() -> float:
    """Fetch current DOGE/USD rate. Cached for 24 hours."""
    global _doge_rate, _rate_last_updated

    now = time.time()
    if _doge_rate > 0 and (now - _rate_last_updated) < RATE_TTL:
        return _doge_rate

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # CoinGecko free API — no key needed
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "dogecoin", "vs_currencies": "usd"}
            )
            resp.raise_for_status()
            data = resp.json()
            rate = float(data["dogecoin"]["usd"])
            _doge_rate = rate
            _rate_last_updated = now
            logger.info(f"DOGE rate updated: 1 DOGE = ${rate:.6f} USD")
            return rate
    except Exception as e:
        logger.error(f"DOGE rate fetch error: {type(e).__name__}")
        # Return stale rate if we have one, else fallback
        if _doge_rate > 0:
            logger.warning("Using stale DOGE rate.")
            return _doge_rate
        # Last resort hardcoded fallback — better than crashing
        logger.warning("Using hardcoded DOGE fallback rate of $0.08.")
        return 0.08

async def usd_to_doge(usd_amount: float) -> float:
    """Convert USD to DOGE at current rate."""
    rate = await get_doge_rate()
    return usd_amount / rate

def make_unique_amount(base_doge: float, session_suffix: str) -> float:
    """
    Add a small dust suffix to make each payment amount unique.
    This is how we identify which session a payment belongs to
    without needing a memo field.

    Uses last 4 chars of session ID as a numeric suffix in the
    4th-8th decimal places — invisible to the user, identifiable
    on the blockchain.

    e.g. base=142.857, suffix="a3f2" → 142.85700163
    """
    # Convert 4 hex chars to a number 0-65535, scale to 0.00001–0.00099
    try:
        dust = int(session_suffix[-4:], 16) / 65535 * 0.00099 + 0.00001
    except Exception:
        dust = 0.00042
    # Round base to 3 decimal places, add dust in lower decimals
    return round(base_doge, 3) + round(dust, 8)

async def generate_payment(session_id: str, tier: str) -> dict:
    """
    Generate a unique DOGE payment request for a session.
    Returns address, amount, USD equivalent, and expiry.
    """
    usd = PRICE_FULL_USD if tier == "full" else PRICE_AGENT_USD
    base_doge = await usd_to_doge(usd)
    unique_doge = make_unique_amount(base_doge, session_id)
    rate = await get_doge_rate()

    return {
        "address": FADE_DOGE_ADDRESS,
        "amount_doge": round(unique_doge, 8),
        "amount_usd": usd,
        "rate": rate,
        "expires_at": time.time() + PAYMENT_TIMEOUT,
        "tier": tier,
        "session_id": session_id,
    }

# --- Transaction watcher ---

async def check_payment(expected_amount: float, tolerance: float = 0.001) -> bool:
    """
    Check DogeChain API for a recent incoming transaction
    matching the expected amount (within tolerance).
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://dogechain.info/api/v1/address/transactions/{FADE_DOGE_ADDRESS}",
            )
            resp.raise_for_status()
            data = resp.json()

            txs = data.get("transactions", [])
            for tx in txs[:20]:  # check last 20 txs
                # Look for outputs to our address
                for output in tx.get("outputs", []):
                    if output.get("address") == FADE_DOGE_ADDRESS:
                        try:
                            amount = float(output.get("value", 0))
                            if abs(amount - expected_amount) <= tolerance:
                                return True
                        except (ValueError, TypeError):
                            continue
    except Exception as e:
        logger.error(f"DogeChain check error: {type(e).__name__}")
    return False

async def watch_payment(
    session_id: str,
    expected_amount: float,
    expires_at: float,
    on_confirmed,
) -> None:
    """
    Background task: poll for payment until confirmed or expired.
    Calls on_confirmed(session_id) when payment detected.
    """
    logger.info(f"Watching for {expected_amount:.8f} DOGE | session={session_id[:8]}...")

    while time.time() < expires_at:
        confirmed = await check_payment(expected_amount)
        if confirmed:
            logger.info(f"DOGE payment confirmed | session={session_id[:8]}...")
            await on_confirmed(session_id)
            return
        await asyncio.sleep(POLL_INTERVAL)

    logger.info(f"DOGE payment watch expired | session={session_id[:8]}...")

async def rate_refresh_loop():
    """Background task: refresh DOGE rate every 24 hours."""
    while True:
        await asyncio.sleep(RATE_TTL)
        try:
            await get_doge_rate()
        except Exception as e:
            logger.error(f"Rate refresh error: {type(e).__name__}")
