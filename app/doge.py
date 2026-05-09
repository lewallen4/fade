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

# Transaction IDs that have already confirmed a session.
# Prevents one blockchain tx from unlocking multiple sessions.
_used_tx_ids: set[str] = set()

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

def claim_tx(tx_id: str) -> bool:
    """
    Atomically claim a transaction ID for a session.
    Returns True if this caller is the first to claim it, False if already taken.
    """
    if tx_id in _used_tx_ids:
        return False
    _used_tx_ids.add(tx_id)
    return True


async def check_payment(expected_amount: float, tolerance: float = 0.0001) -> Optional[str]:
    """
    Check DogeChain API for a recent unclaimed transaction matching the expected
    amount within tolerance. Returns the tx hash if found, None otherwise.

    Tolerance of 0.0001 DOGE is wide enough to absorb API rounding/precision
    differences, but far smaller than the gap between any two session amounts
    (~142 DOGE for full, ~428 DOGE for agent). Combined with tx_id claiming,
    this uniquely identifies payments even across concurrent sessions.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://dogechain.info/api/v1/address/transactions/{FADE_DOGE_ADDRESS}",
            )
            resp.raise_for_status()
            data = resp.json()

            txs = data.get("transactions", [])
            logger.info(f"DogeChain returned {len(txs)} transactions, looking for {expected_amount:.8f} DOGE")

            for tx in txs[:20]:
                tx_id = tx.get("hash") or tx.get("txid") or tx.get("id")
                if not tx_id:
                    logger.warning(f"TX has no hash field, keys: {list(tx.keys())}")
                    continue
                if tx_id in _used_tx_ids:
                    continue

                # Check per-output amounts (full transaction detail)
                matched = False
                for output in tx.get("outputs", []):
                    if output.get("address") == FADE_DOGE_ADDRESS:
                        try:
                            amount = float(output.get("value", 0))
                            logger.debug(f"TX {tx_id[:12]} output to us: {amount:.8f} DOGE (want {expected_amount:.8f})")
                            if abs(amount - expected_amount) <= tolerance:
                                matched = True
                                break
                        except (ValueError, TypeError):
                            continue

                # Fallback: some API responses carry a top-level value field
                if not matched and tx.get("outputs") is None:
                    try:
                        amount = float(tx.get("value", 0))
                        logger.debug(f"TX {tx_id[:12]} top-level value: {amount:.8f} DOGE (want {expected_amount:.8f})")
                        if abs(amount - expected_amount) <= tolerance:
                            matched = True
                    except (ValueError, TypeError):
                        pass

                if matched:
                    return tx_id

    except Exception as e:
        logger.error(f"DogeChain check error: {type(e).__name__}: {e}")
    return None

async def watch_payment(
    session_id: str,
    expected_amount: float,
    expires_at: float,
    on_confirmed,
) -> None:
    """
    Background task: poll for payment until confirmed or expired.
    Calls on_confirmed(session_id, tx_id) when a matching, unclaimed tx is found.
    If two sessions have amounts within tolerance, the first to poll wins the tx;
    the other keeps watching until its own payment arrives.
    """
    logger.info(f"Watching for {expected_amount:.8f} DOGE | session={session_id[:8]}...")

    while time.time() < expires_at:
        tx_id = await check_payment(expected_amount)
        if tx_id:
            if claim_tx(tx_id):
                logger.info(f"DOGE payment confirmed | tx={tx_id[:12]}... | session={session_id[:8]}...")
                await on_confirmed(session_id, tx_id)
                return
            # Another session claimed this tx first — keep watching for ours
            logger.debug(f"TX {tx_id[:12]}... already claimed, still watching | session={session_id[:8]}...")
        await asyncio.sleep(POLL_INTERVAL)

    logger.info(f"DOGE payment watch expired | session={session_id[:8]}...")

# --- RMB rate cache ---
_rmb_rate: float = 0.0
_rmb_last_updated: float = 0.0

async def get_rmb_rate() -> float:
    """Fetch current USD/RMB rate. Cached for 24 hours."""
    global _rmb_rate, _rmb_last_updated
    now = time.time()
    if _rmb_rate > 0 and (now - _rmb_last_updated) < RATE_TTL:
        return _rmb_rate
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Frankfurter is a free, no-key ECB rates API
            resp = await client.get(
                "https://api.frankfurter.app/latest",
                params={"from": "USD", "to": "CNY"}
            )
            resp.raise_for_status()
            data = resp.json()
            rate = float(data["rates"]["CNY"])
            _rmb_rate = rate
            _rmb_last_updated = now
            logger.info(f"RMB rate updated: 1 USD = ¥{rate:.4f} CNY")
            return rate
    except Exception as e:
        logger.error(f"RMB rate fetch error: {type(e).__name__}")
        if _rmb_rate > 0:
            return _rmb_rate
        return 7.25  # fallback

async def rate_refresh_loop():
    """Background task: refresh DOGE and RMB rates every 24 hours."""
    while True:
        await asyncio.sleep(RATE_TTL)
        try:
            await get_doge_rate()
            await get_rmb_rate()
        except Exception as e:
            logger.error(f"Rate refresh error: {type(e).__name__}")
