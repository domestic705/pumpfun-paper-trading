"""
market_data.py - Pump.fun Token Discovery & Price Data

Uses multiple data sources:
1. PumpPortal WebSocket (real-time pump.fun tokens on bonding curve)
2. DexScreener API (graduated tokens)

Finds tokens by:
- Real-time WebSocket stream from pump.fun via PumpPortal
- DexScreener for price data of graduated tokens
"""
from __future__ import annotations

import json
import os
import time
import requests
import pandas as pd
from typing import Dict, List, Optional, Any

# Import pump.fun stream (auto-starts WebSocket connection)
try:
    from pumpfun_stream import (
        get_tokens as get_pumpfun_tokens,
        get_stats as get_pumpfun_stats,
        is_connected as is_pumpfun_connected,
        start_stream as start_pumpfun_stream,
    )
    PUMPFUN_AVAILABLE = True
except Exception:
    # Gracefully handle any import error (ImportError, SSL issues, etc.)
    PUMPFUN_AVAILABLE = False
    def get_pumpfun_tokens(): return []
    def get_pumpfun_stats(): return {}
    def is_pumpfun_connected(): return False
    def start_pumpfun_stream(): return False

# =====================================================================
# CONFIG
# =====================================================================
DEXSCREENER_API = "https://api.dexscreener.com"

# How many tokens to track
MAX_TRACKED_TOKENS = 50
# Price cache TTL (seconds) - keep short for active trades
PRICE_CACHE_TTL = 2.0
# Discovery cache TTL (seconds)
DISCOVERY_CACHE_TTL = 15.0
# How long to keep a token (30 minutes)
TOKEN_EXPIRE_TIME = 30 * 60

# Persistence file
TOKENS_FILE = os.path.join(os.path.dirname(__file__), "tracked_tokens.json")

# =====================================================================
# INTERNAL STATE
# =====================================================================
_LAST_ERROR: Optional[str] = None
_RATE_LIMIT = {"until": 0.0, "backoff": 2.0}

# Token pool: address -> token info
_TOKEN_POOL: Dict[str, Dict[str, Any]] = {}

# Price cache: address -> {ts, price_sol, price_usd, pair_address}
_PRICE_CACHE: Dict[str, Dict[str, Any]] = {}

# Price history for charts
_PRICE_SAMPLES: Dict[str, List[tuple]] = {}

# Discovery cache
_DISCOVERY_CACHE = {"ts": 0.0, "tokens": []}

# Stats
_STATS = {"total_tracked": 0, "new_this_session": 0, "last_new_token": None}

# SOL/USD price cache
_SOL_USD_CACHE = {"ts": 0.0, "price": 0.0}


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def get_last_error() -> Optional[str]:
    return _LAST_ERROR


def _set_error(msg: Optional[str]):
    global _LAST_ERROR
    _LAST_ERROR = msg


def _now_ts() -> float:
    return time.time()


def get_stats() -> Dict[str, Any]:
    return dict(_STATS)


def reset_runtime_state():
    """Clear all in-memory state."""
    global _LAST_ERROR
    _LAST_ERROR = None
    _TOKEN_POOL.clear()
    _PRICE_CACHE.clear()
    _PRICE_SAMPLES.clear()
    _DISCOVERY_CACHE["ts"] = 0.0
    _DISCOVERY_CACHE["tokens"] = []
    _RATE_LIMIT["until"] = 0.0
    _RATE_LIMIT["backoff"] = 2.0
    _STATS["total_tracked"] = 0
    _STATS["new_this_session"] = 0
    _STATS["last_new_token"] = None
    
    if os.path.exists(TOKENS_FILE):
        try:
            os.remove(TOKENS_FILE)
        except Exception:
            pass


def _load_tokens_from_disk():
    if os.path.exists(TOKENS_FILE):
        try:
            with open(TOKENS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _TOKEN_POOL.update(data)
                    _STATS["total_tracked"] = len(_TOKEN_POOL)
        except Exception:
            pass


def _save_tokens_to_disk():
    try:
        with open(TOKENS_FILE, "w") as f:
            json.dump(_TOKEN_POOL, f)
    except Exception:
        pass


def _is_rate_limited() -> bool:
    return _now_ts() < float(_RATE_LIMIT["until"])


def _handle_rate_limit(retry_after: Optional[str] = None):
    wait_s = None
    if retry_after:
        try:
            wait_s = float(retry_after)
        except Exception:
            pass
    if wait_s is None:
        wait_s = float(_RATE_LIMIT["backoff"])
        _RATE_LIMIT["backoff"] = min(float(_RATE_LIMIT["backoff"]) * 2.0, 60.0)
    _RATE_LIMIT["until"] = _now_ts() + wait_s


def _reset_rate_limit():
    _RATE_LIMIT["backoff"] = 2.0
    _RATE_LIMIT["until"] = 0.0


# =====================================================================
# DEXSCREENER API
# =====================================================================
def get_sol_usd_price() -> float:
    """Get current SOL/USD price from DexScreener."""
    now = _now_ts()
    
    # Return cached if fresh (30 second TTL)
    if _SOL_USD_CACHE["price"] > 0 and (now - _SOL_USD_CACHE["ts"]) < 30:
        return _SOL_USD_CACHE["price"]
    
    try:
        # Use wrapped SOL address
        sol_mint = "So11111111111111111111111111111111111111112"
        url = f"{DEXSCREENER_API}/tokens/v1/solana/{sol_mint}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                # Get price from highest liquidity pool
                best = max(data, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
                price = float(best.get("priceUsd") or 0)
                if price > 0:
                    _SOL_USD_CACHE["ts"] = now
                    _SOL_USD_CACHE["price"] = price
                    return price
    except Exception:
        pass
    
    # Fallback to cached or default
    if _SOL_USD_CACHE["price"] > 0:
        return _SOL_USD_CACHE["price"]
    return 100.0  # Default fallback


def _fetch_dexscreener_search(query: str) -> List[Dict]:
    """Search DexScreener for tokens matching query."""
    if _is_rate_limited():
        return []
    
    url = f"{DEXSCREENER_API}/latest/dex/search?q={query}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 429:
            _handle_rate_limit(r.headers.get("retry-after"))
            return []
        if r.status_code != 200:
            return []
        
        _reset_rate_limit()
        data = r.json()
        pairs = data.get("pairs", [])
        return pairs if isinstance(pairs, list) else []
    except Exception as e:
        _set_error(f"DexScreener search failed: {e!r}")
        return []


def _fetch_token_pairs(address: str) -> List[Dict]:
    """Fetch all pairs for a specific token."""
    if _is_rate_limited():
        return []
    
    url = f"{DEXSCREENER_API}/tokens/v1/solana/{address}"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 429:
            _handle_rate_limit(r.headers.get("retry-after"))
            return []
        if r.status_code != 200:
            return []
        
        _reset_rate_limit()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _fetch_token_profiles() -> List[str]:
    """Fetch recently active token addresses from DexScreener profiles."""
    if _is_rate_limited():
        return []
    
    url = f"{DEXSCREENER_API}/token-profiles/latest/v1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []
        
        # Get Solana token addresses
        addresses = []
        for p in data:
            if str(p.get("chainId") or "").lower() == "solana":
                addr = p.get("tokenAddress")
                if addr:
                    addresses.append(str(addr))
        return addresses
    except Exception:
        return []


def _fetch_boosted_tokens() -> List[str]:
    """Fetch boosted/trending token addresses from DexScreener."""
    if _is_rate_limited():
        return []
    
    url = f"{DEXSCREENER_API}/token-boosts/top/v1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []
        
        # Get Solana pump.fun token addresses
        addresses = []
        for t in data:
            if str(t.get("chainId") or "").lower() == "solana":
                addr = t.get("tokenAddress", "")
                if addr:
                    addresses.append(addr)
        return addresses
    except Exception:
        return []


def _discover_tokens() -> List[Dict[str, Any]]:
    """
    Discover meme tokens using DexScreener.
    Returns list of token info dicts.
    """
    now = _now_ts()
    
    # Return cached if fresh
    if _DISCOVERY_CACHE["tokens"] and (now - _DISCOVERY_CACHE["ts"]) < DISCOVERY_CACHE_TTL:
        return _DISCOVERY_CACHE["tokens"]
    
    all_pairs = []
    
    # 1. Fetch boosted/trending tokens first (these are the hot ones!)
    boosted_addresses = _fetch_boosted_tokens()
    if boosted_addresses and not _is_rate_limited():
        # Batch fetch up to 30
        batch = boosted_addresses[:30]
        joined = ",".join(batch)
        url = f"{DEXSCREENER_API}/tokens/v1/solana/{joined}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                pairs = r.json()
                if isinstance(pairs, list):
                    all_pairs.extend(pairs)
                    print(f"[Discovery] Fetched {len(pairs)} boosted token pairs")
        except Exception:
            pass
        time.sleep(0.2)
    
    # 2. Search multiple terms to find Solana meme coins
    search_terms = [
        "pump solana", "pepe solana", "dog solana", "cat solana", 
        "meme solana", "trump", "ai solana", "doge"
    ]
    for term in search_terms:
        pairs = _fetch_dexscreener_search(term)
        all_pairs.extend(pairs)
        if _is_rate_limited():
            break
        time.sleep(0.15)  # Small delay to avoid rate limits
    
    # 3. Also fetch from token profiles
    profile_addresses = _fetch_token_profiles()
    if profile_addresses and not _is_rate_limited():
        # Batch fetch pairs for profile tokens
        for i in range(0, min(len(profile_addresses), 30), 10):
            batch = profile_addresses[i:i+10]
            for addr in batch:
                pairs = _fetch_token_pairs(addr)
                all_pairs.extend(pairs)
                if _is_rate_limited():
                    break
            time.sleep(0.15)
    
    # Filter and dedupe
    seen_addresses = set()
    tokens = []
    
    for pair in all_pairs:
        if not isinstance(pair, dict):
            continue
        
        # Must be Solana
        chain_id = str(pair.get("chainId") or "").lower()
        if chain_id != "solana":
            continue
        
        # Get base token info
        base_token = pair.get("baseToken") or {}
        address = str(base_token.get("address") or "")
        symbol = str(base_token.get("symbol") or "???")
        name = str(base_token.get("name") or "Unknown")
        
        if not address or address in seen_addresses:
            continue
        
        # Skip unknown/invalid symbols
        if symbol.lower() in ("unknown", "???", "", "null", "none"):
            continue
        seen_addresses.add(address)
        
        # Get metrics
        price_usd = 0.0
        try:
            price_usd = float(pair.get("priceUsd") or 0)
        except Exception:
            pass
        
        price_native = 0.0
        try:
            price_native = float(pair.get("priceNative") or 0)
        except Exception:
            pass
        
        market_cap = 0.0
        try:
            market_cap = float(pair.get("marketCap") or pair.get("fdv") or 0)
        except Exception:
            pass
        
        liquidity_usd = 0.0
        try:
            liquidity_usd = float((pair.get("liquidity") or {}).get("usd") or 0)
        except Exception:
            pass
        
        # Get pair creation time
        pair_created_at = pair.get("pairCreatedAt")
        age_hours = 999
        if pair_created_at:
            try:
                age_ms = now * 1000 - float(pair_created_at)
                age_hours = age_ms / (1000 * 60 * 60)
            except Exception:
                pass
        
        # Volume
        volume_h1 = 0.0
        try:
            volume_h1 = float((pair.get("volume") or {}).get("h1") or 0)
        except Exception:
            pass
        
        # Price change
        price_change_h1 = 0.0
        try:
            price_change_h1 = float((pair.get("priceChange") or {}).get("h1") or 0)
        except Exception:
            pass
        
        # Transaction count (buys + sells)
        txns_h1 = 0
        txns_m5 = 0
        volume_m5 = 0.0
        try:
            txns = pair.get("txns") or {}
            h1_txns = txns.get("h1") or {}
            m5_txns = txns.get("m5") or {}
            txns_h1 = int(h1_txns.get("buys", 0)) + int(h1_txns.get("sells", 0))
            txns_m5 = int(m5_txns.get("buys", 0)) + int(m5_txns.get("sells", 0))
            
            # Get 5-minute volume
            vol = pair.get("volume") or {}
            volume_m5 = float(vol.get("m5") or 0)
        except Exception:
            pass
        
        dex_id = str(pair.get("dexId") or "")
        pair_address = str(pair.get("pairAddress") or "")
        
        tokens.append({
            "address": address,
            "symbol": symbol,
            "name": name,
            "price_usd": price_usd,
            "price_sol": price_native,
            "market_cap": market_cap,
            "liquidity_usd": liquidity_usd,
            "volume_h1": volume_h1,
            "volume_m5": volume_m5,
            "price_change_h1": price_change_h1,
            "txns_h1": txns_h1,
            "txns_m5": txns_m5,
            "age_hours": age_hours,
            "dex_id": dex_id,
            "pair_address": pair_address,
        })
    
    # Sort by volume (most active first)
    tokens.sort(key=lambda x: x.get("volume_h1", 0), reverse=True)
    
    # Limit
    tokens = tokens[:MAX_TRACKED_TOKENS]
    
    print(f"[DISCOVERY] Found {len(tokens)} Solana tokens from {len(all_pairs)} pairs")
    if tokens:
        top3 = tokens[:3]
        for t in top3:
            print(f"  - {t['symbol']}: MC ${t['market_cap']:,.0f} | Vol ${t['volume_h1']:,.0f}/h")
    
    _DISCOVERY_CACHE["ts"] = now
    _DISCOVERY_CACHE["tokens"] = tokens
    
    return tokens


# =====================================================================
# TOKEN TRACKING
# =====================================================================
def _update_token_pool():
    """Update token pool from discovery."""
    # Load from disk on first call
    if not _TOKEN_POOL:
        _load_tokens_from_disk()
    
    now = _now_ts()
    discovered = _discover_tokens()
    
    new_count = 0
    for token in discovered:
        addr = token.get("address")
        if not addr:
            continue
        
        if addr not in _TOKEN_POOL:
            _TOKEN_POOL[addr] = {
                **token,
                "discovered_at": now,
            }
            new_count += 1
            _STATS["new_this_session"] += 1
            _STATS["last_new_token"] = _TOKEN_POOL[addr]
            print(f"NEW: {token.get('symbol')} | ${token.get('market_cap', 0):,.0f} MC | ${token.get('volume_h1', 0):,.0f} vol/h")
        else:
            # Update existing with fresh data
            _TOKEN_POOL[addr].update({
                "price_usd": token.get("price_usd", 0),
                "price_sol": token.get("price_sol", 0),
                "market_cap": token.get("market_cap", 0),
                "liquidity_usd": token.get("liquidity_usd", 0),
                "volume_h1": token.get("volume_h1", 0),
                "price_change_h1": token.get("price_change_h1", 0),
            })
    
    # Remove old tokens
    expired = [
        addr for addr, t in _TOKEN_POOL.items()
        if (now - float(t.get("discovered_at", 0))) > TOKEN_EXPIRE_TIME
    ]
    for addr in expired:
        del _TOKEN_POOL[addr]
        _PRICE_CACHE.pop(addr, None)
        _PRICE_SAMPLES.pop(addr, None)
    
    _STATS["total_tracked"] = len(_TOKEN_POOL)
    _save_tokens_to_disk()
    _set_error(None)


def add_token_manually(address: str, symbol: str = "MANUAL") -> bool:
    """Manually add a token address to track."""
    if not address or len(address) < 32:
        return False
    
    address = str(address).strip()
    if address in _TOKEN_POOL:
        return False
    
    # Try to fetch token data
    pairs = _fetch_token_pairs(address)
    
    if pairs:
        # Use best pair (highest liquidity)
        best = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
        base = best.get("baseToken") or {}
        
        _TOKEN_POOL[address] = {
            "address": address,
            "symbol": str(base.get("symbol") or symbol),
            "name": str(base.get("name") or symbol),
            "price_usd": float(best.get("priceUsd") or 0),
            "price_sol": float(best.get("priceNative") or 0),
            "market_cap": float(best.get("marketCap") or best.get("fdv") or 0),
            "liquidity_usd": float((best.get("liquidity") or {}).get("usd") or 0),
            "volume_h1": float((best.get("volume") or {}).get("h1") or 0),
            "price_change_h1": float((best.get("priceChange") or {}).get("h1") or 0),
            "dex_id": str(best.get("dexId") or ""),
            "pair_address": str(best.get("pairAddress") or ""),
            "discovered_at": _now_ts(),
            "manually_added": True,
        }
    else:
        # Add with minimal info
        _TOKEN_POOL[address] = {
            "address": address,
            "symbol": symbol,
            "name": symbol,
            "price_usd": 0,
            "price_sol": 0,
            "market_cap": 0,
            "liquidity_usd": 0,
            "volume_h1": 0,
            "price_change_h1": 0,
            "dex_id": "",
            "pair_address": "",
            "discovered_at": _now_ts(),
            "manually_added": True,
        }
    
    _STATS["total_tracked"] = len(_TOKEN_POOL)
    _save_tokens_to_disk()
    return True


def get_token_price(address: str) -> float:
    """Get token price in SOL. Fetches FRESH data from DexScreener."""
    now = _now_ts()
    cached = _PRICE_CACHE.get(address)
    last_known_price = float(cached.get("price_sol", 0.0)) if cached else 0.0
    cache_age = now - float(cached.get("ts", 0)) if cached else 9999
    
    # Return cached if very fresh (3 seconds - reduced from 5)
    if cached and cache_age < 3:
        price = float(cached.get("price_sol", 0.0))
        if price > 0:
            return price
    
    price_sol = 0.0
    price_usd = 0.0
    
    # ALWAYS try to fetch fresh price from DexScreener first (not from pool!)
    if not _is_rate_limited():
        pairs = _fetch_token_pairs(address)
        if pairs:
            # Filter to Solana pairs with liquidity
            valid_pairs = [
                p for p in pairs 
                if p.get("chainId") == "solana" 
                and float((p.get("liquidity") or {}).get("usd") or 0) > 100
            ]
            
            if valid_pairs:
                # Get best pair by liquidity
                best = max(valid_pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
                try:
                    price_sol = float(best.get("priceNative") or 0)
                except:
                    price_sol = 0.0
                try:
                    price_usd = float(best.get("priceUsd") or 0)
                except:
                    price_usd = 0.0
    
    # Fallback 1: try from pool (discovery data - may be slightly stale)
    if price_sol <= 0:
        token = _TOKEN_POOL.get(address)
        if token:
            pool_price = float(token.get("price_sol", 0))
            pool_ts = float(token.get("discovered_at", 0))
            pool_age = now - pool_ts
            # Only use pool price if it's less than 60 seconds old
            if pool_price > 0 and pool_age < 60:
                price_sol = pool_price
                price_usd = float(token.get("price_usd", 0))
    
    # Fallback 2: use cached price (but mark it as stale)
    if price_sol <= 0 and cached:
        price_sol = float(cached.get("price_sol", 0.0))
        price_usd = float(cached.get("price_usd", 0.0))
        # Don't update the cache timestamp - keep it stale
        return price_sol
    
    # Update cache with fresh price
    if price_sol > 0:
        _PRICE_CACHE[address] = {
            "ts": now,
            "price_sol": price_sol,
            "price_usd": price_usd,
        }
        
        # Record sample for charts
        ts = pd.Timestamp.now(tz="UTC")
        samples = _PRICE_SAMPLES.setdefault(address, [])
        samples.append((ts, price_sol))
        cutoff = ts - pd.Timedelta(minutes=30)
        _PRICE_SAMPLES[address] = [(t, p) for t, p in samples if t >= cutoff]
    
    return price_sol


def get_token_price_usd(address: str) -> float:
    """Get token price in USD."""
    _ = get_token_price(address)
    cached = _PRICE_CACHE.get(address, {})
    return float(cached.get("price_usd", 0.0))


def fetch_candidates() -> List[Dict[str, Any]]:
    """Returns list of tracked tokens with their data.
    
    Merges tokens from:
    1. PumpPortal WebSocket (real-time pump.fun bonding curve tokens)
    2. DexScreener (graduated tokens with DEX liquidity)
    """
    _update_token_pool()
    
    now = _now_ts()
    candidates = []
    seen_addresses = set()
    
    # 1. Add pump.fun tokens from WebSocket (priority - these are the hot new ones)
    if PUMPFUN_AVAILABLE:
        pumpfun_tokens = get_pumpfun_tokens()
        for token in pumpfun_tokens:
            addr = token.get("address", "")
            if not addr or addr in seen_addresses:
                continue
            
            symbol = token.get("symbol", "???")
            if symbol.lower() in ("unknown", "???", "", "null", "none"):
                continue
            
            seen_addresses.add(addr)
            age_s = now - float(token.get("created_at", now))
            
            # Convert market cap from SOL to USD (approximate)
            mc_sol = float(token.get("market_cap_sol", 0))
            sol_price = get_sol_usd_price()
            mc_usd = mc_sol * sol_price
            
            candidates.append({
                "address": addr,
                "symbol": symbol,
                "name": token.get("name", "Unknown"),
                "price_sol": float(token.get("price_sol", 0)),
                "price_usd": float(token.get("price_sol", 0)) * sol_price,
                "market_cap": mc_usd,
                "liquidity_usd": mc_usd,  # For bonding curve, liquidity ~ market cap
                "volume_h1": float(token.get("volume_sol", 0)) * sol_price,
                "price_change_h1": 0.0,
                "age_seconds": age_s,
                "age_minutes": age_s / 60.0,
                "dex_id": "pumpfun",
                "bonding_curve": True,
                "bonding_curve_progress": float(token.get("bonding_curve_progress", 0)),
                "tx_count": int(token.get("tx_count", 0)),
                "source": "pumpfun_ws",
            })
    
    # 2. Add DexScreener tokens (graduated tokens)
    for addr, token in _TOKEN_POOL.items():
        if addr in seen_addresses:
            continue
        
        symbol = token.get("symbol", "???")
        if symbol.lower() in ("unknown", "???", "", "null", "none"):
            continue
        
        seen_addresses.add(addr)
        age_s = now - float(token.get("discovered_at", now))
        
        # Check if this is a pump.fun token (address ends with "pump")
        is_pumpfun_token = addr.lower().endswith("pump")
        
        candidates.append({
            "address": addr,
            "symbol": symbol,
            "name": token.get("name", "Unknown"),
            "price_sol": float(token.get("price_sol", 0)),
            "price_usd": float(token.get("price_usd", 0)),
            "market_cap": float(token.get("market_cap", 0)),
            "liquidity_usd": float(token.get("liquidity_usd", 0)),
            "volume_h1": float(token.get("volume_h1", 0)),
            "volume_m5": float(token.get("volume_m5", 0)),
            "price_change_h1": float(token.get("price_change_h1", 0)),
            "txns_h1": int(token.get("txns_h1", 0)),
            "txns_m5": int(token.get("txns_m5", 0)),
            "age_seconds": age_s,
            "age_minutes": age_s / 60.0,
            "dex_id": token.get("dex_id", ""),
            "bonding_curve": False,
            "source": "pumpfun_graduated" if is_pumpfun_token else "dexscreener",
        })
    
    # Sort by: bonding curve tokens first (by market cap), then graduated tokens (by volume)
    def sort_key(x):
        if x.get("bonding_curve"):
            # Bonding curve tokens: sort by market cap, highest first
            return (0, -x.get("market_cap", 0))
        else:
            # Graduated tokens: sort by volume
            return (1, -x.get("volume_h1", 0))
    
    candidates.sort(key=sort_key)
    
    return candidates


def get_price_sample_stats(address: str) -> Dict[str, Any]:
    """Get stats about price samples."""
    samples = _PRICE_SAMPLES.get(address, [])
    if not samples:
        return {"count": 0, "last_ts": None, "last_price": None}
    try:
        last_ts, last_px = samples[-1]
        return {"count": len(samples), "last_ts": last_ts, "last_price": float(last_px)}
    except Exception:
        return {"count": len(samples), "last_ts": None, "last_price": None}


def get_chart_data(address: str) -> pd.DataFrame:
    """Build OHLC candles from price samples."""
    try:
        _ = get_token_price(address)
        
        samples = _PRICE_SAMPLES.get(address, [])
        if len(samples) < 2:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        
        ts_list = []
        price_list = []
        for item in samples:
            try:
                ts, px = item
                ts_list.append(ts)
                price_list.append(px)
            except Exception:
                continue
        
        n = min(len(ts_list), len(price_list))
        if n < 2:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        
        df = pd.DataFrame({
            "timestamp": ts_list[:n],
            "price": price_list[:n]
        }).set_index("timestamp").sort_index()
        
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=15)
        df = df[df.index >= cutoff]
        
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        
        ohlc = df["price"].resample("1min").ohlc().dropna().reset_index()
        return ohlc.rename(columns={"timestamp": "timestamp"})
    except Exception as e:
        print(f"get_chart_data error: {e!r}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])


def get_last_candidate_stats() -> Dict[str, Any]:
    """Return stats for UI."""
    pumpfun_stats = get_pumpfun_stats() if PUMPFUN_AVAILABLE else {}
    pumpfun_count = len(get_pumpfun_tokens()) if PUMPFUN_AVAILABLE else 0
    
    return {
        "raw": len(_TOKEN_POOL) + pumpfun_count,
        "filtered": len(_TOKEN_POOL) + pumpfun_count,
        "dexscreener": len(_TOKEN_POOL),
        "pumpfun": pumpfun_count,
        "pumpfun_connected": is_pumpfun_connected() if PUMPFUN_AVAILABLE else False,
        "pumpfun_stats": pumpfun_stats,
    }


def get_dexscreener_embed_url(token_address: str) -> Optional[str]:
    """Get embeddable DexScreener chart URL."""
    token = _TOKEN_POOL.get(token_address)
    if token and token.get("pair_address"):
        return (
            f"https://dexscreener.com/solana/{token['pair_address']}"
            "?embed=1&theme=dark&trades=0&info=0&chartLeftToolbar=0&chartTheme=dark"
        )
    
    # Try to fetch
    pairs = _fetch_token_pairs(token_address)
    if pairs:
        best = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
        pair_addr = best.get("pairAddress")
        if pair_addr:
            return (
                f"https://dexscreener.com/solana/{pair_addr}"
                "?embed=1&theme=dark&trades=0&info=0&chartLeftToolbar=0&chartTheme=dark"
            )
    return None


def prefetch_token_data(addresses: List[str]):
    """Prefetch price data for owned tokens from DexScreener."""
    if not addresses:
        return
    
    now = _now_ts()
    fetched_addrs = set()
    
    # Try batch fetch first (if not rate limited)
    if not _is_rate_limited():
        batch = addresses[:30]
        joined = ",".join(batch)
        url = f"{DEXSCREENER_API}/tokens/v1/solana/{joined}"
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 429:
                _handle_rate_limit(r.headers.get("retry-after"))
            elif r.status_code == 200:
                _reset_rate_limit()
                pairs = r.json()
                if isinstance(pairs, list):
                    # Group pairs by base token address
                    for pair in pairs:
                        if not isinstance(pair, dict):
                            continue
                        
                        base = pair.get("baseToken") or {}
                        addr = str(base.get("address") or "")
                        if not addr:
                            continue
                        
                        try:
                            price_sol = float(pair.get("priceNative") or 0)
                        except:
                            price_sol = 0.0
                        try:
                            price_usd = float(pair.get("priceUsd") or 0)
                        except:
                            price_usd = 0.0
                        
                        if price_sol > 0:
                            fetched_addrs.add(addr)
                            new_liq = float((pair.get("liquidity") or {}).get("usd") or 0)
                            
                            # Always update cache with fresh data (this is prefetch!)
                            _PRICE_CACHE[addr] = {
                                "ts": now,
                                "price_sol": price_sol,
                                "price_usd": price_usd,
                                "liquidity": new_liq,
                            }
                            
                            # Also update pool if token is there
                            if addr in _TOKEN_POOL:
                                _TOKEN_POOL[addr]["price_sol"] = price_sol
                                _TOKEN_POOL[addr]["price_usd"] = price_usd
                                _TOKEN_POOL[addr]["discovered_at"] = now
        except Exception as e:
            print(f"prefetch_token_data batch error: {e}")
    
    # Fallback: individually fetch any addresses that weren't returned in batch
    missing = [a for a in addresses if a not in fetched_addrs]
    for addr in missing[:5]:  # Limit individual fetches to avoid rate limits
        if _is_rate_limited():
            break
        
        pairs = _fetch_token_pairs(addr)
        if pairs:
            best = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
            try:
                price_sol = float(best.get("priceNative") or 0)
                price_usd = float(best.get("priceUsd") or 0)
                if price_sol > 0:
                    _PRICE_CACHE[addr] = {
                        "ts": now,
                        "price_sol": price_sol,
                        "price_usd": price_usd,
                        "liquidity": float((best.get("liquidity") or {}).get("usd") or 0),
                    }
                    if addr in _TOKEN_POOL:
                        _TOKEN_POOL[addr]["price_sol"] = price_sol
                        _TOKEN_POOL[addr]["price_usd"] = price_usd
            except:
                pass
        time.sleep(0.1)  # Small delay between individual fetches
