"""
pumpfun_stream.py - Real-time pump.fun token streaming via PumpPortal WebSocket

This module connects to PumpPortal.fun's free WebSocket API to get real-time
token data directly from pump.fun (tokens still on the bonding curve).
"""
import asyncio
import json
import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import websockets
except ImportError:
    websockets = None

# =====================================================================
# CONFIGURATION
# =====================================================================
PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"
MAX_TOKENS = 100  # Max tokens to keep in memory
TOKEN_EXPIRE_SECONDS = 30 * 60  # Remove tokens older than 30 min

# =====================================================================
# GLOBAL STATE
# =====================================================================
_TOKENS: Dict[str, Dict[str, Any]] = {}  # address -> token data
_TRADES: Dict[str, List[Dict]] = {}  # address -> list of recent trades
_LOCK = threading.Lock()
_WS_THREAD: Optional[threading.Thread] = None
_RUNNING = False
_LAST_ERROR = ""
_CONNECTED = False
_STATS = {
    "tokens_received": 0,
    "trades_received": 0,
    "last_token_time": 0,
    "connection_time": 0,
}


@dataclass
class PumpToken:
    """Token data from pump.fun"""
    address: str
    symbol: str
    name: str
    uri: str  # metadata URI
    creator: str
    created_at: float
    market_cap_sol: float = 0.0
    bonding_curve_progress: float = 0.0
    price_sol: float = 0.0
    volume_sol: float = 0.0
    tx_count: int = 0


def _now_ts() -> float:
    return time.time()


def _add_token(data: Dict[str, Any]) -> None:
    """Add or update a token from WebSocket event."""
    global _STATS
    
    address = data.get("mint") or data.get("token") or ""
    if not address:
        return
    
    with _LOCK:
        now = _now_ts()
        
        # Parse token data from WebSocket message
        # Pump.fun bonding curve: Total supply is 1 billion tokens
        # Price = market_cap_sol / 1_000_000_000
        TOTAL_SUPPLY = 1_000_000_000
        
        market_cap_sol = float(data.get("marketCapSol") or data.get("vSolInBondingCurve") or 0)
        v_sol = float(data.get("vSolInBondingCurve") or market_cap_sol)
        
        # Calculate price from market cap
        price_sol = market_cap_sol / TOTAL_SUPPLY if market_cap_sol > 0 else 0
        
        # Calculate bonding curve progress (graduates at ~85 SOL)
        bc_progress = min((v_sol / 85.0) * 100, 100) if v_sol > 0 else 0
        
        token = {
            "address": address,
            "symbol": str(data.get("symbol") or data.get("name", "")[:8] or "???"),
            "name": str(data.get("name") or "Unknown"),
            "uri": str(data.get("uri") or ""),
            "creator": str(data.get("traderPublicKey") or data.get("creator") or ""),
            "created_at": now,
            "market_cap_sol": market_cap_sol,
            "bonding_curve_progress": bc_progress,
            "price_sol": price_sol,
            "volume_sol": 0.0,
            "tx_count": 0,
            "source": "pumpfun",
        }
        
        _TOKENS[address] = token
        _STATS["tokens_received"] += 1
        _STATS["last_token_time"] = now
        
        # Cleanup old tokens
        if len(_TOKENS) > MAX_TOKENS:
            # Remove oldest tokens
            sorted_tokens = sorted(_TOKENS.items(), key=lambda x: x[1].get("created_at", 0))
            for addr, _ in sorted_tokens[:len(_TOKENS) - MAX_TOKENS]:
                del _TOKENS[addr]


def _add_trade(data: Dict[str, Any]) -> None:
    """Add a trade event."""
    global _STATS
    
    address = data.get("mint") or data.get("token") or ""
    if not address:
        return
    
    with _LOCK:
        now = _now_ts()
        
        trade = {
            "ts": now,
            "type": "BUY" if data.get("txType") == "buy" else "SELL",
            "sol_amount": float(data.get("solAmount") or 0) / 1e9,  # Convert lamports
            "token_amount": float(data.get("tokenAmount") or 0),
            "trader": str(data.get("traderPublicKey") or ""),
            "new_market_cap_sol": float(data.get("marketCapSol") or 0),
        }
        
        if address not in _TRADES:
            _TRADES[address] = []
        _TRADES[address].append(trade)
        
        # Keep only last 50 trades per token
        if len(_TRADES[address]) > 50:
            _TRADES[address] = _TRADES[address][-50:]
        
        # Update token market cap and price
        TOTAL_SUPPLY = 1_000_000_000
        new_mc = trade["new_market_cap_sol"]
        
        if address in _TOKENS:
            _TOKENS[address]["market_cap_sol"] = new_mc
            _TOKENS[address]["tx_count"] = _TOKENS[address].get("tx_count", 0) + 1
            _TOKENS[address]["volume_sol"] = _TOKENS[address].get("volume_sol", 0) + trade["sol_amount"]
            
            # Update price from market cap
            if new_mc > 0:
                _TOKENS[address]["price_sol"] = new_mc / TOTAL_SUPPLY
                _TOKENS[address]["bonding_curve_progress"] = min((new_mc / 85.0) * 100, 100)
        else:
            # Token not in our list - add it from trade data
            symbol = str(data.get("symbol") or "???")
            if symbol != "???":
                _TOKENS[address] = {
                    "address": address,
                    "symbol": symbol,
                    "name": str(data.get("name") or symbol),
                    "created_at": now,
                    "market_cap_sol": new_mc,
                    "bonding_curve_progress": min((new_mc / 85.0) * 100, 100) if new_mc > 0 else 0,
                    "price_sol": new_mc / TOTAL_SUPPLY if new_mc > 0 else 0,
                    "volume_sol": trade["sol_amount"],
                    "tx_count": 1,
                    "source": "pumpfun",
                }
                _STATS["tokens_received"] += 1
        
        _STATS["trades_received"] += 1


async def _websocket_loop():
    """Main WebSocket connection loop."""
    global _RUNNING, _LAST_ERROR, _CONNECTED, _STATS
    
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    while _RUNNING:
        try:
            _LAST_ERROR = ""
            print("[PumpPortal] Connecting to WebSocket...")
            
            async with websockets.connect(PUMPPORTAL_WS, ssl=ssl_context) as ws:
                _CONNECTED = True
                _STATS["connection_time"] = _now_ts()
                print("[PumpPortal] Connected! Subscribing to events...")
                
                # Subscribe to new token events
                await ws.send(json.dumps({"method": "subscribeNewToken"}))
                
                # Listen for messages
                async for message in ws:
                    if not _RUNNING:
                        break
                    
                    try:
                        data = json.loads(message)
                        
                        # Handle different message types
                        if "mint" in data and "name" in data:
                            # New token event
                            _add_token(data)
                            print(f"[PumpPortal] New token: {data.get('symbol', 'N/A')} | MC: {data.get('marketCapSol', 0):.2f} SOL")
                        
                        elif "txType" in data:
                            # Trade event
                            _add_trade(data)
                        
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"[PumpPortal] Message error: {e}")
        
        except Exception as e:
            _CONNECTED = False
            _LAST_ERROR = str(e)
            print(f"[PumpPortal] Connection error: {e}")
            
            if _RUNNING:
                print("[PumpPortal] Reconnecting in 5 seconds...")
                await asyncio.sleep(5)


def _run_websocket():
    """Run the WebSocket loop in a thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_websocket_loop())


def start_stream() -> bool:
    """Start the WebSocket stream in a background thread."""
    global _WS_THREAD, _RUNNING
    
    if websockets is None:
        print("[PumpPortal] websockets library not installed")
        return False
    
    if _WS_THREAD is not None and _WS_THREAD.is_alive():
        return True  # Already running
    
    _RUNNING = True
    _WS_THREAD = threading.Thread(target=_run_websocket, daemon=True)
    _WS_THREAD.start()
    print("[PumpPortal] Stream started")
    return True


def stop_stream() -> None:
    """Stop the WebSocket stream."""
    global _RUNNING, _CONNECTED
    _RUNNING = False
    _CONNECTED = False


def is_connected() -> bool:
    """Check if WebSocket is connected."""
    return _CONNECTED


def get_tokens() -> List[Dict[str, Any]]:
    """Get all tracked pump.fun tokens, sorted by market cap."""
    with _LOCK:
        now = _now_ts()
        # Filter out expired tokens
        tokens = [
            t for t in _TOKENS.values()
            if (now - t.get("created_at", 0)) < TOKEN_EXPIRE_SECONDS
        ]
        # Sort by market cap (highest first)
        tokens.sort(key=lambda x: x.get("market_cap_sol", 0), reverse=True)
        return tokens


def get_token(address: str) -> Optional[Dict[str, Any]]:
    """Get a specific token by address."""
    with _LOCK:
        return _TOKENS.get(address)


def get_trades(address: str) -> List[Dict]:
    """Get recent trades for a token."""
    with _LOCK:
        return list(_TRADES.get(address, []))


def get_stats() -> Dict[str, Any]:
    """Get stream statistics."""
    return {
        **_STATS,
        "connected": _CONNECTED,
        "tokens_tracked": len(_TOKENS),
        "last_error": _LAST_ERROR,
    }


def clear_tokens() -> None:
    """Clear all tracked tokens."""
    global _TOKENS, _TRADES, _STATS
    with _LOCK:
        _TOKENS.clear()
        _TRADES.clear()
        _STATS = {
            "tokens_received": 0,
            "trades_received": 0,
            "last_token_time": 0,
            "connection_time": _STATS.get("connection_time", 0),
        }


# Auto-start when module is imported
if websockets is not None:
    start_stream()
