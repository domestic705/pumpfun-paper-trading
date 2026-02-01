"""
bot_logic.py - Pump.fun Paper Trading Strategy V2

IMPROVED STRATEGY based on research:
1. MOMENTUM ENTRY: Only buy tokens with positive price momentum
2. VOLUME CONFIRMATION: Require significant volume activity  
3. HIGHER TAKE PROFIT: 50-100% targets (meme coins move fast)
4. TRAILING STOP: Lock in profits as price rises
5. FASTER TIMEOUT: 3-5 min (pump.fun moves in seconds)

Key insights:
- Most pump.fun tokens fail - be selective
- Winners can 2-10x quickly - let them run
- Cut losers fast - don't wait for recovery
- Volume spike = interest = potential pump

This is PAPER TRADING - no real money is used.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from market_data import fetch_candidates, get_token_price
from state_manager import StateManager


# =====================================================================
# STRATEGY CONFIGURATION V2 - RESEARCH OPTIMIZED
# =====================================================================

# Position Sizing
BUY_SIZE_SOL = 0.5           # Amount per trade
MAX_OPEN_POSITIONS = 10      # Max simultaneous positions
MAX_BUYS_PER_TICK = 2        # Up to 2 buys per cycle (catch momentum)

# ===================
# ENTRY FILTERS - STRICT (avoid dead tokens!)
# ===================
# Volume & Liquidity (proof of interest)
MIN_VOLUME_H1 = 5000         # Min $5K volume/hour (INCREASED - need active trading)
MIN_LIQUIDITY_USD = 10000    # Min $10K liquidity (INCREASED - avoid low-liq rugs)

# Market Cap Range
MIN_MARKET_CAP = 30000       # Min $30K (INCREASED - avoid dust)
MAX_MARKET_CAP = 500_000     # Max $500K (focus on early growth, not late stage)

# Price filters
MIN_PRICE_USD = 0.0000001    # Minimum price

# MOMENTUM FILTERS (CRITICAL for avoiding dead tokens!)
MIN_PRICE_CHANGE_H1 = 10.0   # Must be UP at least 10% in last hour (strong momentum)
MIN_TX_COUNT = 50            # Minimum 50 transactions (REAL activity, not dead)
MIN_VOLUME_M5 = 500          # Must have $500 volume in last 5 MINUTES (actively trading NOW)

# ===================
# EXIT STRATEGY
# ===================
# Main exits
TAKE_PROFIT_PCT = 50.0       # 50% TP (was 25% - too conservative)
STOP_LOSS_PCT = -15.0        # 15% SL (was -10% - getting stopped too early)
TIME_EXIT_SECONDS = 10 * 60  # 10 min timeout (need time for price to move)
MIN_EXIT_CHANGE_PCT = 3.0    # Only timeout exit if price moved at least 3% (avoid commission-only losses)

# Trailing stop (NEW!)
TRAILING_STOP_ENABLED = True
TRAILING_STOP_ACTIVATION = 25.0  # Activate trailing after 25% gain
TRAILING_STOP_DISTANCE = 10.0    # Trail 10% behind peak

# Default commission (pump.fun charges ~1% swap fee)
DEFAULT_COMMISSION_PCT = 1.0


def run_strategy(
    state_manager: StateManager,
    trade_pumpfun: bool = True,
    trade_dex: bool = True,
    max_positions: int = 10,
    commission_pct: float = 0.0,
    strategy_config: dict = None,  # All strategy parameters
) -> Dict[str, Any]:
    """
    Execute the pump.fun paper trading strategy.
    
    Args:
        state_manager: The state manager instance
        trade_pumpfun: Whether to trade pump.fun bonding curve tokens
        trade_dex: Whether to trade graduated DEX tokens
        max_positions: Maximum number of simultaneous open positions
        commission_pct: Trading fee as percentage
        strategy_config: Dict with all strategy parameters from UI
    """
    # Use config or defaults
    cfg = strategy_config or {}
    
    # Position sizing
    buy_size_sol = float(cfg.get("buy_size", BUY_SIZE_SOL))
    
    # Exit settings
    take_profit_pct = float(cfg.get("take_profit", TAKE_PROFIT_PCT))
    stop_loss_pct = float(cfg.get("stop_loss", STOP_LOSS_PCT))
    timeout_seconds = float(cfg.get("timeout_min", TIME_EXIT_SECONDS / 60)) * 60
    min_exit_change = float(cfg.get("min_exit_change", MIN_EXIT_CHANGE_PCT))
    
    # Trailing stop
    trailing_enabled = bool(cfg.get("trailing_enabled", TRAILING_STOP_ENABLED))
    trailing_activation = float(cfg.get("trailing_activation", TRAILING_STOP_ACTIVATION))
    trailing_distance = float(cfg.get("trailing_distance", TRAILING_STOP_DISTANCE))
    
    # Entry filters
    min_volume_h1 = float(cfg.get("min_volume_h1", MIN_VOLUME_H1))
    min_volume_m5 = float(cfg.get("min_volume_m5", MIN_VOLUME_M5))
    min_liquidity = float(cfg.get("min_liquidity", MIN_LIQUIDITY_USD))
    min_mcap = float(cfg.get("min_mcap", MIN_MARKET_CAP))
    max_mcap = float(cfg.get("max_mcap", MAX_MARKET_CAP))
    min_price_change = float(cfg.get("min_price_change", MIN_PRICE_CHANGE_H1))
    min_txns = int(cfg.get("min_txns", MIN_TX_COUNT))
    
    # Cooldown
    cooldown_minutes = float(cfg.get("cooldown_min", state_manager.COOLDOWN_SECONDS / 60))
    state_manager.COOLDOWN_SECONDS = cooldown_minutes * 60
    actions: List[Dict[str, Any]] = []
    candidates = fetch_candidates()
    
    now = time.time()
    
    # =========================================================
    # BUY LOGIC
    # =========================================================
    buys_this_tick = 0
    
    for c in candidates:
        # Filter by platform
        source = str(c.get("source", "dexscreener"))
        addr = str(c.get("address", ""))
        # Consider both bonding curve and graduated pump.fun tokens
        is_pumpfun = source in ("pumpfun_ws", "pumpfun_graduated") or addr.lower().endswith("pump")
        
        if is_pumpfun and not trade_pumpfun:
            continue
        if not is_pumpfun and not trade_dex:
            continue
        
        # Position limits
        if len(state_manager.active_trades) >= max_positions:
            break
        if buys_this_tick >= MAX_BUYS_PER_TICK:
            break
        if state_manager.virtual_balance < buy_size_sol:
            break
        
        symbol = str(c.get("symbol", "???"))
        
        # Skip unknown/invalid symbols
        if symbol.lower() in ("unknown", "???", "", "null", "none"):
            continue
        
        # Skip if already owned OR on cooldown (recently sold)
        if not state_manager.can_buy(addr):
            continue
        
        # Extract all metrics
        volume_h1 = float(c.get("volume_h1", 0))
        liquidity = float(c.get("liquidity_usd", 0))
        market_cap = float(c.get("market_cap", 0))
        price_sol = float(c.get("price_sol", 0))
        price_usd = float(c.get("price_usd", 0))
        price_change_h1 = float(c.get("price_change_h1", 0))
        tx_count = int(c.get("tx_count", c.get("txns_h1", 0)))
        
        # =====================================================
        # IMPROVED ENTRY CRITERIA V2
        # =====================================================
        
        if is_pumpfun:
            # PUMP.FUN TOKENS: Focus on bonding curve momentum
            if market_cap < 5000:  # Min $5K (avoid ultra-new)
                continue
            if market_cap > 500000:  # Max $500K (before graduation)
                continue
            
            # Calculate price from market cap if not available
            if price_sol <= 0 and market_cap > 0:
                sol_price_usd = 100  # Approximate
                market_cap_sol = market_cap / sol_price_usd
                price_sol = market_cap_sol / 1_000_000_000  # 1B supply
            
            if price_sol <= 0:
                continue
                
            # MOMENTUM CHECK: Want tokens that are moving UP
            # (bonding curve progress increasing = more buyers)
            bonding_pct = float(c.get("bonding_curve_progress", 0))
            if bonding_pct > 0 and bonding_pct < 10:
                continue  # Too early, wait for some traction
                
        else:
            # DEX TOKENS: Volume + Momentum confirmation
            volume_m5_val = float(c.get("volume_m5", 0))
            txns_m5 = int(c.get("txns_m5", 0))
            
            # Basic filters (using configurable values)
            if volume_h1 < min_volume_h1:
                continue
            if liquidity < min_liquidity:
                continue
            if market_cap < min_mcap or market_cap > max_mcap:
                continue
            if price_sol <= 0 or price_usd < MIN_PRICE_USD:
                continue
            
            # *** RECENT ACTIVITY FILTER (configurable) ***
            if min_volume_m5 > 0 and volume_m5_val < min_volume_m5:
                continue  # No recent trading = DEAD token, skip!
            
            if min_volume_m5 > 0 and txns_m5 < 3:
                continue  # Less than 3 trades in 5 min = basically dead
            
            # *** MOMENTUM FILTER (configurable) ***
            if price_change_h1 < min_price_change:
                continue  # Price not rising = skip
            
            # *** ACTIVITY FILTER (configurable) ***
            if min_txns > 0 and tx_count > 0 and tx_count < min_txns:
                continue  # Not enough total activity
        
        # Execute buy - use correct source for pump.fun tokens
        trade_source = "pumpfun" if is_pumpfun else "dexscreener"
        ok = state_manager.buy_token(
            symbol=symbol,
            address=addr,
            price=price_sol,
            amount_sol=buy_size_sol,
            source=trade_source,
            commission_pct=commission_pct,
        )
        
        if ok:
            buys_this_tick += 1
            actions.append({
                "type": "BUY",
                "symbol": symbol,
                "address": addr,
                "price": price_sol,
                "amount_sol": BUY_SIZE_SOL,
                "reason": f"vol=${volume_h1:,.0f}/h, liq=${liquidity:,.0f}, mc=${market_cap:,.0f}",
                "ts": now,
            })
            print(f"📈 BUY: {symbol} @ {price_sol:.10f} SOL | MC: ${market_cap:,.0f} | Vol: ${volume_h1:,.0f}/h")
    
    # =========================================================
    # SELL LOGIC V2 - WITH TRAILING STOP
    # =========================================================
    active = list(state_manager.active_trades)
    
    for t in active:
        addr = str(t["address"])
        symbol = str(t.get("symbol", "???"))
        entry_price = float(t.get("entry_price", 0.0))
        entry_ts = float(t.get("entry_ts", 0.0))
        peak_price = float(t.get("peak_price", entry_price))
        
        if entry_price <= 0:
            continue
        
        current_price = float(get_token_price(addr))
        if current_price <= 0:
            continue
        
        # =====================================================
        # PRICE SANITY CHECK
        # =====================================================
        # Reject obviously bad price data (>60% swing in short time is suspicious)
        price_ratio = current_price / entry_price
        held_seconds = max(0.0, now - entry_ts)
        
        # If price dropped more than 60% AND held less than 2 minutes, likely bad data
        if price_ratio < 0.4 and held_seconds < 120:
            print(f"⚠️ SUSPICIOUS PRICE for {symbol}: {entry_price:.10f} -> {current_price:.10f} ({(price_ratio-1)*100:.1f}%) in {held_seconds:.0f}s - SKIPPING")
            continue
        
        # If price spiked more than 200% AND held less than 1 minute, also suspicious  
        if price_ratio > 3.0 and held_seconds < 60:
            print(f"⚠️ SUSPICIOUS SPIKE for {symbol}: {entry_price:.10f} -> {current_price:.10f} ({(price_ratio-1)*100:.1f}%) in {held_seconds:.0f}s - SKIPPING")
            continue
        
        # Update peak price tracking
        if current_price > peak_price:
            state_manager.update_peak_price(addr, current_price)
            peak_price = current_price
        
        change_pct = ((current_price / entry_price) - 1.0) * 100.0
        peak_change_pct = ((peak_price / entry_price) - 1.0) * 100.0
        
        should_sell = False
        reason = ""
        
        # =====================================================
        # EXIT CONDITIONS (using configurable parameters)
        # =====================================================
        
        # 1. TAKE PROFIT - Hit target
        if change_pct >= take_profit_pct:
            should_sell = True
            reason = f"🎯 TP hit(+{change_pct:.1f}%)"
        
        # 2. TRAILING STOP - Lock in profits on winners
        elif trailing_enabled and peak_change_pct >= trailing_activation:
            # Calculate trailing stop level
            trailing_stop_level = peak_price * (1 - trailing_distance / 100.0)
            drop_from_peak_pct = ((peak_price - current_price) / peak_price) * 100.0
            
            if current_price <= trailing_stop_level:
                should_sell = True
                reason = f"📉 Trailing stop (peak +{peak_change_pct:.0f}%, dropped {drop_from_peak_pct:.1f}%)"
        
        # 3. STOP LOSS - Cut losses
        elif change_pct <= stop_loss_pct:
            should_sell = True
            reason = f"🛑 Stop loss({change_pct:.1f}%)"
        
        # 4. TIME EXIT - Don't hold forever, but only if there's meaningful movement
        elif held_seconds >= timeout_seconds:
            # Only exit on timeout if price moved significantly (avoid pure commission losses)
            if abs(change_pct) >= min_exit_change:
                should_sell = True
                reason = f"⏰ Timeout({held_seconds/60:.1f}m, {change_pct:+.1f}%)"
            elif held_seconds >= timeout_seconds * 2:
                # Force exit after 2x timeout regardless of price
                should_sell = True
                reason = f"⏰ Force exit({held_seconds/60:.1f}m, {change_pct:+.1f}%)"
        
        if should_sell:
            closed = state_manager.sell_token(addr, current_price, commission_pct=commission_pct)
            if closed:
                pnl_sol = float(closed.get("pnl_sol", 0.0))
                pnl_pct = float(closed.get("pnl_pct", 0.0))
                
                emoji = "✅" if pnl_sol > 0 else "❌"
                print(f"{emoji} SELL: {symbol} | PnL: {pnl_sol:+.4f} SOL ({pnl_pct:+.1f}%) | {reason}")
                
                actions.append({
                    "type": "SELL",
                    "symbol": symbol,
                    "address": addr,
                    "price": current_price,
                    "pnl_sol": pnl_sol,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                    "ts": now,
                })
    
    return {"actions": actions, "candidates": candidates}


def get_strategy_description() -> str:
    """Return strategy description."""
    trailing = f"Trailing Stop: +{TRAILING_STOP_ACTIVATION:.0f}% trigger, {TRAILING_STOP_DISTANCE:.0f}% trail" if TRAILING_STOP_ENABLED else "Trailing Stop: Disabled"
    return f"""
**Pump.fun Momentum Strategy V2** 🚀

**Entry Rules (Selective):**
- 📊 Min volume: ${MIN_VOLUME_H1:,}/hr
- 💧 Min liquidity: ${MIN_LIQUIDITY_USD:,}
- 📈 Momentum: +{MIN_PRICE_CHANGE_H1:.0f}% (1hr)
- 🔢 Min transactions: {MIN_TX_COUNT}
- 💰 Market cap: ${MIN_MARKET_CAP:,} - ${MAX_MARKET_CAP:,}
- 🎯 Position: {BUY_SIZE_SOL} SOL

**Exit Rules (Smart):**
- 🎯 Take Profit: +{TAKE_PROFIT_PCT:.0f}%
- 📉 {trailing}
- 🛑 Stop Loss: {STOP_LOSS_PCT:.0f}%
- ⏰ Timeout: {TIME_EXIT_SECONDS/60:.0f} min
"""
