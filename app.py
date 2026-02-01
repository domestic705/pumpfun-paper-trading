"""
Solana Paper Trading Dashboard for Pump.fun

This dashboard simulates trading on pump.fun tokens using REAL price data
from pumpfunapi.org. No real money is used - it's for strategy testing.

Data Source: pumpfunapi.org (free, no API key)
"""
from __future__ import annotations

import os
import time
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from bot_logic import run_strategy, get_strategy_description
from market_data import (
    fetch_candidates,
    get_chart_data,
    get_last_candidate_stats,
    get_last_error,
    get_price_sample_stats,
    get_dexscreener_embed_url,
    prefetch_token_data,
    reset_runtime_state,
    get_token_price,
    get_stats,
    add_token_manually,
    get_sol_usd_price,
)
from state_manager import StateManager

# Start pump.fun WebSocket stream (once per session)
if "pumpfun_started" not in st.session_state:
    try:
        from pumpfun_stream import start_stream, is_connected
        start_stream()
        st.session_state["pumpfun_started"] = True
    except Exception:
        # Handle any import/connection errors gracefully
        st.session_state["pumpfun_started"] = False


def _rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def _fmt_sol(x: float) -> str:
    return f"{x:,.4f} SOL"


def _fmt_usd(x: float) -> str:
    if abs(x) < 0.01:
        return f"${x:.6f}"
    elif abs(x) < 1:
        return f"${x:.4f}"
    return f"${x:,.2f}"


def _sol_to_usd(sol: float, sol_price: float) -> float:
    return sol * sol_price


# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Pump.fun Paper Trading",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Pump.fun Paper Trading Dashboard")
st.caption(f"Real-time paper trading with LIVE pump.fun token data from pumpfunapi.org | Session: {st.session_state.get('session_id', 'local')}")

# Generate unique session ID for each user
import os
import uuid

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

# Detect if running on Streamlit Cloud (check multiple env vars)
IS_CLOUD = bool(
    os.environ.get("STREAMLIT_SHARING_MODE") or 
    os.environ.get("STREAMLIT_SERVER_HEADLESS") or
    os.environ.get("HOSTNAME", "").startswith("streamlit")
)

# Initialize state manager (per-user session state)
# On cloud: use memory_only=True for complete isolation
# Locally: use file-based persistence
if "state_manager" not in st.session_state:
    st.session_state["state_manager"] = StateManager(memory_only=True)  # Always memory-only for cloud safety
state = st.session_state["state_manager"]

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Auto-refresh settings
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_seconds = st.slider("Refresh interval (seconds)", 1, 10, 3)
    
    st.divider()
    
    # Strategy toggle
    run_bot = st.checkbox("🤖 Run Strategy", value=True, help="Enable automatic buy/sell based on strategy rules")
    
    # =====================================================
    # STRATEGY PARAMETERS (all adjustable!)
    # =====================================================
    
    with st.expander("⚙️ **Strategy Settings**", expanded=False):
        st.caption("📊 **Position Settings:**")
        max_positions = st.slider("Max Open Positions", 1, 50, 20, help="Maximum simultaneous trades")
        buy_size = st.slider("Buy Size (SOL)", 0.1, 2.0, 0.5, 0.1, help="Amount per trade")
        
        st.caption("🎯 **Exit Settings:**")
        take_profit = st.slider("Take Profit %", 10, 200, 50, 5, help="Sell when profit reaches this %")
        stop_loss = st.slider("Stop Loss %", -50, -5, -15, 5, help="Sell when loss reaches this %")
        timeout_min = st.slider("Timeout (minutes)", 5, 60, 10, 5, help="Force sell after this time if price moved")
        
        st.caption("📈 **Trailing Stop:**")
        trailing_enabled = st.checkbox("Enable Trailing Stop", value=True, help="Lock in profits as price rises")
        trailing_activation = st.slider("Trailing Activation %", 10, 50, 25, 5, help="Activate trailing after this gain")
        trailing_distance = st.slider("Trailing Distance %", 5, 25, 10, 5, help="Trail this % behind peak")
        
        st.caption("🔍 **Entry Filters (DEX tokens):**")
        min_volume_h1 = st.slider("Min Volume/hr ($)", 500, 20000, 5000, 500, help="Minimum hourly volume")
        min_volume_m5 = st.slider("Min Volume/5min ($)", 0, 2000, 500, 100, help="Must be trading NOW (0=disabled)")
        min_liquidity = st.slider("Min Liquidity ($)", 1000, 50000, 10000, 1000, help="Minimum liquidity")
        min_mcap = st.slider("Min Market Cap ($K)", 5, 100, 30, 5, help="Minimum market cap in thousands")
        max_mcap = st.slider("Max Market Cap ($K)", 100, 2000, 500, 50, help="Maximum market cap in thousands")
        min_price_change = st.slider("Min Price Change % (1hr)", -10, 50, 10, 5, help="Momentum filter")
        min_txns = st.slider("Min Transactions (1hr)", 0, 200, 50, 10, help="Activity filter (0=disabled)")
        
        st.caption("⏱️ **Cooldown:**")
        cooldown_min = st.slider("Cooldown (minutes)", 0, 60, 5, 1, help="Wait time before re-buying same token (0=disabled)")
        
        # Store all in session state
        st.session_state["strategy"] = {
            "max_positions": max_positions,
            "buy_size": buy_size,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "timeout_min": timeout_min,
            "trailing_enabled": trailing_enabled,
            "trailing_activation": trailing_activation,
            "trailing_distance": trailing_distance,
            "min_volume_h1": min_volume_h1,
            "min_volume_m5": min_volume_m5,
            "min_liquidity": min_liquidity,
            "min_mcap": min_mcap * 1000,  # Convert to actual $
            "max_mcap": max_mcap * 1000,
            "min_price_change": min_price_change,
            "min_txns": min_txns,
            "cooldown_min": cooldown_min,
        }
    
    # Quick settings outside expander
    st.caption("⚙️ **Quick Settings:**")
    max_positions = st.session_state.get("strategy", {}).get("max_positions", 10)
    st.text(f"Max Positions: {max_positions}")
    
    # Commission settings
    enable_commissions = st.checkbox("💰 Simulate Commissions (1%)", value=True, help="Include swap fees")
    commission_pct = 1.0 if enable_commissions else 0.0
    st.session_state["commission_pct"] = commission_pct
    
    st.divider()
    
    # Platform toggles
    st.caption("📡 Trading Platforms:")
    trade_pumpfun = st.checkbox("🔥 pump.fun (Bonding Curve)", value=True, help="Trade tokens on pump.fun bonding curve")
    trade_dex = st.checkbox("📈 DEX (Graduated)", value=True, help="Trade graduated tokens on DEX")
    
    # Store in session state for bot_logic to access
    st.session_state["trade_pumpfun"] = trade_pumpfun
    st.session_state["trade_dex"] = trade_dex
    
    st.divider()
    
    # Manual controls
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Refresh"):
            _rerun()
    with col_b:
        if st.button("📊 Reset Account"):
            state.reset()
            _rerun()
    
    # Hard reset with confirmation
    st.markdown(
        """<style>
        div[data-testid="stSidebar"] button[kind="primary"] {
          background-color: #d32f2f !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    
    if st.button("🔴 Hard Reset (Clear All)", type="primary"):
        data_path = os.path.join(os.path.dirname(__file__), "paper_data.json")
        if os.path.exists(data_path):
            os.remove(data_path)
        reset_runtime_state()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _rerun()
    
    st.divider()
    
    # Manually add token
    st.subheader("📥 Add Token Manually")
    st.caption("Paste a token address from pump.fun to track it")
    
    manual_address = st.text_input("Token Address", placeholder="Enter mint address...", key="manual_token_input")
    manual_symbol = st.text_input("Symbol (optional)", placeholder="e.g. PEPE", value="MANUAL", key="manual_symbol_input")
    
    if st.button("➕ Add Token"):
        if manual_address:
            success = add_token_manually(manual_address.strip(), manual_symbol.strip() or "MANUAL")
            if success:
                st.success(f"Added {manual_symbol or manual_address[:8]}...")
                _rerun()
            else:
                st.warning("Token already tracked or invalid address")
    
    st.divider()
    
    # Strategy info
    st.subheader("📋 Strategy Rules")
    st.markdown(get_strategy_description())
    
    # Stats
    stats = get_stats()
    st.divider()
    st.subheader("📊 Session Stats")
    st.write(f"Tracked tokens: **{stats.get('total_tracked', 0)}**")
    st.write(f"New this session: **{stats.get('new_this_session', 0)}**")
    if stats.get("last_new_token"):
        last = stats["last_new_token"]
        st.write(f"Latest: **{last.get('symbol', '?')}**")


# =====================================================================
# MAIN CONTENT
# =====================================================================
# Get platform settings from session state
trade_pumpfun_enabled = st.session_state.get("trade_pumpfun", True)
trade_dex_enabled = st.session_state.get("trade_dex", True)

# Run strategy or just fetch candidates
strategy_config = st.session_state.get("strategy", {})
max_positions_setting = strategy_config.get("max_positions", 10)
commission_setting = st.session_state.get("commission_pct", 0.0)
if run_bot:
    result = run_strategy(
        state, 
        trade_pumpfun=trade_pumpfun_enabled, 
        trade_dex=trade_dex_enabled,
        max_positions=max_positions_setting,
        commission_pct=commission_setting,
        strategy_config=strategy_config,
    )
else:
    result = {"actions": [], "candidates": fetch_candidates()}

# Show errors
err = get_last_error()
if err:
    st.warning(f"⚠️ API Note: {err}")

# Get SOL/USD price for conversions
sol_usd_price = get_sol_usd_price()

# Prefetch prices for active trades (forces fresh API calls)
from datetime import datetime
price_fetch_time = datetime.now()
prefetch_token_data([t["address"] for t in state.active_trades])
current_prices: Dict[str, float] = {
    t["address"]: float(get_token_price(t["address"])) for t in state.active_trades
}
st.session_state["last_price_fetch"] = price_fetch_time
pnl = state.get_pnl(current_prices)
realized_pnl = sum(float(t.get("pnl_sol", 0.0)) for t in state.trade_history)

# Get commission settings - if disabled, we add back commissions retroactively
commission_pct = st.session_state.get("commission_pct", 1.0)
show_with_commissions = commission_pct > 0

# When commissions are OFF, recalculate everything as if no commissions existed
if show_with_commissions:
    # Normal display with commissions
    equity_sol_display = pnl["equity_sol"]
    balance_usd = state.virtual_balance * sol_usd_price
    active_value_usd = pnl["active_value_sol"] * sol_usd_price
    equity_usd = equity_sol_display * sol_usd_price
    unrealized_usd = pnl["unrealized_pnl_sol"] * sol_usd_price
    realized_usd = realized_pnl * sol_usd_price
else:
    # No-commission display: add back all commissions
    commission_adjustment_sol = state.total_commissions
    
    balance_sol_display = state.virtual_balance + commission_adjustment_sol
    equity_sol_display = pnl["equity_sol"] + commission_adjustment_sol
    
    # Calculate what total_pnl should be without commissions
    total_pnl_no_comm = equity_sol_display - state.starting_balance_sol
    
    # Unrealized stays the same (it doesn't include commission in cost)
    unrealized_sol_display = pnl["unrealized_pnl_sol"]
    
    # Realized gets whatever adjustment is needed to make the math work:
    # unrealized + realized = total_pnl
    realized_sol_display = total_pnl_no_comm - unrealized_sol_display
    
    balance_usd = balance_sol_display * sol_usd_price
    active_value_usd = pnl["active_value_sol"] * sol_usd_price
    equity_usd = equity_sol_display * sol_usd_price
    unrealized_usd = unrealized_sol_display * sol_usd_price
    realized_usd = realized_sol_display * sol_usd_price

# =====================================================================
# METRICS (all in USD)
# =====================================================================
st.caption(f"SOL Price: ${sol_usd_price:.2f}")

# Calculate equity growth percentage from starting balance (using adjusted equity)
starting_balance_usd = state.starting_balance_sol * sol_usd_price
equity_growth_pct = ((equity_sol_display - state.starting_balance_sol) / state.starting_balance_sol) * 100 if state.starting_balance_sol > 0 else 0

# Calculate all stats
open_trades = len(state.active_trades)
open_wins = 0
open_losses = 0
for t in state.active_trades:
    addr = t.get("address", "")
    entry_price = float(t.get("entry_price", 0))
    current_price = float(current_prices.get(addr, entry_price))
    if current_price > entry_price:
        open_wins += 1
    elif current_price < entry_price:
        open_losses += 1
open_win_rate = (open_wins / open_trades * 100) if open_trades > 0 else 0

closed_trades = len(state.trade_history)
closed_wins = len([t for t in state.trade_history if float(t.get("pnl_sol", 0)) > 0])
closed_losses = closed_trades - closed_wins
closed_win_rate = (closed_wins / closed_trades * 100) if closed_trades > 0 else 0

starting_usd = state.starting_balance_sol * sol_usd_price

# Growth calculations:
# - Total Growth = based on total equity (balance + active) vs starting
# - Unrealized Growth = unrealized PnL / starting (open positions only)
# - Realized Growth = realized PnL / starting (closed trades only)
total_growth_pct = equity_growth_pct  # Already calculated above
unrealized_growth_pct = (unrealized_usd / starting_usd * 100) if starting_usd > 0 else 0
realized_growth_pct = (realized_usd / starting_usd * 100) if starting_usd > 0 else 0

# Total PnL = equity - starting (this is the TRUE PnL including commissions)
total_pnl_usd = equity_usd - starting_usd

from datetime import datetime
reset_time = datetime.fromtimestamp(state.reset_timestamp)
time_since_reset = datetime.now() - reset_time
hours_running = time_since_reset.total_seconds() / 3600
# Show commissions as 0 when disabled (retroactive), otherwise show actual
total_commissions_usd = 0.0 if not show_with_commissions else state.total_commissions * sol_usd_price

# =====================================================================
# ROW 0: TOTAL (Account Overview)
# =====================================================================
st.markdown("### 💎 Total")
total_row = st.columns(8)
total_row[0].metric("🎯 Starting", _fmt_usd(starting_usd))
total_row[1].metric("💎 Total Equity", _fmt_usd(equity_usd))
total_row[2].metric("📊 Total PnL", _fmt_usd(total_pnl_usd))
total_row[3].metric("📈 Total Growth", f"{total_growth_pct:+.1f}%")
total_row[4].metric("💰 Balance", _fmt_usd(balance_usd))
total_row[5].metric("📈 Active Value", _fmt_usd(active_value_usd))
total_row[6].metric("💸 Commissions", _fmt_usd(total_commissions_usd))
total_row[7].metric("🕐 Running", f"{hours_running:.1f}h")

st.divider()

# =====================================================================
# MAIN LAYOUT: LEFT (Open) | RIGHT (Closed)
# =====================================================================
col_left, col_right = st.columns(2)

# ----- LEFT SIDE: OPEN POSITIONS -----
with col_left:
    st.markdown("### 📂 Open Positions")
    
    # Open trades stats
    l1 = st.columns(4)
    l1[0].metric("Trades", open_trades)
    l1[1].metric("🟢 Win", open_wins)
    l1[2].metric("🔴 Lose", open_losses)
    l1[3].metric("Win%", f"{open_win_rate:.0f}%")
    
    # Unrealized PnL (only from open trades)
    l2 = st.columns(2)
    l2[0].metric("📊 Unrealized PnL", _fmt_usd(unrealized_usd))
    l2[1].metric("📈 Unrealized Growth", f"{unrealized_growth_pct:+.1f}%")

# ----- RIGHT SIDE: CLOSED TRADES -----
with col_right:
    st.markdown("### 📁 Closed Trades")
    
    # Closed trades stats
    r1 = st.columns(4)
    r1[0].metric("Trades", closed_trades)
    r1[1].metric("🏆 Win", closed_wins)
    r1[2].metric("❌ Lose", closed_losses)
    r1[3].metric("Win%", f"{closed_win_rate:.0f}%")
    
    # Realized PnL (only from closed trades)
    r2 = st.columns(2)
    r2[0].metric("💵 Realized PnL", _fmt_usd(realized_usd))
    r2[1].metric("📈 Realized Growth", f"{realized_growth_pct:+.1f}%")

# Reset time caption
st.caption(f"🕐 Reset: {reset_time.strftime('%Y-%m-%d %H:%M')} | Total PnL: {_fmt_usd(total_pnl_usd)} | Net PnL: {_fmt_usd(realized_usd - total_commissions_usd)}")

# Record equity snapshot for the curve
state.record_equity_snapshot(current_prices)

# =====================================================================
# EQUITY CURVE
# =====================================================================
if state.equity_history and len(state.equity_history) > 1:
    import plotly.graph_objects as go
    
    with st.expander("📈 **Equity Curve**", expanded=True):
        # Toggle between Total, Realized, and Unrealized
        curve_type = st.radio(
            "Show:",
            ["Total (Balance + Open)", "Realized (Closed Trades)", "Unrealized (Open Positions)"],
            horizontal=True,
            key="equity_curve_type"
        )
        
        # Build data for chart - only include data from after the last reset
        eq_df = pd.DataFrame(state.equity_history)
        eq_df["time"] = pd.to_datetime(eq_df["ts"], unit="s")
        
        # Filter to only data after last reset
        reset_time_dt = pd.to_datetime(state.reset_timestamp, unit="s")
        eq_df = eq_df[eq_df["time"] >= reset_time_dt].copy()
        
        if len(eq_df) == 0:
            # No data since reset, create empty frame
            eq_df = pd.DataFrame(columns=["ts", "time", "equity_sol", "balance_sol", "active_value_sol", "equity_usd", "balance_usd", "unrealized_usd", "pnl_pct"])
        else:
            eq_df["equity_usd"] = eq_df["equity_sol"] * sol_usd_price
            # Handle missing balance_sol in older records - fallback to equity_sol
            if "balance_sol" not in eq_df.columns:
                eq_df["balance_sol"] = eq_df["equity_sol"]
            eq_df["balance_sol"] = eq_df["balance_sol"].fillna(eq_df["equity_sol"])
            eq_df["balance_usd"] = eq_df["balance_sol"] * sol_usd_price  # Realized equity (balance only)
            # Unrealized = active value (equity - balance)
            if "active_value_sol" in eq_df.columns:
                eq_df["unrealized_usd"] = eq_df["active_value_sol"] * sol_usd_price
            else:
                eq_df["unrealized_usd"] = (eq_df["equity_sol"] - eq_df["balance_sol"]) * sol_usd_price
            eq_df["pnl_pct"] = ((eq_df["equity_sol"] / state.starting_balance_sol) - 1) * 100
        
        # Starting balance from last reset
        start_equity = state.starting_balance_sol * sol_usd_price
        reset_time_dt = pd.to_datetime(state.reset_timestamp, unit="s")
        
        # Ensure the chart STARTS at starting balance from reset time
        if len(eq_df) == 0 or eq_df["time"].iloc[0] > reset_time_dt:
            start_row = pd.DataFrame([{
                "ts": state.reset_timestamp,
                "time": reset_time_dt,
                "equity_usd": start_equity,
                "balance_usd": start_equity,
                "unrealized_usd": 0.0,  # No open positions at start
                "equity_sol": state.starting_balance_sol,
                "balance_sol": state.starting_balance_sol,
                "active_value_sol": 0.0,
                "pnl_pct": 0.0,
            }])
            eq_df = pd.concat([start_row, eq_df], ignore_index=True)
        
        # Fill any missing unrealized_usd values
        if "unrealized_usd" not in eq_df.columns:
            eq_df["unrealized_usd"] = 0.0
        eq_df["unrealized_usd"] = eq_df["unrealized_usd"].fillna(0.0)
        
        # Choose which data to show based on selection
        if "Realized" in curve_type:
            y_data = eq_df["balance_usd"]
            chart_label = "Realized"
            # For realized, baseline is starting balance
            baseline = start_equity
        elif "Unrealized" in curve_type:
            y_data = eq_df["unrealized_usd"]
            chart_label = "Unrealized"
            # For unrealized, baseline is 0 (no open positions = no unrealized)
            baseline = 0.0
        else:  # Total
            y_data = eq_df["equity_usd"]
            chart_label = "Total"
            baseline = start_equity
        
        current_value = y_data.iloc[-1]
        
        # Create the chart
        fig = go.Figure()
        
        import numpy as np
        y_values = y_data.values.astype(float)
        x_values = eq_df["time"].values
        
        # Build separate green/red arrays with crossover points included in both
        y_green = []
        y_red = []
        x_plot = []
        
        for i in range(len(y_values)):
            x_plot.append(x_values[i])
            y_val = y_values[i]
            
            if y_val >= baseline:
                y_green.append(y_val)
                y_red.append(np.nan)
            else:
                y_green.append(np.nan)
                y_red.append(y_val)
            
            # Check for crossover to next point - add the threshold point to both traces
            if i < len(y_values) - 1:
                curr = y_values[i]
                next_val = y_values[i + 1]
                # Crossing from above to below or below to above
                if (curr >= baseline and next_val < baseline) or (curr < baseline and next_val >= baseline):
                    # Insert the crossover point (at the threshold line)
                    x_plot.append(x_values[i])  # Use same x (approximate)
                    y_green.append(baseline)
                    y_red.append(baseline)
        
        # Add green line (above start)
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_green,
            mode="lines",
            name="Profit",
            line=dict(color="#00ff88", width=3),
            hovertemplate="$%{y:.2f}<extra></extra>",
            connectgaps=False,
        ))
        
        # Add red line (below start)
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_red,
            mode="lines",
            name="Loss",
            line=dict(color="#ff4444", width=3),
            hovertemplate="$%{y:.2f}<extra></extra>",
            connectgaps=False,
        ))
        
        # Baseline reference line (always visible)
        baseline_label = "$0" if "Unrealized" in curve_type else f"Start: {_fmt_usd(baseline)}"
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="white",
            line_width=2,
            annotation_text=baseline_label,
            annotation_position="right",
            annotation_font_color="white",
        )
        
        # Calculate Y-axis range - ALWAYS include baseline
        y_min = min(y_data.min(), baseline)
        y_max = max(y_data.max(), baseline)
        y_range = y_max - y_min
        if y_range < 20:  # Minimum range of $20
            y_range = 20
        y_padding = y_range * 0.2  # 20% padding
        
        # Style with proper auto-scaling
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                title=f"{chart_label} Equity (USD)",
                range=[y_min - y_padding, y_max + y_padding],
            ),
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats under chart
        if len(eq_df) > 0:
            max_val = y_data.max()
            min_val = y_data.min()
            
            # Change calculation depends on curve type
            if "Unrealized" in curve_type:
                # For unrealized, show as % of starting balance
                change_pct = (current_value / start_equity) * 100 if start_equity > 0 else 0
            else:
                # For total/realized, show change from baseline
                change_pct = ((current_value / baseline) - 1) * 100 if baseline > 0 else 0
            
            chart_cols = st.columns(4)
            chart_cols[0].metric("Current", _fmt_usd(current_value))
            chart_cols[1].metric("High", _fmt_usd(max_val))
            chart_cols[2].metric("Low", _fmt_usd(min_val))
            chart_cols[3].metric("Change", f"{change_pct:+.1f}%")

# =====================================================================
# BOT ACTIONS
# =====================================================================
if result["actions"]:
    st.subheader("🤖 Latest Bot Actions")
    actions_df = pd.DataFrame(result["actions"])
    if "ts" in actions_df.columns:
        actions_df["ts"] = pd.to_datetime(actions_df["ts"], unit="s")
        actions_df = actions_df.sort_values("ts", ascending=False)
    
    # Color code buy/sell
    def style_action(row):
        if row.get("type") == "BUY":
            return ["background-color: rgba(0, 255, 0, 0.1)"] * len(row)
        elif row.get("type") == "SELL":
            pnl = row.get("pnl_sol", 0)
            if pnl > 0:
                return ["background-color: rgba(0, 255, 0, 0.2)"] * len(row)
            else:
                return ["background-color: rgba(255, 0, 0, 0.2)"] * len(row)
        return [""] * len(row)
    
    st.dataframe(actions_df, width="stretch", hide_index=True)

# =====================================================================
# MAIN LAYOUT: LEFT (data) | RIGHT (chart)
# =====================================================================
left, right = st.columns([1.3, 1.0], gap="large")

with left:
    # Token Feed
    st.subheader("🪙 Token Feed (Live from pump.fun)")
    cands = result["candidates"]
    
    # Filter out unknown symbols
    cands = [c for c in cands if c.get("symbol", "").lower() not in ("unknown", "???", "", "null", "none")]
    
    cdf = pd.DataFrame(cands)
    stats = get_last_candidate_stats()
    
    # Show connection status
    pf_connected = stats.get("pumpfun_connected", False)
    pf_count = stats.get("pumpfun", 0)
    dex_count = stats.get("dexscreener", 0)
    pf_stats = stats.get("pumpfun_stats", {})
    
    if pf_connected:
        tokens_received = pf_stats.get("tokens_received", 0)
        if pf_count > 0:
            st.caption(f"🟢 **Live pump.fun stream** | 🔥 {pf_count} bonding curve | 📈 {dex_count} graduated")
        else:
            st.caption(f"🟢 **Live pump.fun stream** | ⏳ Waiting for tokens... ({tokens_received}) | 📈 {dex_count} graduated")
    else:
        # Check if we have boosted tokens (they show as 🔥 even when WebSocket is down)
        pumpfun_tokens = len([c for c in cands if c.get("source") == "pumpfun_ws" or "pump" in str(c.get("address", "")).lower()])
        if pumpfun_tokens > 0:
            st.caption(f"🟡 WebSocket reconnecting... | 🔥 {pumpfun_tokens} pump.fun (via DexScreener) | 📈 {dex_count} graduated")
        else:
            st.caption(f"🟡 Connecting to pump.fun... | 📈 {dex_count} from DexScreener")
    
    if not cdf.empty:
        # Helper to set chart from feed
        def set_feed_chart(address: str):
            st.session_state["selected_chart_addr"] = address
        
        # Header row - now includes momentum indicator
        feed_header = st.columns([0.3, 0.7, 0.7, 0.5, 0.6, 0.6, 0.3])
        feed_header[0].markdown("**Src**")
        feed_header[1].markdown("**Token**")
        feed_header[2].markdown("**MCap**")
        feed_header[3].markdown("**1h%**")  # Momentum indicator
        feed_header[4].markdown("**Vol**")
        feed_header[5].markdown("**Liq**")
        feed_header[6].markdown("**📈**")
        
        st.divider()
        
        # Scrollable container for data rows
        with st.container(height=250):
            for idx, (_, row) in enumerate(cdf.head(20).iterrows()):
                addr = row.get("address", "")
                symbol = row.get("symbol", "???")
                source = row.get("source", "dexscreener")
                # 🔥 for pump.fun (bonding curve or graduated), 📈 for other DEX
                is_pumpfun = source in ("pumpfun_ws", "pumpfun_graduated", "pumpfun") or addr.lower().endswith("pump")
                src_icon = "🔥" if is_pumpfun else "📈"
                mc = row.get("market_cap", 0)
                price_change = row.get("price_change_h1", 0)
                vol = row.get("volume_h1", 0)
                liq = row.get("liquidity_usd", 0)
                
                # Format momentum with color indicator
                if price_change > 0:
                    momentum_str = f"🟢+{price_change:.0f}%"
                elif price_change < 0:
                    momentum_str = f"🔴{price_change:.0f}%"
                else:
                    momentum_str = "—"
                
                feed_row = st.columns([0.3, 0.7, 0.7, 0.5, 0.6, 0.6, 0.3])
                feed_row[0].text(src_icon)
                feed_row[1].text(symbol[:8])
                feed_row[2].text(f"${mc/1000:.0f}K" if mc >= 1000 else f"${mc:.0f}")
                feed_row[3].text(momentum_str)
                feed_row[4].text(f"${vol/1000:.0f}K" if vol >= 1000 else f"${vol:.0f}")
                feed_row[5].text(f"${liq/1000:.0f}K" if liq >= 1000 else f"${liq:.0f}")
                feed_row[6].button("📈", key=f"feed_chart_{idx}", on_click=set_feed_chart, args=(addr,))
        
        st.divider()
    else:
        st.info("🔍 Waiting for new pump.fun tokens... Keep the dashboard running to discover tokens as they launch.")
    
    # Active Trades
    price_time = st.session_state.get("last_price_fetch")
    price_time_str = price_time.strftime("%H:%M:%S") if price_time else "N/A"
    st.subheader(f"📊 Active Trades")
    st.caption(f"💹 Prices updated: {price_time_str}")
    adf = pd.DataFrame(state.active_trades)
    if not adf.empty:
        adf["entry_ts"] = pd.to_datetime(adf["entry_ts"], unit="s")
        
        # Get current prices - fallback to entry price if fetch returns 0
        def get_current_price(row):
            addr = row["address"]
            entry_px = float(row.get("entry_price", 0))
            px = float(current_prices.get(addr, 0))
            if px <= 0:
                px = float(get_token_price(addr))
            if px <= 0:
                px = entry_px  # Fallback to entry price
            return px
        
        adf["current_price"] = adf.apply(get_current_price, axis=1)
        adf["value_sol"] = adf["tokens"] * adf["current_price"]
        adf["pnl_sol"] = adf["value_sol"] - adf["amount_sol"]
        adf["pnl_pct"] = (adf["pnl_sol"] / adf["amount_sol"]) * 100.0
        
        # Convert to USD
        adf["cost_usd"] = adf["amount_sol"] * sol_usd_price
        adf["value_usd"] = adf["value_sol"] * sol_usd_price
        adf["pnl_usd"] = adf["pnl_sol"] * sol_usd_price
        
        # Build custom table with chart button in each row
        # Header row
        header_cols = st.columns([0.4, 1.0, 0.7, 0.6, 0.9, 0.9, 0.6, 0.5, 0.4])
        header_cols[0].markdown("**Src**")
        header_cols[1].markdown("**Time**")
        header_cols[2].markdown("**Token**")
        header_cols[3].markdown("**Cost**")
        header_cols[4].markdown("**Entry**")
        header_cols[5].markdown("**Current**")
        header_cols[6].markdown("**PnL ($)**")
        header_cols[7].markdown("**PnL %**")
        header_cols[8].markdown("**📈**")
        
        st.divider()
        
        # Helper function to set chart selection
        def set_chart_addr(address: str):
            st.session_state["selected_chart_addr"] = address
        
        # Data rows
        for idx, (_, row) in enumerate(adf.iterrows()):
            addr = str(row.get("address", ""))
            symbol = row.get("symbol", "???")
            source = str(row.get("source", "dexscreener"))
            # 🔥 for pump.fun (bonding curve or graduated), 📈 for other DEX
            is_pumpfun = source in ("pumpfun_ws", "pumpfun_graduated", "pumpfun") or addr.lower().endswith("pump")
            src_icon = "🔥" if is_pumpfun else "📈"
            entry_time = row["entry_ts"].strftime("%H:%M:%S") if hasattr(row["entry_ts"], "strftime") else str(row["entry_ts"])
            cost = f"${row['cost_usd']:.2f}"
            entry_px = f"{row['entry_price']:.10f}"
            current_px = f"{row['current_price']:.10f}"
            pnl_usd = f"${row['pnl_usd']:+.2f}"
            pnl_pct = f"{row['pnl_pct']:+.1f}%"
            
            row_cols = st.columns([0.4, 1.0, 0.7, 0.6, 0.9, 0.9, 0.6, 0.5, 0.4])
            row_cols[0].text(src_icon)
            row_cols[1].text(entry_time)
            row_cols[2].text(symbol)
            row_cols[3].text(cost)
            row_cols[4].text(entry_px)
            row_cols[5].text(current_px)
            
            # Color PnL
            if row['pnl_usd'] >= 0:
                row_cols[6].markdown(f":green[{pnl_usd}]")
                row_cols[7].markdown(f":green[{pnl_pct}]")
            else:
                row_cols[6].markdown(f":red[{pnl_usd}]")
                row_cols[7].markdown(f":red[{pnl_pct}]")
            
            # Chart button with callback
            row_cols[8].button(
                "📈", 
                key=f"chart_{addr}_{idx}", 
                help=f"View {symbol} chart",
                on_click=set_chart_addr,
                args=(addr,)
            )
        
        st.divider()
        
        # Manual sell
        sell_addr = st.selectbox(
            "Manual sell:",
            options=[""] + adf["address"].tolist(),
            format_func=lambda a: "Select position..." if a == "" else f"{adf.loc[adf['address']==a,'symbol'].iloc[0]} — {a[:8]}...",
        )
        if sell_addr:
            px = float(get_token_price(sell_addr))
            closed = state.sell_token(sell_addr, px)
            if closed:
                pnl_usd = closed['pnl_sol'] * sol_usd_price
                st.success(f"Sold {closed['symbol']} | PnL: ${pnl_usd:+.2f} ({closed['pnl_pct']:.1f}%)")
                _rerun()
    else:
        st.caption("No active trades. Strategy will buy tokens as they launch.")
    
    # Trade History
    st.subheader("📜 Trade History")
    hdf = pd.DataFrame(state.trade_history)
    if not hdf.empty:
        hdf["entry_ts"] = pd.to_datetime(hdf["entry_ts"], unit="s")
        hdf["exit_ts"] = pd.to_datetime(hdf["exit_ts"], unit="s")
        hdf = hdf.sort_values("exit_ts", ascending=False)
        
        # Convert PnL to USD
        hdf["pnl_usd"] = hdf["pnl_sol"] * sol_usd_price
        hdf["cost_usd"] = hdf["amount_sol"] * sol_usd_price
        
        # Build custom table with chart button in each row
        # Header row
        h_header_cols = st.columns([0.4, 1.0, 0.7, 0.6, 0.9, 0.9, 0.6, 0.5, 0.4])
        h_header_cols[0].markdown("**Src**")
        h_header_cols[1].markdown("**Time**")
        h_header_cols[2].markdown("**Token**")
        h_header_cols[3].markdown("**Cost**")
        h_header_cols[4].markdown("**Entry**")
        h_header_cols[5].markdown("**Exit**")
        h_header_cols[6].markdown("**PnL ($)**")
        h_header_cols[7].markdown("**PnL %**")
        h_header_cols[8].markdown("**📈**")
        
        st.divider()
        
        # Helper function to set chart selection
        def set_hist_chart_addr(address: str):
            st.session_state["selected_chart_addr"] = address
        
        # Data rows in scrollable container (show ALL trades)
        with st.container(height=400):
            for idx, (_, row) in enumerate(hdf.iterrows()):
                addr = str(row.get("address", ""))
                symbol = row.get("symbol", "???")
                source = str(row.get("source", "dexscreener"))
                # 🔥 for pump.fun (bonding curve or graduated), 📈 for other DEX
                is_pumpfun = source in ("pumpfun_ws", "pumpfun_graduated", "pumpfun") or addr.lower().endswith("pump")
                src_icon = "🔥" if is_pumpfun else "📈"
                exit_time = row["exit_ts"].strftime("%H:%M:%S") if hasattr(row["exit_ts"], "strftime") else str(row["exit_ts"])
                cost = f"${row['cost_usd']:.2f}"
                entry_px = f"{row['entry_price']:.10f}"
                exit_px = f"{row['exit_price']:.10f}"
                pnl_usd_val = row['pnl_usd']
                pnl_pct_val = row['pnl_pct']
                pnl_usd = f"${pnl_usd_val:+.2f}"
                pnl_pct = f"{pnl_pct_val:+.1f}%"
                
                h_row_cols = st.columns([0.4, 1.0, 0.7, 0.6, 0.9, 0.9, 0.6, 0.5, 0.4])
                h_row_cols[0].text(src_icon)
                h_row_cols[1].text(exit_time)
                h_row_cols[2].text(symbol)
                h_row_cols[3].text(cost)
                h_row_cols[4].text(entry_px)
                h_row_cols[5].text(exit_px)
                
                # Color PnL
                if pnl_usd_val >= 0:
                    h_row_cols[6].markdown(f":green[{pnl_usd}]")
                    h_row_cols[7].markdown(f":green[{pnl_pct}]")
                else:
                    h_row_cols[6].markdown(f":red[{pnl_usd}]")
                    h_row_cols[7].markdown(f":red[{pnl_pct}]")
                
                # Chart button with callback
                h_row_cols[8].button(
                    "📈", 
                    key=f"hist_chart_{addr}_{idx}", 
                    help=f"View {symbol} chart",
                    on_click=set_hist_chart_addr,
                    args=(addr,)
                )
        
        st.divider()
    else:
        st.caption("No trades completed yet.")

with right:
    st.subheader("📈 Token Chart")
    
    # Helper to check if symbol is valid (not unknown)
    def is_valid_symbol(symbol: str) -> bool:
        if not symbol:
            return False
        return symbol.lower() not in ("unknown", "???", "", "null", "none")
    
    # Build a combined list of all tokens we can chart
    all_chart_tokens = {}  # address -> {"symbol": str, "source": str}
    
    # Add active trades (highest priority)
    for t in state.active_trades:
        addr = t.get("address", "")
        symbol = t.get("symbol", "???")
        if addr and is_valid_symbol(symbol):
            all_chart_tokens[addr] = {"symbol": symbol, "source": "🟢 Active"}
    
    # Add trade history
    for t in state.trade_history:
        addr = t.get("address", "")
        symbol = t.get("symbol", "???")
        if addr and addr not in all_chart_tokens and is_valid_symbol(symbol):
            all_chart_tokens[addr] = {"symbol": symbol, "source": "📜 History"}
    
    # Add candidates
    for c in result["candidates"][:20]:  # Limit to top 20 candidates
        addr = c.get("address", "")
        symbol = c.get("symbol", "???")
        if addr and addr not in all_chart_tokens and is_valid_symbol(symbol):
            all_chart_tokens[addr] = {"symbol": symbol, "source": "🔍 Feed"}
    
    chart_addr = ""
    chart_symbol = ""
    
    # Get saved chart selection from session state FIRST
    saved_addr = st.session_state.get("selected_chart_addr", "")
    
    # If saved addr is not in our token list, add it
    if saved_addr and saved_addr not in all_chart_tokens:
        # Try to get symbol from active trades or history
        symbol_found = "Selected"
        for t in state.active_trades:
            if t.get("address") == saved_addr:
                symbol_found = t.get("symbol", "Selected")
                break
        else:
            for t in state.trade_history:
                if t.get("address") == saved_addr:
                    symbol_found = t.get("symbol", "Selected")
                    break
        all_chart_tokens[saved_addr] = {"symbol": symbol_found, "source": "📌 Selected"}
    
    if all_chart_tokens:
        # Create options list
        options = list(all_chart_tokens.keys())
        
        def format_option(addr):
            info = all_chart_tokens.get(addr, {})
            symbol = info.get("symbol", addr[:8])
            source = info.get("source", "")
            return f"{source} {symbol}"
        
        # Find the index of the saved address, or default to 0
        default_index = 0
        if saved_addr and saved_addr in options:
            default_index = options.index(saved_addr)
        
        # Use on_change callback to update session state
        def on_chart_change():
            st.session_state["selected_chart_addr"] = st.session_state["chart_selector"]
        
        chart_addr = st.selectbox(
            "Select token:",
            options=options,
            index=default_index,
            format_func=format_option,
            key="chart_selector",
            on_change=on_chart_change,
        )
        
        # If we have a saved addr that's valid, use it (button click takes priority)
        if saved_addr and saved_addr in options:
            chart_addr = saved_addr
        
        chart_symbol = all_chart_tokens.get(chart_addr, {}).get("symbol", "")
    
    if chart_addr:
        st.caption(f"**{chart_symbol}** — `{chart_addr[:20]}...`")
        
        # Try DexScreener embed first (works for graduated tokens)
        use_dex_embed = st.toggle("Use DexScreener Chart (if available)", value=True)
        
        if use_dex_embed:
            url = get_dexscreener_embed_url(chart_addr)
            if url:
                components.iframe(url, height=450, scrolling=False)
            else:
                st.info("Token not yet on DexScreener (still on bonding curve). Using price samples.")
                use_dex_embed = False
        
        if not use_dex_embed:
            # Show our own chart from price samples
            df = get_chart_data(chart_addr)
            if df.empty:
                st.info("Collecting price data... Chart will appear after a few refreshes.")
            else:
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=df["timestamp"],
                            open=df["open"],
                            high=df["high"],
                            low=df["low"],
                            close=df["close"],
                            increasing_line_color="#00ff00",
                            decreasing_line_color="#ff0000",
                        )
                    ]
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=400,
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig, width="stretch")
        
        # Price info
        sample_stats = get_price_sample_stats(chart_addr)
        current_price = get_token_price(chart_addr)
        
        st.caption(f"Current price: **{current_price:.10f} SOL** | Samples: {sample_stats.get('count', 0)}")
        
        # Quick links
        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🌐 View on Pump.fun", f"https://pump.fun/{chart_addr}")
        with col2:
            st.link_button("📊 View on DexScreener", f"https://dexscreener.com/solana/{chart_addr}")
    else:
        st.info("🔍 Waiting for tokens... Keep the dashboard running to discover new launches.")


# =====================================================================
# AUTO-REFRESH
# =====================================================================
if auto_refresh:
    time.sleep(refresh_seconds)
    _rerun()
