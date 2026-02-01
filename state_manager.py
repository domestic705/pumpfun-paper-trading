from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ActiveTrade:
    symbol: str
    address: str
    entry_price: float  # in SOL per token (simulated)
    amount_sol: float
    tokens: float
    entry_ts: float
    source: str = "dexscreener"  # "pumpfun_ws", "pumpfun", or "dexscreener"
    peak_price: float = 0.0      # Highest price seen (for trailing stop)


@dataclass
class ClosedTrade:
    symbol: str
    address: str
    entry_price: float
    exit_price: float
    amount_sol: float
    tokens: float
    entry_ts: float
    exit_ts: float
    pnl_sol: float
    pnl_pct: float
    source: str = "dexscreener"  # "pumpfun_ws" or "dexscreener"


class StateManager:
    """
    Persists paper trading state to a local JSON file.

    Attributes:
      - virtual_balance: float (SOL)
      - active_trades: list[dict]
      - trade_history: list[dict]
      - cooldowns: dict[address -> expiry_timestamp] - prevents re-buying too quickly
    """
    
    # Cooldown period after selling before we can buy the same token again
    COOLDOWN_SECONDS = 15 * 60  # 15 minutes

    def __init__(self, path: str = "paper_data.json", starting_balance_sol: float = 10.0):
        self.path = path
        self.starting_balance_sol = float(starting_balance_sol)

        self.virtual_balance: float = self.starting_balance_sol
        self.active_trades: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.cooldowns: Dict[str, float] = {}  # address -> expiry timestamp
        self.reset_timestamp: float = time.time()  # When account was created/reset
        self.total_commissions: float = 0.0  # Total fees paid (in SOL)
        self.equity_history: List[Dict[str, Any]] = []  # Equity snapshots over time

        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._save()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # If the file is corrupted, start fresh (but keep a backup).
            try:
                os.replace(self.path, f"{self.path}.corrupt.{int(time.time())}.bak")
            except Exception:
                pass
            self.virtual_balance = self.starting_balance_sol
            self.active_trades = []
            self.trade_history = []
            self._save()
            return

        self.virtual_balance = float(data.get("virtual_balance", self.starting_balance_sol))
        self.active_trades = list(data.get("active_trades", []))
        self.trade_history = list(data.get("trade_history", []))
        self.cooldowns = dict(data.get("cooldowns", {}))
        self.reset_timestamp = float(data.get("reset_timestamp", time.time()))
        self.total_commissions = float(data.get("total_commissions", 0.0))
        self.equity_history = list(data.get("equity_history", []))

    def _save(self) -> None:
        """Save state to file. Silently fails on read-only filesystems (like Streamlit Cloud)."""
        try:
            tmp_path = f"{self.path}.tmp"
            payload = {
                "virtual_balance": self.virtual_balance,
                "active_trades": self.active_trades,
                "trade_history": self.trade_history,
                "cooldowns": self.cooldowns,
                "reset_timestamp": self.reset_timestamp,
                "total_commissions": self.total_commissions,
                "equity_history": self.equity_history[-500:],  # Keep last 500 snapshots
                "saved_at": time.time(),
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp_path, self.path)
        except (OSError, IOError):
            # Silently fail on read-only filesystems (Streamlit Cloud)
            pass

    def _find_active_trade_idx(self, address: str) -> Optional[int]:
        for i, t in enumerate(self.active_trades):
            if t.get("address") == address:
                return i
        return None

    def is_owned(self, address: str) -> bool:
        return self._find_active_trade_idx(address) is not None
    
    def is_on_cooldown(self, address: str) -> bool:
        """Check if a token is on cooldown (recently sold)."""
        expiry = self.cooldowns.get(address, 0)
        if expiry > time.time():
            return True
        # Clean up expired cooldown
        if address in self.cooldowns:
            del self.cooldowns[address]
        return False
    
    def can_buy(self, address: str) -> bool:
        """Check if we can buy this token (not owned and not on cooldown)."""
        return not self.is_owned(address) and not self.is_on_cooldown(address)

    def buy_token(
        self, 
        symbol: str, 
        address: str, 
        price: float, 
        amount_sol: float, 
        source: str = "dexscreener",
        commission_pct: float = 0.0,  # Commission as percentage (e.g. 1.0 = 1%)
    ) -> bool:
        price = float(price)
        amount_sol = float(amount_sol)
        if price <= 0 or amount_sol <= 0:
            return False

        # Check if we can buy (not owned AND not on cooldown)
        if not self.can_buy(address):
            return False

        # Calculate commission
        commission = amount_sol * (commission_pct / 100.0)
        total_cost = amount_sol + commission
        
        if total_cost > self.virtual_balance + 1e-12:
            return False

        tokens = amount_sol / price
        trade = ActiveTrade(
            symbol=symbol,
            address=address,
            entry_price=price,
            amount_sol=amount_sol,
            tokens=tokens,
            entry_ts=time.time(),
            source=source,
            peak_price=price,  # Initialize peak at entry
        )

        self.virtual_balance -= total_cost
        self.total_commissions += commission
        self.active_trades.append(asdict(trade))
        self._save()
        return True
    
    def update_peak_price(self, address: str, current_price: float) -> None:
        """Update peak price for trailing stop calculation."""
        idx = self._find_active_trade_idx(address)
        if idx is None:
            return
        
        current_peak = float(self.active_trades[idx].get("peak_price", 0))
        entry_price = float(self.active_trades[idx].get("entry_price", 0))
        
        # Initialize peak_price if not set (for old trades)
        if current_peak <= 0:
            current_peak = entry_price
        
        # Update if new high
        if current_price > current_peak:
            self.active_trades[idx]["peak_price"] = current_price
            self._save()

    def sell_token(self, address: str, price: float, commission_pct: float = 0.0) -> Optional[Dict[str, Any]]:
        price = float(price)
        if price <= 0:
            return None

        idx = self._find_active_trade_idx(address)
        if idx is None:
            return None

        t = self.active_trades.pop(idx)
        tokens = float(t["tokens"])
        gross_proceeds = tokens * price
        entry_cost = float(t["amount_sol"])
        
        # Calculate sell commission
        commission = gross_proceeds * (commission_pct / 100.0)
        net_proceeds = gross_proceeds - commission
        self.total_commissions += commission

        pnl_sol = net_proceeds - entry_cost
        pnl_pct = (pnl_sol / entry_cost) * 100.0 if entry_cost > 0 else 0.0

        closed = ClosedTrade(
            symbol=str(t.get("symbol", "")),
            address=str(t.get("address", "")),
            entry_price=float(t["entry_price"]),
            exit_price=price,
            amount_sol=entry_cost,
            tokens=tokens,
            entry_ts=float(t["entry_ts"]),
            exit_ts=time.time(),
            pnl_sol=pnl_sol,
            pnl_pct=pnl_pct,
            source=str(t.get("source", "dexscreener")),
        )

        self.virtual_balance += net_proceeds
        self.trade_history.append(asdict(closed))
        
        # Set cooldown to prevent re-buying this token too quickly
        self.cooldowns[address] = time.time() + self.COOLDOWN_SECONDS
        
        self._save()
        return asdict(closed)

    def get_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates unrealized PnL for active trades.
        current_prices: {address: price}
        Returns dict with summary fields in SOL:
          - unrealized_pnl_sol
          - active_value_sol
          - active_cost_sol
          - equity_sol (balance + active_value)
        """
        active_value = 0.0
        active_cost = 0.0
        for t in self.active_trades:
            addr = t.get("address")
            if not addr:
                continue
            price = float(current_prices.get(addr, float(t.get("entry_price", 0.0)) or 0.0))
            tokens = float(t.get("tokens", 0.0))
            active_value += tokens * price
            active_cost += float(t.get("amount_sol", 0.0))

        unrealized = active_value - active_cost
        equity = self.virtual_balance + active_value
        return {
            "unrealized_pnl_sol": unrealized,
            "active_value_sol": active_value,
            "active_cost_sol": active_cost,
            "equity_sol": equity,
        }
    
    def record_equity_snapshot(self, current_prices: Dict[str, float]) -> None:
        """Record an equity snapshot for the equity curve."""
        pnl = self.get_pnl(current_prices)
        equity_sol = pnl["equity_sol"]
        
        snapshot = {
            "ts": time.time(),
            "equity_sol": equity_sol,
            "balance_sol": self.virtual_balance,
            "active_value_sol": pnl["active_value_sol"],
            "trades_count": len(self.trade_history),
        }
        
        # Only record if enough time has passed (10 seconds) or equity changed
        if self.equity_history:
            last = self.equity_history[-1]
            time_diff = snapshot["ts"] - last["ts"]
            equity_diff = abs(snapshot["equity_sol"] - last["equity_sol"])
            
            # Record if 10+ seconds passed OR equity changed by 0.01+ SOL
            if time_diff < 10 and equity_diff < 0.01:
                return
        
        self.equity_history.append(snapshot)
        
        # Trim old entries (keep last 500)
        if len(self.equity_history) > 500:
            self.equity_history = self.equity_history[-500:]
        
        self._save()

    def reset(self) -> None:
        self.virtual_balance = self.starting_balance_sol
        self.active_trades = []
        self.trade_history = []
        self.cooldowns = {}
        self.total_commissions = 0.0
        self.reset_timestamp = time.time()
        self.equity_history = [{"ts": time.time(), "equity_sol": self.starting_balance_sol, "balance_sol": self.starting_balance_sol, "active_value_sol": 0, "trades_count": 0}]
        self._save()
