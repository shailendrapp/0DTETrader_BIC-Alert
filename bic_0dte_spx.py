"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        BREAKEVEN IRON CONDOR (BIC) — 0DTE SPX AUTOMATED ALERT SYSTEM       ║
║                    v3.1 | GitHub Actions Native                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v3.1 additions:                                                             ║
║    ✅ --monitor mode: leg breach detection every ~12 min                     ║
║    ✅ Live delta/gamma fetched from Tradier (not entry delta)                ║
║    ✅ Telegram retry with exponential backoff (3 attempts)                   ║
║    ✅ Pipeline error alerts — silent API failures now surfaced               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Run modes:                                                                  ║
║    python bic_0dte_spx.py --morning      (06:30 PST cron)                   ║
║    python bic_0dte_spx.py --entry 1      (entry #1 — 07:35 PST)             ║
║    python bic_0dte_spx.py --entry 2      (entry #2 — 08:35 PST)             ║
║    python bic_0dte_spx.py --entry 3      (entry #3 — 09:35 PST)             ║
║    python bic_0dte_spx.py --entry 4      (entry #4 — 10:35 PST)             ║
║    python bic_0dte_spx.py --entry 5      (entry #5 — 11:35 PST)             ║
║    python bic_0dte_spx.py --exit         (14:30 PST cron)                   ║
║    python bic_0dte_spx.py --monitor      (every 12 min — leg breach watch)  ║
║    python bic_0dte_spx.py --test         (manual test, bypasses time guards) ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, math, logging, argparse, time
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

# ── Third-party (all in requirements.txt) ─────────────────────────────────────
import yfinance as yf
import numpy as np
import requests
import anthropic
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("BIC")

ET  = ZoneInfo("America/New_York")
PST = ZoneInfo("America/Los_Angeles")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # All keys read from environment variables — set as GitHub Secrets
    TRADIER_KEY      = "".join(os.getenv("TRADIER_API_KEY",    "").split())
    FLASHALPHA_KEY   = "".join(os.getenv("FLASHALPHA_API_KEY", "").split())
    ANTHROPIC_KEY    = "".join(os.getenv("ANTHROPIC_API_KEY",  "").split())
    TELEGRAM_TOKEN   = "".join(os.getenv("TELEGRAM_BOT_TOKEN", "").split())
    TELEGRAM_CHAT_ID = "".join(os.getenv("TELEGRAM_CHAT_ID",   "").split())

    # Strategy parameters
    TARGET_DELTA_MIN  = 0.05
    TARGET_DELTA_MAX  = 0.10
    WING_WIDTH_MIN    = 25
    WING_WIDTH_MAX    = 35
    MIN_CREDIT_SIDE   = 50
    STOP_BUFFER       = 5       # total credit + $5 = stop per side
    PROFIT_TAKE_PCT   = 0.50

    # Risk controls
    VIX_SKIP_ABOVE    = 30
    VIX_CAUTION_ABOVE = 22

    # Monitor — breach alert thresholds
    BREACH_DELTA      = 0.30    # alert when live |delta| approaches this
    BREACH_WARN_DELTA = 0.20    # early warning level
    SPX_1MIN_MOVE     = 0.20    # conservative SPX pts/min for time estimate

    # API endpoints
    TRADIER_BASE    = "https://api.tradier.com/v1"
    FLASHALPHA_BASE = "https://api.flashalpha.com/v1"
    RISK_FREE_RATE  = 0.05

cfg = Config()


# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS  — all comparisons done in ET; GA runner is always UTC
# ─────────────────────────────────────────────────────────────────────────────
def now_et() -> datetime:
    return datetime.now(ET)

def now_pst() -> datetime:
    return datetime.now(PST)

def is_market_open() -> bool:
    n = now_et()
    if n.weekday() >= 5:
        return False
    return n.replace(hour=9, minute=30, second=0, microsecond=0) <= n <= \
           n.replace(hour=16, minute=15, second=0, microsecond=0)

def is_entry_window() -> bool:
    """Valid BIC entry: 9:35 AM ET – 2:30 PM ET"""
    n = now_et()
    return n.replace(hour=9, minute=35, second=0, microsecond=0) <= n <= \
           n.replace(hour=14, minute=30, second=0, microsecond=0)

def is_monitor_window() -> bool:
    """Monitor active: 9:35 AM ET – 2:30 PM ET (same as entry window)"""
    return is_entry_window()


# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────────────────────────────────────
class MarketData:

    def get_spx_vix(self) -> dict:
        try:
            spx_info = yf.Ticker("^GSPC").fast_info
            vix_info = yf.Ticker("^VIX").fast_info
            spx_px   = round(spx_info.last_price, 2)
            prev     = spx_info.previous_close
            return {
                "spx":            spx_px,
                "vix":            round(vix_info.last_price, 2),
                "spx_change_pct": round((spx_px - prev) / prev * 100, 2),
                "timestamp":      now_et().strftime("%H:%M:%S ET"),
            }
        except Exception as e:
            log.error(f"SPX/VIX fetch error: {e}")
            return {"spx": None, "vix": None, "error": str(e)}

    def get_today_expiry(self) -> str:
        today = date.today()
        if today.weekday() >= 5:
            today += timedelta(days=7 - today.weekday())
        return today.strftime("%Y-%m-%d")

    def get_options_chain(self, expiry: str) -> Optional[list]:
        if not cfg.TRADIER_KEY:
            log.warning("No TRADIER_API_KEY — using Black-Scholes fallback")
            return None
        try:
            r = requests.get(
                f"{cfg.TRADIER_BASE}/markets/options/chains",
                headers={"Authorization": f"Bearer {cfg.TRADIER_KEY}",
                         "Accept": "application/json"},
                params={"symbol": "SPX", "expiration": expiry, "greeks": "true"},
                timeout=12,
            )
            r.raise_for_status()
            opts = r.json().get("options", {}).get("option", [])
            log.info(f"Tradier: {len(opts)} contracts for {expiry}")
            return opts or None
        except requests.HTTPError as e:
            code = e.response.status_code
            log.error(f"Tradier HTTP {code} — {'bad key' if code==401 else 'no RT access' if code==403 else str(e)}")
            return None
        except Exception as e:
            log.error(f"Tradier error: {e}")
            return None

    def get_open_positions(self) -> list:
        """
        Fetch open SPX 0DTE option positions from Tradier.
        Returns list of dicts with symbol, option_type, strike, quantity, cost_basis.
        Only returns today's expiry positions.
        """
        if not cfg.TRADIER_KEY:
            log.warning("No TRADIER_API_KEY — cannot fetch positions for monitor")
            return []
        try:
            r = requests.get(
                f"{cfg.TRADIER_BASE}/accounts/{self._get_account_id()}/positions",
                headers={"Authorization": f"Bearer {cfg.TRADIER_KEY}",
                         "Accept": "application/json"},
                timeout=12,
            )
            r.raise_for_status()
            raw = r.json().get("positions", {}).get("position", [])
            if isinstance(raw, dict):
                raw = [raw]  # single position comes back as dict, not list

            today_exp = self.get_today_expiry().replace("-", "")  # YYYYMMDD
            positions = []
            for p in raw:
                sym = p.get("symbol", "")
                # SPX option symbols contain the expiry date: SPXYYYYMMDD
                if "SPX" in sym and today_exp in sym:
                    qty = p.get("quantity", 0)
                    if qty == 0:
                        continue
                    positions.append({
                        "symbol":      sym,
                        "quantity":    qty,
                        "cost_basis":  p.get("cost_basis", 0),
                    })
            log.info(f"Open 0DTE positions: {len(positions)}")
            return positions
        except Exception as e:
            log.error(f"Position fetch error: {e}")
            return []

    def _get_account_id(self) -> str:
        """Fetch Tradier account ID dynamically."""
        try:
            r = requests.get(
                f"{cfg.TRADIER_BASE}/user/profile",
                headers={"Authorization": f"Bearer {cfg.TRADIER_KEY}",
                         "Accept": "application/json"},
                timeout=10,
            )
            r.raise_for_status()
            accounts = r.json().get("profile", {}).get("account", [])
            if isinstance(accounts, dict):
                accounts = [accounts]
            return str(accounts[0].get("account_number", ""))
        except Exception as e:
            log.error(f"Account ID fetch error: {e}")
            return ""

    def get_live_greeks(self, symbol: str, expiry: str) -> Optional[dict]:
        """
        Fetch live Greeks for a specific option symbol from Tradier.
        Returns dict with delta, gamma, theta, vega, iv — or None on failure.
        """
        if not cfg.TRADIER_KEY:
            return None
        try:
            r = requests.get(
                f"{cfg.TRADIER_BASE}/markets/options/chains",
                headers={"Authorization": f"Bearer {cfg.TRADIER_KEY}",
                         "Accept": "application/json"},
                params={"symbol": "SPX", "expiration": expiry, "greeks": "true"},
                timeout=12,
            )
            r.raise_for_status()
            opts = r.json().get("options", {}).get("option", [])
            for opt in opts:
                if opt.get("symbol") == symbol:
                    g = opt.get("greeks") or {}
                    return {
                        "delta": float(g.get("delta") or 0),
                        "gamma": float(g.get("gamma") or 0),
                        "theta": float(g.get("theta") or 0),
                        "vega":  float(g.get("vega")  or 0),
                        "iv":    float(g.get("mid_iv") or g.get("smv_vol") or 0),
                        "bid":   float(opt.get("bid")  or 0),
                        "ask":   float(opt.get("ask")  or 0),
                    }
            log.warning(f"Symbol {symbol} not found in chain")
            return None
        except Exception as e:
            log.error(f"Greeks fetch error for {symbol}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# GEX DATA
# ─────────────────────────────────────────────────────────────────────────────
class GEXData:

    def get(self, vix_fallback: float = 20.0) -> dict:
        if not cfg.FLASHALPHA_KEY:
            log.warning("No FLASHALPHA_API_KEY — VIX fallback")
            return self._vix_fallback(vix_fallback)
        try:
            r = requests.get(
                f"{cfg.FLASHALPHA_BASE}/exposure/gex",
                headers={"Authorization": f"Bearer {cfg.FLASHALPHA_KEY}"},
                params={"symbol": "SPX"},
                timeout=10,
            )
            r.raise_for_status()
            d   = r.json()
            gex = d.get("net_gex", d.get("gex", 0))
            return {
                "gex_value":  gex,
                "gamma_flip": d.get("gamma_flip"),
                "call_wall":  d.get("call_wall"),
                "put_wall":   d.get("put_wall"),
                "regime":     self._regime(gex),
                "source":     "FlashAlpha",
            }
        except requests.HTTPError as e:
            log.warning(f"FlashAlpha HTTP {e.response.status_code} — VIX fallback")
            return self._vix_fallback(vix_fallback)
        except Exception as e:
            log.error(f"FlashAlpha error: {e}")
            return self._vix_fallback(vix_fallback)

    def _regime(self, gex: float) -> str:
        if gex >= 0:       return "GO"
        if gex >= -500e6:  return "CAUTION"
        return "SKIP"

    def _vix_fallback(self, vix: float) -> dict:
        regime = "GO" if vix < 18 else ("CAUTION" if vix < 25 else "SKIP")
        return {
            "gex_value": None, "gamma_flip": None,
            "call_wall": None, "put_wall":   None,
            "regime":    regime,
            "source":    f"VIX fallback (VIX={vix:.1f})",
        }


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_delta(S, K, T, r, sigma, opt) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return float(norm.cdf(d1) if opt == "call" else norm.cdf(d1) - 1)

def bs_price(S, K, T, r, sigma, opt) -> float:
    if T <= 0:
        return max(0.0, S - K if opt == "call" else K - S)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def time_to_expiry_years() -> float:
    n = now_et()
    close = n.replace(hour=16, minute=15, second=0, microsecond=0)
    return max(0.0, (close - n).total_seconds()) / (365 * 24 * 3600)


# ─────────────────────────────────────────────────────────────────────────────
# BIC LEG SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
class BICSelector:

    def find_best_legs(self, options: list, spx: float,
                       vix: float, regime: str) -> Optional[dict]:
        T     = time_to_expiry_years()
        sigma = vix / 100
        r     = cfg.RISK_FREE_RATE

        if T <= 0:
            log.warning("Time to expiry is zero — no trade possible (outside market hours?)")
            return None

        wing = cfg.WING_WIDTH_MAX if vix > cfg.VIX_CAUTION_ABOVE else cfg.WING_WIDTH_MIN

        put_leg  = self._best_leg(options, spx, T, sigma, r, "put",  wing)
        call_leg = self._best_leg(options, spx, T, sigma, r, "call", wing)

        if not put_leg or not call_leg:
            return None

        put_credit  = round((put_leg["short_mid"]  - put_leg["long_mid"])  * 100)
        call_credit = round((call_leg["short_mid"] - call_leg["long_mid"]) * 100)
        total       = put_credit + call_credit

        if total < cfg.MIN_CREDIT_SIDE * 2:
            log.warning(f"Total credit ${total} below minimum — skip")
            return None

        stop = total + cfg.STOP_BUFFER

        return {
            "put_short":     put_leg["short_strike"],
            "put_long":      put_leg["long_strike"],
            "put_delta":     put_leg["delta"],
            "put_credit":    put_credit,
            "call_short":    call_leg["short_strike"],
            "call_long":     call_leg["long_strike"],
            "call_delta":    call_leg["delta"],
            "call_credit":   call_credit,
            "total_credit":  total,
            "stop_per_side": stop,
            "max_loss":      stop * 2,
            "profit_target": round(total * cfg.PROFIT_TAKE_PCT),
            "bp_required":   round(max(
                (put_leg["short_strike"]  - put_leg["long_strike"])  * 100,
                (call_leg["long_strike"] - call_leg["short_strike"]) * 100,
            )),
            "defended_low":  put_leg["short_strike"],
            "defended_high": call_leg["short_strike"],
            "range_pts":     call_leg["short_strike"] - put_leg["short_strike"],
            "T_hours":       round(T * 365 * 24, 1),
            "sigma_pct":     round(sigma * 100, 1),
            "imbalance_pct": round(abs(put_credit - call_credit) / total * 100, 1),
            "wing_used":     wing,
            "vix_regime":    "HIGH" if vix > cfg.VIX_CAUTION_ABOVE else "NORMAL",
        }

    def _best_leg(self, options, spx, T, sigma, r, opt_type, wing):
        result = self._from_chain(options, spx, T, sigma, r, opt_type, wing)
        return result or self._from_bs(spx, T, sigma, r, opt_type, wing)

    def _from_chain(self, options, spx, T, sigma, r, opt_type, wing):
        candidates = []
        for opt in options:
            if opt.get("option_type", "").lower() != opt_type:
                continue
            strike = float(opt.get("strike", 0))
            if strike <= 0:
                continue
            greeks = opt.get("greeks") or {}
            raw_d  = greeks.get("delta")
            delta  = float(raw_d) if raw_d is not None \
                     else bs_delta(spx, strike, T, r, sigma, opt_type)
            abs_d  = abs(delta)
            if not (cfg.TARGET_DELTA_MIN <= abs_d <= cfg.TARGET_DELTA_MAX):
                continue
            bid = float(opt.get("bid") or 0)
            ask = float(opt.get("ask") or 0)
            if bid <= 0 or ask <= 0:
                continue
            mid    = (bid + ask) / 2
            long_k = strike - wing if opt_type == "put" else strike + wing
            lmid   = self._long_price(options, long_k, opt_type, spx, T, sigma, r)
            if (mid - lmid) * 100 >= cfg.MIN_CREDIT_SIDE:
                candidates.append({
                    "short_strike": strike, "long_strike": long_k,
                    "delta": round(abs_d, 3),
                    "short_mid": round(mid, 2), "long_mid": round(lmid, 2),
                })
        if not candidates:
            return None
        candidates.sort(key=lambda x: abs(x["delta"] - 0.07))
        return candidates[0]

    def _long_price(self, options, long_k, opt_type, spx, T, sigma, r):
        for opt in options:
            if opt.get("option_type", "").lower() == opt_type:
                if abs(float(opt.get("strike", -9999)) - long_k) < 0.5:
                    bid = float(opt.get("bid") or 0)
                    ask = float(opt.get("ask") or 0)
                    if bid > 0 and ask > 0:
                        return round((bid + ask) / 2, 2)
        return round(bs_price(spx, long_k, T, r, sigma, opt_type), 2)

    def _from_bs(self, spx, T, sigma, r, opt_type, wing):
        for offset in range(10, 400, 5):
            if opt_type == "put":
                sk = round((spx - offset) / 5) * 5
                lk = sk - wing
            else:
                sk = round((spx + offset) / 5) * 5
                lk = sk + wing
            d  = abs(bs_delta(spx, sk, T, r, sigma, opt_type))
            if cfg.TARGET_DELTA_MIN <= d <= cfg.TARGET_DELTA_MAX:
                sp = bs_price(spx, sk, T, r, sigma, opt_type)
                lp = bs_price(spx, lk, T, r, sigma, opt_type)
                if (sp - lp) * 100 >= cfg.MIN_CREDIT_SIDE:
                    return {"short_strike": sk, "long_strike": lk,
                            "delta": round(d, 3),
                            "short_mid": round(sp, 2), "long_mid": round(lp, 2)}
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE HAIKU — AI ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
class ClaudeAnalyst:

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_KEY) \
                      if cfg.ANTHROPIC_KEY else None

    def premarket_brief(self, market: dict, gex: dict) -> str:
        prompt = (
            "You are a 0DTE SPX options trader. Give a concise 3-line morning brief "
            "for the Breakeven Iron Condor strategy.\n\n"
            f"SPX: {market.get('spx')} ({market.get('spx_change_pct',0):+.2f}%)\n"
            f"VIX: {market.get('vix')}\n"
            f"GEX regime: {gex.get('regime')} | source: {gex.get('source')}\n"
            f"Gamma flip: {gex.get('gamma_flip','N/A')} | "
            f"Call wall: {gex.get('call_wall','N/A')} | Put wall: {gex.get('put_wall','N/A')}\n\n"
            "Line 1: Overall market character today.\n"
            "Line 2: What GEX/VIX setup means for BIC.\n"
            "Line 3: GO / CAUTION / SKIP directive with one-line reason."
        )
        return self._call(prompt, 160)

    def analyze_trade(self, market: dict, gex: dict, trade: dict, entry_num: int) -> str:
        stop_c = trade["stop_per_side"] / 100
        prompt = (
            f"Expert 0DTE SPX trader reviewing BIC Entry #{entry_num}.\n\n"
            f"MARKET ({now_et().strftime('%H:%M ET')}): "
            f"SPX {market.get('spx')} ({market.get('spx_change_pct',0):+.2f}%) "
            f"VIX {market.get('vix')} | GEX {gex.get('regime')} | {gex.get('source')}\n"
            f"Gamma flip: {gex.get('gamma_flip','N/A')} | "
            f"Walls: {gex.get('call_wall','N/A')} / {gex.get('put_wall','N/A')}\n\n"
            f"TRADE:\n"
            f"  PUT:  Sell {trade['put_short']}P / Buy {trade['put_long']}P "
            f"Δ{trade['put_delta']} ${trade['put_credit']} credit\n"
            f"  CALL: Sell {trade['call_short']}C / Buy {trade['call_long']}C "
            f"Δ{trade['call_delta']} ${trade['call_credit']} credit\n"
            f"  Total ${trade['total_credit']} | Stop ${trade['stop_per_side']} per side "
            f"(${stop_c:.2f}/contract) | Target ${trade['profit_target']}\n"
            f"  Range {trade['defended_low']}–{trade['defended_high']} ({trade['range_pts']} pts) "
            f"| Imbalance {trade['imbalance_pct']}% | {trade['T_hours']}h left | "
            f"Wings {trade['wing_used']}pt\n\n"
            "Reply in EXACTLY this format:\n"
            "VERDICT: [GO ✅ | WAIT ⏳ | SKIP ❌]\n"
            "REASONING: [2 sentences]\n"
            "ADJUSTMENTS: [specific tweaks or None needed]\n"
            "RISK NOTE: [one specific risk right now]"
        )
        return self._call(prompt, 200)

    def _call(self, prompt: str, max_tokens: int) -> str:
        if not self.client:
            return ("VERDICT: WAIT ⏳\nREASONING: ANTHROPIC_API_KEY not set.\n"
                    "ADJUSTMENTS: Add the secret in GitHub repo settings.\n"
                    "RISK NOTE: Trade manually using GEX regime only.")
        try:
            msg = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            log.error(f"Claude error: {e}")
            return (f"VERDICT: WAIT ⏳\nREASONING: Claude API error — {e}\n"
                    "ADJUSTMENTS: Check ANTHROPIC_API_KEY.\n"
                    "RISK NOTE: No AI analysis — use GEX regime rules.")


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM  — with retry + backoff
# ─────────────────────────────────────────────────────────────────────────────
class Telegram:

    def send(self, text: str, retries: int = 3) -> bool:
        if not cfg.TELEGRAM_TOKEN or not cfg.TELEGRAM_CHAT_ID:
            log.warning("Telegram not configured — printing to stdout only")
            print("\n" + "="*60 + "\n" + text + "\n" + "="*60)
            return False
        url = f"https://api.telegram.org/bot{cfg.TELEGRAM_TOKEN}/sendMessage"
        for attempt in range(retries):
            try:
                r = requests.post(
                    url,
                    json={"chat_id": cfg.TELEGRAM_CHAT_ID, "text": text,
                          "parse_mode": "HTML", "disable_web_page_preview": True},
                    timeout=15,
                )
                if r.status_code == 200:
                    log.info(f"Telegram: sent {len(text)} chars")
                    return True
                if r.status_code == 429:
                    wait = 2 ** attempt
                    log.warning(f"Telegram rate limited — retry in {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
            except Exception as e:
                log.error(f"Telegram attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        # All retries exhausted — at minimum print so GitHub Actions log captures it
        log.error(f"Telegram FAILED after {retries} attempts")
        print(text)
        return False

    def morning_msg(self, market: dict, gex: dict, brief: str) -> str:
        re = {"GO": "🟢", "CAUTION": "🟡", "SKIP": "🔴"}.get(gex.get("regime","?"), "⚪")
        lines = [
            f"<b>☀️ BIC MORNING SCAN — {now_pst().strftime('%H:%M')} PST</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            f"  SPX <b>{market.get('spx','?')}</b>  ({market.get('spx_change_pct',0):+.2f}%)    "
            f"VIX <b>{market.get('vix','?')}</b>",
            f"  GEX {re} <b>{gex.get('regime','?')}</b>  |  {gex.get('source','')}",
        ]
        if gex.get("gamma_flip"):
            lines.append(f"  Gamma flip: {gex['gamma_flip']}")
        if gex.get("call_wall") and gex.get("put_wall"):
            lines.append(f"  Call wall: {gex['call_wall']}  |  Put wall: {gex['put_wall']}")
        lines += ["━━━━━━━━━━━━━━━━━━━━", brief,
                  "━━━━━━━━━━━━━━━━━━━━",
                  "<b>ENTRY WINDOWS (PST):</b>  07:35 · 08:35 · 09:35 · 10:35 · 11:35",
                  "⛔ Hard exit: 14:30 PST"]
        return "\n".join(lines)

    def trade_msg(self, market: dict, gex: dict, trade: dict,
                  verdict: str, entry_num: int) -> str:
        re    = {"GO": "🟢", "CAUTION": "🟡", "SKIP": "🔴"}.get(gex.get("regime","?"), "⚪")
        ve    = "✅" if "GO ✅" in verdict else ("⏳" if "WAIT" in verdict else "❌")
        stop_c = trade["stop_per_side"] / 100
        lines = [
            f"<b>🎯 BIC ENTRY #{entry_num} {ve}  —  "
            f"{now_pst().strftime('%H:%M')} PST / {now_et().strftime('%H:%M')} ET</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            "<b>MARKET</b>",
            f"  SPX <b>{market.get('spx','?')}</b>  ({market.get('spx_change_pct',0):+.2f}%)    "
            f"VIX <b>{market.get('vix','?')}</b>",
            f"  GEX {re} <b>{gex.get('regime','?')}</b>  |  {gex.get('source','')}",
        ]
        if gex.get("gamma_flip"):
            lines.append(f"  Gamma flip: {gex['gamma_flip']}")
        if gex.get("call_wall") and gex.get("put_wall"):
            lines.append(f"  Walls ▲{gex['call_wall']}  ▼{gex['put_wall']}")
        lines += [
            "━━━━━━━━━━━━━━━━━━━━",
            "<b>TRADE LEGS</b>",
            f"  📉 SELL <b>{trade['put_short']}P</b>  /  BUY {trade['put_long']}P"
            f"    Δ{trade['put_delta']}  ${trade['put_credit']}",
            f"  📈 SELL <b>{trade['call_short']}C</b>  /  BUY {trade['call_long']}C"
            f"    Δ{trade['call_delta']}  ${trade['call_credit']}",
            "━━━━━━━━━━━━━━━━━━━━",
            "<b>RISK MANAGEMENT</b>",
            f"  Total credit    <b>${trade['total_credit']}</b>",
            f"  Stop per side   <b>${trade['stop_per_side']}</b>  (spread value &gt; ${stop_c:.2f}/contract)",
            f"  50% target      <b>${trade['profit_target']}</b>",
            f"  Max loss (2×)   ${trade['max_loss']}",
            f"  BP required     ${trade['bp_required']}",
            f"  Defended        {trade['defended_low']} – {trade['defended_high']}  ({trade['range_pts']} pts)",
            f"  Time to exp     ~{trade['T_hours']} hrs  |  IV {trade['sigma_pct']}%",
            "━━━━━━━━━━━━━━━━━━━━",
            "<b>SET OCO STOPS IMMEDIATELY</b>",
            f"  Put spread:   stop if value &gt; <b>${stop_c:.2f}</b>",
            f"  Call spread:  stop if value &gt; <b>${stop_c:.2f}</b>",
            "  (stop-limit first + stop-market backup 0.15 behind)",
            "━━━━━━━━━━━━━━━━━━━━",
            "<b>AI ANALYSIS</b>",
            verdict,
            "━━━━━━━━━━━━━━━━━━━━",
            f"<i>Wings: {trade['wing_used']}pt  |  Imbalance: {trade['imbalance_pct']}%  |  "
            f"Hard exit: 14:30 PST</i>",
        ]
        return "\n".join(lines)

    def skip_msg(self, reason: str, regime: str, spx, vix) -> str:
        re = {"GO":"🟢","CAUTION":"🟡","SKIP":"🔴"}.get(regime,"⚪")
        return (f"<b>⛔ BIC SKIP  —  {now_pst().strftime('%H:%M')} PST</b>\n"
                f"Regime {re} {regime}  |  SPX {spx}  |  VIX {vix}\n{reason}")

    def no_setup_msg(self, spx, vix, regime) -> str:
        return (f"<b>⚠️ BIC NO SETUP  —  {now_pst().strftime('%H:%M')} PST</b>\n"
                f"SPX {spx}  |  VIX {vix}  |  Regime {regime}\n"
                "No delta 5–10 legs with adequate credit found.\n"
                "Possible: little time left, low IV, wide spreads.\n"
                "<i>Next window in ~60 min</i>")

    def exit_msg(self) -> str:
        return (f"<b>🔔 HARD EXIT  —  {now_pst().strftime('%H:%M')} PST</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "⚠️ <b>CLOSE ALL 0DTE POSITIONS NOW</b>\n"
                "BIC rule: no positions past 2:30 PM PST.\n"
                "Use market orders — do not wait for fills.\n"
                "Gamma is extreme in the final 90 minutes.\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "<i>Next alert: tomorrow 06:30 PST</i>")

    def monitor_breach_msg(self, alerts: list, spx: float, time_str: str) -> str:
        lines = [
            f"<b>🚨 BIC LEG MONITOR — {time_str}</b>",
            f"SPX: <b>{spx}</b>",
            "━━━━━━━━━━━━━━━━━━━━",
        ]
        lines += alerts
        lines += ["━━━━━━━━━━━━━━━━━━━━",
                  "<i>Check positions — consider closing threatened leg(s)</i>"]
        return "\n".join(lines)

    def monitor_warn_msg(self, warnings: list, spx: float, time_str: str) -> str:
        lines = [
            f"<b>⚠️ BIC LEG WARNING — {time_str}</b>",
            f"SPX: <b>{spx}</b>",
            "━━━━━━━━━━━━━━━━━━━━",
        ]
        lines += warnings
        lines += ["━━━━━━━━━━━━━━━━━━━━",
                  "<i>Monitor closely — no action required yet</i>"]
        return "\n".join(lines)

    def pipeline_error_msg(self, error: str) -> str:
        return (f"<b>🔴 BIC PIPELINE ERROR — {now_pst().strftime('%H:%M')} PST</b>\n"
                f"{error}\n"
                "<i>Check GitHub Actions logs</i>")


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
class BICSystem:

    def __init__(self):
        self.md  = MarketData()
        self.gex = GEXData()
        self.sel = BICSelector()
        self.ai  = ClaudeAnalyst()
        self.tg  = Telegram()

    def run_morning(self):
        log.info("=== MORNING SCAN ===")
        market = self.md.get_spx_vix()
        gex    = self.gex.get(market.get("vix", 20.0))
        brief  = self.ai.premarket_brief(market, gex)
        self.tg.send(self.tg.morning_msg(market, gex, brief))

    def run_entry(self, entry_num: int, force: bool = False):
        log.info(f"=== ENTRY SCAN #{entry_num} ===")

        if not force and not is_entry_window():
            log.info("Outside entry window — cron fired outside valid window")
            return

        market = self.md.get_spx_vix()
        spx    = market.get("spx")
        vix    = market.get("vix")

        if not spx or not vix:
            log.error("Market data unavailable")
            return

        if vix > cfg.VIX_SKIP_ABOVE:
            self.tg.send(self.tg.skip_msg(
                f"VIX {vix:.1f} exceeds skip threshold {cfg.VIX_SKIP_ABOVE}. No trades today.",
                "SKIP", spx, vix))
            return

        gex = self.gex.get(vix)

        if gex["regime"] == "SKIP":
            self.tg.send(self.tg.skip_msg(
                "Negative GEX regime — dealers amplify moves. Stand aside this window.",
                "SKIP", spx, vix))
            return

        expiry  = self.md.get_today_expiry()
        options = self.md.get_options_chain(expiry) or []
        trade   = self.sel.find_best_legs(options, spx, vix, gex["regime"])

        if not trade:
            self.tg.send(self.tg.no_setup_msg(spx, vix, gex["regime"]))
            return

        verdict = self.ai.analyze_trade(market, gex, trade, entry_num)
        self.tg.send(self.tg.trade_msg(market, gex, trade, verdict, entry_num))

    def run_exit(self):
        log.info("=== EXIT REMINDER ===")
        if is_market_open():
            self.tg.send(self.tg.exit_msg())
        else:
            log.info("Market closed — exit reminder suppressed")

    def run_monitor(self):
        """
        Leg breach monitor — called every ~12 min by the leg-monitor GA job.
        Fetches all open 0DTE SPX positions, checks live delta on each short leg,
        and fires tiered Telegram alerts:
          🚨 BREACH  — live |delta| >= BREACH_DELTA (0.30) OR already ITM
          ⚠️  WARNING — live |delta| >= BREACH_WARN_DELTA (0.20) OR < 15 min to breach
        Exits silently if no open positions or outside trading hours.
        """
        log.info("=== LEG MONITOR ===")

        if not is_monitor_window():
            log.info("Outside monitor window — skipping")
            return

        # Fetch open positions — exit quietly if none
        try:
            positions = self.md.get_open_positions()
        except Exception as e:
            self.tg.send(self.tg.pipeline_error_msg(f"Position fetch failed: {e}"))
            raise

        if not positions:
            log.info("No open 0DTE positions — monitor quiet")
            return

        # Fetch live SPX price for context
        market = self.md.get_spx_vix()
        spx    = market.get("spx") or 0.0
        vix    = market.get("vix") or 20.0
        expiry = self.md.get_today_expiry()
        time_str = f"{now_pst().strftime('%H:%M')} PST / {now_et().strftime('%H:%M')} ET"

        breach_alerts  = []   # 🚨 — immediate action needed
        warning_alerts = []   # ⚠️  — close watch

        for pos in positions:
            sym = pos["symbol"]
            qty = pos["quantity"]

            # Determine option type from symbol (standard OCC format: ...C... or ...P...)
            # OCC symbol: SPX + YYMMDD + C/P + 8-digit strike * 1000
            opt_type = None
            for i, ch in enumerate(sym):
                if ch in ("C", "P") and i > 6:
                    opt_type = ch.lower()
                    break
            if not opt_type:
                log.warning(f"Cannot determine option type for {sym} — skipping")
                continue

            # Skip long legs (positive quantity = long; negative = short)
            # Only short legs (qty < 0) carry breach risk
            if qty > 0:
                log.info(f"Long leg {sym} qty={qty} — no breach risk, skipping")
                continue

            # Fetch live greeks
            greeks = self.md.get_live_greeks(sym, expiry)
            if not greeks:
                log.warning(f"Could not fetch greeks for {sym} — skipping leg")
                warning_alerts.append(
                    f"  ⚠️ {sym}: <b>greeks unavailable</b> — verify manually"
                )
                continue

            live_delta = abs(greeks["delta"])
            live_gamma = abs(greeks["gamma"])
            bid        = greeks["bid"]
            ask        = greeks["ask"]
            mid        = round((bid + ask) / 2, 2) if bid and ask else 0.0

            log.info(f"{sym} | Δ={live_delta:.3f} γ={live_gamma:.4f} mid={mid}")

            # ── Tier 1: Already breached ──────────────────────────────────────
            if live_delta >= cfg.BREACH_DELTA:
                breach_alerts.append(
                    f"  🔴 <b>{sym}</b>  Δ=<b>{live_delta:.2f}</b>  "
                    f"γ={live_gamma:.4f}  mid=${mid:.2f}\n"
                    f"       DELTA ≥ {cfg.BREACH_DELTA} — CLOSE THIS LEG NOW"
                )
                continue

            # ── Tier 2: Time-to-breach estimate ──────────────────────────────
            # At current gamma, delta grows ~gamma per $1 SPX move.
            # Using conservative SPX_1MIN_MOVE pts/min → minutes to breach.
            if live_gamma > 0:
                delta_headroom  = cfg.BREACH_DELTA - live_delta
                dollars_to_hit  = delta_headroom / live_gamma
                mins_to_breach  = dollars_to_hit / cfg.SPX_1MIN_MOVE
            else:
                mins_to_breach = 999.0

            if mins_to_breach <= 15:
                breach_alerts.append(
                    f"  🟠 <b>{sym}</b>  Δ=<b>{live_delta:.2f}</b>  "
                    f"γ={live_gamma:.4f}  mid=${mid:.2f}\n"
                    f"       ~<b>{mins_to_breach:.0f} min</b> to Δ{cfg.BREACH_DELTA} breach"
                )
            elif live_delta >= cfg.BREACH_WARN_DELTA or mins_to_breach <= 30:
                warning_alerts.append(
                    f"  🟡 <b>{sym}</b>  Δ={live_delta:.2f}  "
                    f"γ={live_gamma:.4f}  mid=${mid:.2f}  "
                    f"~{mins_to_breach:.0f} min to breach"
                )

        # ── Send alerts (breach takes priority) ──────────────────────────────
        if breach_alerts:
            self.tg.send(self.tg.monitor_breach_msg(breach_alerts, spx, time_str))
        elif warning_alerts:
            self.tg.send(self.tg.monitor_warn_msg(warning_alerts, spx, time_str))
        else:
            log.info(f"All {len(positions)} legs within safe thresholds — no alert sent")

    def run_test(self):
        """Complete cycle ignoring time guards — for manual testing."""
        log.info("=== TEST RUN (time guards bypassed) ===")
        market = self.md.get_spx_vix()
        spx    = market.get("spx") or 5500.0
        vix    = market.get("vix") or 18.0
        log.info(f"SPX={spx}  VIX={vix}  T={time_to_expiry_years()*365*24:.1f}h")

        gex     = self.gex.get(vix)
        expiry  = self.md.get_today_expiry()
        options = self.md.get_options_chain(expiry) or []
        trade   = self.sel.find_best_legs(options, spx, vix, "GO")

        if not trade:
            self.tg.send(
                "⚠️ <b>TEST RUN — No trade constructed</b>\n"
                f"SPX={spx}  VIX={vix}  T={time_to_expiry_years()*365*24:.1f}h\n"
                "If outside market hours this is expected (T≈0 → no credit).\n"
                "Re-run between 09:35–14:00 ET on a weekday."
            )
            return

        verdict = self.ai.analyze_trade(market, gex, trade, 1)
        msg     = "<b>[TEST RUN — not a live recommendation]</b>\n" + \
                  self.tg.trade_msg(market, gex, trade, verdict, 1)
        self.tg.send(msg)
        log.info("Test complete — check Telegram")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIC 0DTE SPX Alert System v3.1")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--morning", action="store_true",   help="Morning regime scan")
    group.add_argument("--entry",   type=int, metavar="N", help="Entry scan #N (1-5)")
    group.add_argument("--exit",    action="store_true",   help="Exit reminder")
    group.add_argument("--monitor", action="store_true",   help="Leg breach monitor (runs every ~12 min)")
    group.add_argument("--test",    action="store_true",   help="Test run (bypasses time guards)")
    args = parser.parse_args()

    sys = BICSystem()
    if   args.morning:             sys.run_morning()
    elif args.entry is not None:   sys.run_entry(args.entry)
    elif args.exit:                sys.run_exit()
    elif args.monitor:             sys.run_monitor()
    elif args.test:                sys.run_test()
