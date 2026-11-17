"""cbBTC 1-minute trading bot for btc-race-24h-20261117-0603.

Strategy: range-low mean reversion, one regime only, full size, long only.

  The dev window is range bound (about 63,900 to 65,500, ends only +0.5%
  from where it started) and its sharp moves revert: after a 20-minute
  rally of +1% the next 60 minutes average about -140 bps, and after a
  20-minute drop of -1% the next 60 to 120 minutes average about +40 to
  +130 bps. Chasing momentum therefore loses on this market, while
  buying near the trailing low and waiting for the bounce back wins.
  So this bot buys only when the close is within 0.4% of the lowest
  close of the last 200 bars, and only on an uptick (the close above
  the previous close), which is the first sign the dip is turning.

  Exit: take profit at +1.2% versus cost, stop loss at -1.2% versus
  cost, or a 480-bar time exit, whichever comes first. A full round
  trip costs roughly 37 bps here (4 bps fee + 2 bps slippage + about
  12 bps of pool impact each way at the depths this window shows),
  so the +1.2% target leaves a wide margin over costs.

  Race-window safety nets (they never fire on the dev window, where
  the first trade comes at bar 200):
  - two consecutive stop-loss exits disable new entries for the rest
    of the window, so a strongly down-trending race window cannot
    bleed the account trade after trade;
  - no new entries after bar 1300, so a late buy cannot be flattened
    at the close before it has time to work;
  - if nothing has traded by bar 1200, a looser fallback dip entry
    fires once, so a quiet race window still meets the 1-trade
    guardrail.

  All comparisons are integer cross-multiplications on the raw integer
  closes, so the bot is exactly deterministic. No network, no wall
  clock, no randomness; the only state is the list of closes seen so
  far. Protocol: agent-fun/bars-replay/v1, via the vendored kit shim.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"
MAX_SLIPPAGE_BPS = 100

# Range-low entry: close within 0.4% of the 200-bar trailing low.
RANGE_BARS = 200
RANGE_BPS = 40
# Exits, versus the harness-reported cost (fees included).
TAKE_BPS = 120
STOP_BPS = 120
MAX_HOLD_BARS = 480
# After a sell, wait this many bars before the next entry.
COOLDOWN_BARS = 20
# Safety nets for the unseen race window.
STOP_BREAKER = 2          # consecutive stop-loss exits that halt entries
ENTRY_CUTOFF_SEQ = 1300   # no new entries after this batch
# Fallback (only when nothing has traded yet): a 1.0% dip from the
# 30-bar high, quicker target, tighter stop.
FALLBACK_SEQ = 1200
FB_DIP_BARS = 30
FB_DIP_BPS = 100
FB_TAKE_BPS = 100
FB_STOP_BPS = 60
FB_MAX_HOLD_BARS = 200


class Bot:
    def __init__(self):
        self.closes = []
        self.mode = None          # "range" or "fb": regime of the open trade
        self.entry_seq = -1       # batch seq the buy order was sent on
        self.filled_buys = 0
        self.last_sell_seq = -1000000
        self.consec_stops = 0
        self.pending_exit = None  # "take", "stop" or "time" for the sell sent

    def on_batch(self, state, batch):
        for fill in batch.get("fills", []):
            if fill.get("status") != "filled":
                continue
            if fill.get("side") == "buy":
                self.filled_buys += 1
            elif fill.get("side") == "sell" and self.pending_exit is not None:
                if self.pending_exit == "stop":
                    self.consec_stops += 1
                else:
                    self.consec_stops = 0
                self.pending_exit = None
                self.last_sell_seq = state.seq
        for bar in batch.get("bars", []):
            if bar.get("symbol") == SYMBOL:
                self.closes.append(int(bar["c"]))
        if state.warming_up:
            return []
        if not self.closes:
            return []
        held = state.positions.get(SYMBOL)
        if held is not None:
            return self._maybe_exit(state, held)
        return self._maybe_enter(state)

    def _maybe_exit(self, state, held):
        value = state.value(SYMBOL, held.qty)
        cost = held.cost
        if self.mode == "fb":
            take_bps = FB_TAKE_BPS
            stop_bps = FB_STOP_BPS
            max_hold = FB_MAX_HOLD_BARS
        else:
            take_bps = TAKE_BPS
            stop_bps = STOP_BPS
            max_hold = MAX_HOLD_BARS
        if value * 10000 >= cost * (10000 + take_bps):
            reason = "take"
        elif value * 10000 <= cost * (10000 - stop_bps):
            reason = "stop"
        elif state.seq - self.entry_seq >= max_hold:
            reason = "time"
        else:
            return []
        self.pending_exit = reason
        return [{"side": "sell", "symbol": SYMBOL, "qty": held.qty,
                 "client_id": "s%d" % state.seq,
                 "max_slippage_bps": MAX_SLIPPAGE_BPS}]

    def _maybe_enter(self, state):
        if self.consec_stops >= STOP_BREAKER:
            return []
        if state.seq - self.last_sell_seq < COOLDOWN_BARS:
            return []
        if state.seq > ENTRY_CUTOFF_SEQ:
            return []
        closes = self.closes
        now = closes[-1]
        uptick = len(closes) >= 2 and now > closes[-2]
        if not uptick:
            return []
        # Depth-aware sizing guard: skip the entry when the pool at the
        # latest close is so shallow that a full-size buy would pay more
        # than 2.0% in price impact alone. On the dev window the depth
        # stays near $800k, so this never fires there; it only protects
        # the race window against a shallow-pool day.
        impact = state.impact_bps(SYMBOL, state.cash)
        if impact is not None and impact > 200:
            return []
        if len(closes) >= RANGE_BARS:
            low = min(closes[-RANGE_BARS:])
            if now * 10000 <= low * (10000 + RANGE_BPS):
                self.mode = "range"
                self.entry_seq = state.seq
                return self._buy(state)
        if self.filled_buys == 0 and state.seq >= FALLBACK_SEQ:
            if len(closes) > FB_DIP_BARS:
                high = max(closes[-1 - FB_DIP_BARS:])
                if now * 10000 <= high * (10000 - FB_DIP_BPS):
                    self.mode = "fb"
                    self.entry_seq = state.seq
                    return self._buy(state)
        return []

    def _buy(self, state):
        cash = state.cash
        if cash < int(state.hello["fees"]["min_notional_quote_units"]):
            return []
        return [{"side": "buy", "symbol": SYMBOL,
                 "notional_quote_units": cash,
                 "client_id": "b%d" % state.seq,
                 "max_slippage_bps": MAX_SLIPPAGE_BPS}]


BOT = Bot()


def on_batch(state, batch):
    return BOT.on_batch(state, batch)


makefun_bars.run(on_batch)
