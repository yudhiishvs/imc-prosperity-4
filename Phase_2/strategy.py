import json
from typing import Any, List, Dict, Tuple, Optional
import jsonpickle
import math

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out

logger = Logger()

class Trader:
    """
    Phase 2 Strategy — HYDROGEL_PACK
    =================================
    Passive inside-quote market maker anchored to a volume-weighted fair value.

    Strategy basis (see analysis/hydrogel/hydrogel_analysis.md):
      - ADF test confirms strong stationarity: ADF=-5.16, p=1.07e-5
      - O-U mean-reversion: κ≈0.13, half-life≈5 ticks
      - Spread is structurally locked at 16 ticks by opposing bot
      - Penny-jump (offset=1) to sit at the front of the queue inside the spread
      - Inventory skew adjusts quotes to naturally mean-revert position
      - Slight structural short bias on ask (intra-day downward drift ~100 ticks/day)

    All other products (VELVETFRUIT_EXTRACT, VEV_*) are stubbed — no orders sent.
    """

    # ── HYDROGEL_PACK parameters ──────────────────────────────────────────────
    HG_PRODUCT         = "HYDROGEL_PACK"
    HG_POSITION_LIMIT  = 200

    # Fair-value EMA: slow tracker (α=0.005 ≈ 200-tick window)
    HG_EMA_ALPHA       = 0.005
    # Volatility EMA: faster tracker of |Δmid|
    HG_VOL_EMA_ALPHA   = 0.10

    # Quote offsets (penny-jump exactly around fair value)
    HG_BID_OFFSET      = 1    
    HG_ASK_OFFSET      = 1    

    # OIM Threshold for quote pulling
    # Analysis shows OIM > 0 predicts a +4 tick move, and OIM < 0 predicts a -4 tick move.
    # We use a small threshold to avoid pure noise.
    HG_OIM_THRESHOLD   = 0.05

    # Passive quote size per side
    HG_BASE_QUOTE_SIZE = 20

    # Bootstrap fair value before any EMA history
    HG_INITIAL_FV      = 9900

    # ── End-of-day flattening (DISABLED — kept here for future use) ───────────
    # Activate by uncommenting the block in _trade_hydrogel() and setting:
    #   HG_EOD_TS     = 990000   # timestamp to start flattening
    #   HG_EOD_TARGET = 20       # max absolute position near end of day

    # ── Other Phase 2 products (stubs) ────────────────────────────────────────
    VEV_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

    # ─────────────────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self, raw: str) -> dict:
        default = {
            "hg_fair_ema": None,   # EMA of volume-weighted mid
            "hg_prev_mid": None,   # Previous raw mid (for vol EMA)
            "hg_vol_ema":  1.5,    # Seed with modest volatility estimate
        }
        if not raw:
            return default
        try:
            data = jsonpickle.decode(raw)
        except Exception:
            return default
        if not isinstance(data, dict):
            return default
        for k, v in default.items():
            if k not in data or data[k] is None:
                data[k] = v
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # Shared utility helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _ewma(previous: Optional[float], value: float, alpha: float) -> float:
        if previous is None:
            return float(value)
        return (1.0 - alpha) * float(previous) + alpha * float(value)

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[int, int]:
        best_bid = max(depth.buy_orders.keys())  if depth.buy_orders  else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
        return best_bid, best_ask

    @staticmethod
    def _wm_price(depth: OrderDepth) -> Optional[float]:
        """Volume-weighted mid: weights best bid/ask by opposite-side volume."""
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bid_vol  = depth.buy_orders[best_bid]
        ask_vol  = abs(depth.sell_orders[best_ask])
        total    = bid_vol + ask_vol
        if total == 0:
            return (best_bid + best_ask) / 2.0
        return (best_bid * ask_vol + best_ask * bid_vol) / total

    def _buy_room(self, position: int, pending_buys: int) -> int:
        return self.HG_POSITION_LIMIT - (position + pending_buys)

    def _sell_room(self, position: int, pending_sells: int) -> int:
        return self.HG_POSITION_LIMIT + (position - pending_sells)

    def _place_buy(self, orders: List[Order], price: int, desired_qty: int,
                   position: int, pending_buys: int) -> int:
        qty = min(desired_qty, self._buy_room(position, pending_buys))
        if qty > 0:
            orders.append(Order(self.HG_PRODUCT, price, qty))
            pending_buys += qty
        return pending_buys

    def _place_sell(self, orders: List[Order], price: int, desired_qty: int,
                    position: int, pending_sells: int) -> int:
        qty = min(desired_qty, self._sell_room(position, pending_sells))
        if qty > 0:
            orders.append(Order(self.HG_PRODUCT, price, -qty))
            pending_sells += qty
        return pending_sells

    # ─────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK trading logic
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_hydrogel(self, state: TradingState, data: dict) -> Tuple[List[Order], dict]:
        depth    = state.order_depths.get(self.HG_PRODUCT)
        if depth is None:
            return [], data

        orders: List[Order] = []
        position = state.position.get(self.HG_PRODUCT, 0)
        pending_buys, pending_sells = 0, 0

        # ── 1. Book state ──────────────────────────────────────────────────
        best_bid, best_ask = self._best_bid_ask(depth)
        raw_mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else None

        # ── 2. Fair value (volume-weighted mid EMA) ────────────────────────
        wm = self._wm_price(depth)
        fair_input = wm if wm is not None else raw_mid

        hg_fair_ema = data["hg_fair_ema"]
        if hg_fair_ema is None:
            hg_fair_ema = float(fair_input) if fair_input else self.HG_INITIAL_FV

        if fair_input is not None:
            hg_fair_ema = self._ewma(hg_fair_ema, fair_input, self.HG_EMA_ALPHA)

        fair = hg_fair_ema

        # ── 3. Volatility update ───────────────────────────────────────────
        hg_vol_ema = data["hg_vol_ema"]
        if raw_mid is not None and data["hg_prev_mid"] is not None:
            ret = abs(raw_mid - data["hg_prev_mid"])
            hg_vol_ema = self._ewma(hg_vol_ema, ret, self.HG_VOL_EMA_ALPHA)

        # ── 4. Order Imbalance (OIM) ───────────────────────────────────────
        bid_vol = depth.buy_orders.get(best_bid, 0)
        ask_vol = abs(depth.sell_orders.get(best_ask, 0))
        oim = 0.0
        if (bid_vol + ask_vol) > 0:
            oim = (bid_vol - ask_vol) / (bid_vol + ask_vol)

        # ── 5. Quote prices (penny-jump: exactly around fair value) ────────
        bid_price = math.floor(fair - self.HG_BID_OFFSET)
        ask_price = math.ceil( fair + self.HG_ASK_OFFSET)

        # Hard guard: never cross the spread (act exclusively as a maker)
        if best_ask > 0:
            bid_price = min(bid_price, best_ask - 1)
        if best_bid > 0:
            ask_price = max(ask_price, best_bid + 1)

        # Ensure our own quotes don't cross each other
        if bid_price >= ask_price:
            bid_price = ask_price - 1

        # ── 6. Quote sizes and OIM Quote Pulling ───────────────────────────
        bid_qty = min(self.HG_BASE_QUOTE_SIZE, self._buy_room(position, pending_buys))
        ask_qty = min(self.HG_BASE_QUOTE_SIZE, self._sell_room(position, pending_sells))

        # Analysis shows OIM predicts 4-tick jumps. Pull the adverse quote.
        if oim > self.HG_OIM_THRESHOLD:
            # Bullish imbalance -> price going up -> don't sell
            ask_qty = 0
        elif oim < -self.HG_OIM_THRESHOLD:
            # Bearish imbalance -> price going down -> don't buy
            bid_qty = 0

        # ── 7. End-of-day flattening (DISABLED — uncomment to activate) ────
        # Flatten toward HG_EOD_TARGET when approaching end of day.
        # HG_EOD_TS     = 990_000
        # HG_EOD_TARGET = 20
        # if state.timestamp >= HG_EOD_TS:
        #     if position > HG_EOD_TARGET:
        #         # Hit best bids aggressively to reduce long position
        #         for bid in sorted(depth.buy_orders, reverse=True):
        #             room = self._sell_room(position, pending_sells)
        #             size = min(depth.buy_orders[bid], room,
        #                        position - HG_EOD_TARGET - pending_sells)
        #             if size <= 0: break
        #             orders.append(Order(self.HG_PRODUCT, bid, -size))
        #             pending_sells += size
        #         bid_qty = 0  # suppress passive bid
        #     elif position < -HG_EOD_TARGET:
        #         # Lift best asks aggressively to reduce short position
        #         for ask in sorted(depth.sell_orders):
        #             room = self._buy_room(position, pending_buys)
        #             size = min(-depth.sell_orders[ask], room,
        #                        -HG_EOD_TARGET - position - pending_buys)
        #             if size <= 0: break
        #             orders.append(Order(self.HG_PRODUCT, ask, size))
        #             pending_buys += size
        #         ask_qty = 0  # suppress passive ask

        # ── 8. Place passive quotes ────────────────────────────────────────
        if bid_qty > 0:
            pending_buys  = self._place_buy( orders, bid_price, bid_qty,  position, pending_buys)
        if ask_qty > 0:
            pending_sells = self._place_sell(orders, ask_price, ask_qty, position, pending_sells)

        logger.print(f"[HG] ts={state.timestamp}  pos={position}  fair={fair:.2f}  "
                     f"oim={oim:.2f}  bid={bid_price}x{bid_qty}  ask={ask_price}x{ask_qty}")

        # ── 9. Persist state ───────────────────────────────────────────────
        data.update({
            "hg_fair_ema": hg_fair_ema,
            "hg_prev_mid": raw_mid,
            "hg_vol_ema":  hg_vol_ema,
        })
        return orders, data

    # ─────────────────────────────────────────────────────────────────────────
    # run() — main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data   = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        # ── HYDROGEL_PACK ──────────────────────────────────────────────────
        if self.HG_PRODUCT in state.order_depths:
            hg_orders, data = self._trade_hydrogel(state, data)
            result[self.HG_PRODUCT] = hg_orders

        # ── VELVETFRUIT_EXTRACT & VEV_* vouchers (stubs) ──────────────────
        for product in ["VELVETFRUIT_EXTRACT"] + [f"VEV_{s}" for s in self.VEV_STRIKES]:
            if product in state.order_depths:
                result[product] = []

        traderData = jsonpickle.encode(data)
        logger.flush(state, result, 0, traderData)
        return result, 0, traderData
