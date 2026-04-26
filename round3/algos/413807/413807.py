import json
from typing import Any, Dict, List, Optional, Tuple
import jsonpickle
import math

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
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

        max_item_length = max(0, (self.max_log_length - base_length) // 3)

        print(
            self.to_json(
                [
                    self.compress_state(
                        state,
                        self.truncate(state.traderData, max_item_length),
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed: List[List[Any]] = []

        for listing in listings.values():
            compressed.append(
                [
                    listing.symbol,
                    listing.product,
                    listing.denomination,
                ]
            )

        return compressed

    def compress_order_depths(
        self,
        order_depths: Dict[Symbol, OrderDepth],
    ) -> Dict[Symbol, List[Any]]:
        compressed: Dict[Symbol, List[Any]] = {}

        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders,
                order_depth.sell_orders,
            ]

        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed: List[List[Any]] = []

        for trade_list in trades.values():
            for trade in trade_list:
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

    def compress_observations(self, observations: Observation) -> List[Any]:
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

        return [
            observations.plainValueObservations,
            conversion_observations,
        ]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed: List[List[Any]] = []

        for order_list in orders.values():
            for order in order_list:
                compressed.append(
                    [
                        order.symbol,
                        order.price,
                        order.quantity,
                    ]
                )

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if max_length <= 0:
            return ""

        lo = 0
        hi = min(len(value), max_length)
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
    HYDROGEL_PACK Adaptive Statistical Market Maker with wall-aware fair value.

    Active:
        HYDROGEL_PACK

    Disabled for now:
        VELVETFRUIT_EXTRACT
        VEV_4000
        VEV_4500
        VEV_5000
        VEV_5100
        VEV_5200
        VEV_5300
        VEV_5400
        VEV_5500
        VEV_6000
        VEV_6500

    Strategy:
        - Estimate base fair value using volume-weighted mid EMA.
        - Detect bid/ask walls from visible order-book depth.
        - Use wall prices as a capped secondary fair-value modifier.
        - Estimate volatility using EWMA absolute mid moves.
        - Estimate stationarity stress using abs(mid - adjusted_fair) / vol.
        - Use order imbalance as a short-term autocorrelation signal.
        - Quote maker-only around adjusted fair value.
        - Size down when volatility, stationarity stress, or inventory risk rises.
        - Stop adding risk near soft limits.
        - Emergency unwind only near hard limits.
    """

    # -------------------------------------------------------------------------
    # Product universe
    # -------------------------------------------------------------------------

    HG_PRODUCT = "HYDROGEL_PACK"

    DISABLED_PRODUCTS = [
        "VELVETFRUIT_EXTRACT",
        "VEV_4000",
        "VEV_4500",
        "VEV_5000",
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
        "VEV_5400",
        "VEV_5500",
        "VEV_6000",
        "VEV_6500",
    ]

    # -------------------------------------------------------------------------
    # Position limits
    # -------------------------------------------------------------------------

    HG_POSITION_LIMIT = 200

    # Soft limit: stop adding risk.
    HG_SOFT_POSITION_LIMIT = 140

    # Hard limit: emergency reduce inventory.
    # Kept high so we do not cross spread unless position is genuinely scary.
    HG_HARD_POSITION_LIMIT = 190

    # Emergency unwind target.
    HG_EMERGENCY_TARGET_POSITION = 150

    # Maximum size for one emergency unwind order.
    HG_MAX_EMERGENCY_TRADE = 30

    # -------------------------------------------------------------------------
    # Fair value, volatility, and stationarity parameters
    # -------------------------------------------------------------------------

    # Slow fair-value tracker.
    HG_EMA_ALPHA = 0.005

    # Faster volatility tracker.
    HG_VOL_EMA_ALPHA = 0.10

    # Stationarity stress tracker.
    HG_STRESS_EMA_ALPHA = 0.05

    # Bootstrap fair value if no data exists.
    HG_INITIAL_FV = 9900.0

    # -------------------------------------------------------------------------
    # Wall-aware fair value parameters
    # -------------------------------------------------------------------------

    WALL_ENABLED = True

    # A wall must be meaningfully larger than average visible size on that side.
    WALL_VOLUME_MULTIPLIER = 1.35

    # Ignore very small "walls".
    WALL_MIN_VOLUME = 18

    # Cap how far a wall can pull the fair value before blending.
    WALL_MAX_ADJUST_TICKS = 4.0

    # Final fair = base_fair + WALL_BLEND * capped(wall_anchor - base_fair).
    # Keep this modest so walls modify fair value rather than replace it.
    WALL_BLEND = 0.25

    # -------------------------------------------------------------------------
    # Quoting parameters
    # -------------------------------------------------------------------------

    HG_BID_OFFSET = 1
    HG_ASK_OFFSET = 1

    HG_BASE_QUOTE_SIZE = 20
    HG_MIN_QUOTE_SIZE = 4

    # Inventory skew:
    # Long inventory lowers quote fair value.
    # Short inventory raises quote fair value.
    HG_INVENTORY_SKEW_TICKS = 6.0

    # Order imbalance threshold.
    HG_OIM_THRESHOLD = 0.05

    # -------------------------------------------------------------------------
    # Volatility targeting
    # -------------------------------------------------------------------------

    HG_TARGET_VOL = 1.50
    HG_EXTREME_VOL = 6.00
    HG_KILL_VOL = 6.00

    # -------------------------------------------------------------------------
    # Stationarity stress targeting
    #
    # stress_z = abs(mid - adjusted_fair) / max(1, vol)
    # -------------------------------------------------------------------------

    HG_TARGET_STRESS_Z = 4.0
    HG_HIGH_STRESS_Z = 8.0
    HG_KILL_STRESS_Z = 12.0

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------

    def _load_data(self, raw: str) -> Dict[str, Any]:
        default = {
            "hg_fair_ema": None,
            "hg_prev_mid": None,
            "hg_vol_ema": 1.5,
            "hg_stress_ema": 0.0,
        }

        if not raw:
            return default

        try:
            data = jsonpickle.decode(raw)
        except Exception:
            return default

        if not isinstance(data, dict):
            return default

        for key, value in default.items():
            if key not in data or data[key] is None:
                data[key] = value

        return data

    def _save_hydrogel_state(
        self,
        data: Dict[str, Any],
        hg_fair_ema: float,
        raw_mid: float,
        hg_vol_ema: float,
        hg_stress_ema: float,
    ) -> Dict[str, Any]:
        data.update(
            {
                "hg_fair_ema": hg_fair_ema,
                "hg_prev_mid": raw_mid,
                "hg_vol_ema": hg_vol_ema,
                "hg_stress_ema": hg_stress_ema,
            }
        )

        return data

    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------

    @staticmethod
    def _ewma(previous: Optional[float], value: float, alpha: float) -> float:
        if previous is None:
            return float(value)

        return (1.0 - alpha) * float(previous) + alpha * float(value)

    @staticmethod
    def _safe_book(
        depth: Optional[OrderDepth],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns:
            best_bid, best_bid_volume, best_ask, best_ask_volume

        Prosperity convention:
            buy_orders volumes are positive.
            sell_orders volumes are negative.
        """

        if depth is None:
            return None

        if not depth.buy_orders or not depth.sell_orders:
            return None

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        best_bid_volume = depth.buy_orders.get(best_bid, 0)
        best_ask_volume = -depth.sell_orders.get(best_ask, 0)

        if best_bid <= 0 or best_ask <= 0:
            return None

        if best_bid_volume <= 0 or best_ask_volume <= 0:
            return None

        if best_bid >= best_ask:
            return None

        return best_bid, best_bid_volume, best_ask, best_ask_volume

    @staticmethod
    def _weighted_mid_price(
        best_bid: int,
        best_bid_volume: int,
        best_ask: int,
        best_ask_volume: int,
    ) -> float:
        """
        Volume-weighted mid.

        Large bid volume pushes fair value toward ask.
        Large ask volume pushes fair value toward bid.
        """

        total_volume = best_bid_volume + best_ask_volume

        if total_volume <= 0:
            return (best_bid + best_ask) / 2.0

        return (
            best_bid * best_ask_volume
            + best_ask * best_bid_volume
        ) / total_volume

    def _buy_room(self, position: int, pending_buys: int) -> int:
        return max(0, self.HG_POSITION_LIMIT - (position + pending_buys))

    def _sell_room(self, position: int, pending_sells: int) -> int:
        return max(0, self.HG_POSITION_LIMIT + (position - pending_sells))

    # -------------------------------------------------------------------------
    # Wall-price fair value helper functions
    # -------------------------------------------------------------------------

    def _largest_qualified_wall(
        self,
        orders: Dict[int, int],
        is_sell_side: bool,
    ) -> Tuple[Optional[int], int, float]:
        """
        Finds the largest meaningful visible wall on one side of the order book.

        Returns:
            wall_price, wall_volume, wall_strength

        wall_strength = largest_visible_volume / average_visible_volume

        If no qualified wall exists:
            returns None, 0, 0.0
        """

        if not orders:
            return None, 0, 0.0

        levels: List[Tuple[int, int]] = []

        for price, volume in orders.items():
            visible_volume = abs(volume) if is_sell_side else volume

            if price <= 0 or visible_volume <= 0:
                continue

            levels.append((price, visible_volume))

        # Need at least two visible levels to call something "unusually large."
        if len(levels) < 2:
            return None, 0, 0.0

        total_volume = sum(volume for _, volume in levels)
        avg_volume = total_volume / len(levels)

        if avg_volume <= 0:
            return None, 0, 0.0

        wall_price, wall_volume = max(levels, key=lambda x: x[1])
        wall_strength = wall_volume / avg_volume

        if wall_volume < self.WALL_MIN_VOLUME:
            return None, 0, 0.0

        if wall_strength < self.WALL_VOLUME_MULTIPLIER:
            return None, 0, 0.0

        return wall_price, wall_volume, wall_strength

    def _wall_anchor(
        self,
        depth: OrderDepth,
        base_fair: float,
    ) -> Tuple[float, str, Optional[int], Optional[int], float, float]:
        """
        Computes a wall-informed anchor.

        Important:
            This does NOT replace fair value.
            It proposes an anchor that later gets capped and blended.

        Cases:
            both bid and ask walls:
                anchor = midpoint of bid wall and ask wall

            bid wall only:
                anchor = max(base_fair, bid_wall_price)
                This treats the bid wall as support.

            ask wall only:
                anchor = min(base_fair, ask_wall_price)
                This treats the ask wall as resistance.

            no wall:
                anchor = base_fair
        """

        bid_wall_price, bid_wall_volume, bid_wall_strength = self._largest_qualified_wall(
            orders=depth.buy_orders,
            is_sell_side=False,
        )

        ask_wall_price, ask_wall_volume, ask_wall_strength = self._largest_qualified_wall(
            orders=depth.sell_orders,
            is_sell_side=True,
        )

        if bid_wall_price is not None and ask_wall_price is not None:
            if bid_wall_price < ask_wall_price:
                anchor = (bid_wall_price + ask_wall_price) / 2.0
                return (
                    anchor,
                    "BOTH",
                    bid_wall_price,
                    ask_wall_price,
                    bid_wall_strength,
                    ask_wall_strength,
                )

        if bid_wall_price is not None:
            anchor = max(base_fair, float(bid_wall_price))
            return (
                anchor,
                "BID_ONLY",
                bid_wall_price,
                None,
                bid_wall_strength,
                0.0,
            )

        if ask_wall_price is not None:
            anchor = min(base_fair, float(ask_wall_price))
            return (
                anchor,
                "ASK_ONLY",
                None,
                ask_wall_price,
                0.0,
                ask_wall_strength,
            )

        return (
            base_fair,
            "NONE",
            None,
            None,
            0.0,
            0.0,
        )

    def _apply_wall_adjustment(
        self,
        depth: OrderDepth,
        base_fair: float,
    ) -> Tuple[float, str, Optional[int], Optional[int], float, float, float]:
        """
        Applies the wall-price fair-value adjustment.

        adjusted_fair = base_fair + WALL_BLEND * capped(anchor - base_fair)

        Returns:
            adjusted_fair,
            wall_mode,
            bid_wall_price,
            ask_wall_price,
            bid_wall_strength,
            ask_wall_strength,
            applied_adjustment
        """

        if not self.WALL_ENABLED:
            return base_fair, "DISABLED", None, None, 0.0, 0.0, 0.0

        (
            anchor,
            wall_mode,
            bid_wall_price,
            ask_wall_price,
            bid_wall_strength,
            ask_wall_strength,
        ) = self._wall_anchor(
            depth=depth,
            base_fair=base_fair,
        )

        raw_adjustment = anchor - base_fair

        capped_adjustment = max(
            -self.WALL_MAX_ADJUST_TICKS,
            min(self.WALL_MAX_ADJUST_TICKS, raw_adjustment),
        )

        applied_adjustment = self.WALL_BLEND * capped_adjustment
        adjusted_fair = base_fair + applied_adjustment

        return (
            adjusted_fair,
            wall_mode,
            bid_wall_price,
            ask_wall_price,
            bid_wall_strength,
            ask_wall_strength,
            applied_adjustment,
        )

    # -------------------------------------------------------------------------
    # Statistical sizing
    # -------------------------------------------------------------------------

    def _dynamic_quote_size(
        self,
        hg_vol_ema: float,
        hg_stress_ema: float,
        position: int,
    ) -> int:
        """
        quote_size = base_size
                     * volatility_multiplier
                     * stationarity_multiplier
                     * inventory_multiplier
        """

        safe_vol = max(0.25, float(hg_vol_ema))

        if safe_vol >= self.HG_EXTREME_VOL:
            volatility_multiplier = 0.25
        else:
            volatility_multiplier = self.HG_TARGET_VOL / safe_vol
            volatility_multiplier = max(0.25, min(1.00, volatility_multiplier))

        safe_stress = max(0.0, float(hg_stress_ema))

        if safe_stress >= self.HG_HIGH_STRESS_Z:
            stationarity_multiplier = 0.40
        elif safe_stress >= self.HG_TARGET_STRESS_Z:
            stationarity_multiplier = 0.70
        else:
            stationarity_multiplier = 1.00

        inventory_ratio = abs(position) / self.HG_POSITION_LIMIT

        if inventory_ratio >= 0.75:
            inventory_multiplier = 0.40
        elif inventory_ratio >= 0.50:
            inventory_multiplier = 0.60
        elif inventory_ratio >= 0.25:
            inventory_multiplier = 0.80
        else:
            inventory_multiplier = 1.00

        raw_size = (
            self.HG_BASE_QUOTE_SIZE
            * volatility_multiplier
            * stationarity_multiplier
            * inventory_multiplier
        )

        final_size = int(round(raw_size))

        return max(
            self.HG_MIN_QUOTE_SIZE,
            min(self.HG_BASE_QUOTE_SIZE, final_size),
        )

    # -------------------------------------------------------------------------
    # Order placement
    # -------------------------------------------------------------------------

    def _place_buy(
        self,
        orders: List[Order],
        price: int,
        desired_quantity: int,
        position: int,
        pending_buys: int,
    ) -> int:
        quantity = min(
            desired_quantity,
            self._buy_room(position, pending_buys),
        )

        if quantity > 0:
            orders.append(Order(self.HG_PRODUCT, price, quantity))
            pending_buys += quantity

        return pending_buys

    def _place_sell(
        self,
        orders: List[Order],
        price: int,
        desired_quantity: int,
        position: int,
        pending_sells: int,
    ) -> int:
        quantity = min(
            desired_quantity,
            self._sell_room(position, pending_sells),
        )

        if quantity > 0:
            orders.append(Order(self.HG_PRODUCT, price, -quantity))
            pending_sells += quantity

        return pending_sells

    # -------------------------------------------------------------------------
    # Emergency inventory control
    # -------------------------------------------------------------------------

    def _maybe_emergency_unwind(
        self,
        orders: List[Order],
        position: int,
        best_bid: int,
        best_bid_volume: int,
        best_ask: int,
        best_ask_volume: int,
        pending_buys: int,
        pending_sells: int,
    ) -> Tuple[bool, int, int]:
        """
        Returns:
            did_unwind, pending_buys, pending_sells

        This is the only part of the strategy that crosses the spread.
        It triggers only near the hard inventory limit.
        """

        if position >= self.HG_HARD_POSITION_LIMIT:
            desired_reduce = position - self.HG_EMERGENCY_TARGET_POSITION

            reduce_size = min(
                desired_reduce,
                self.HG_MAX_EMERGENCY_TRADE,
                best_bid_volume,
                self._sell_room(position, pending_sells),
            )

            if reduce_size > 0:
                orders.append(Order(self.HG_PRODUCT, best_bid, -reduce_size))
                pending_sells += reduce_size

            logger.print(
                f"[HG EMERGENCY LONG] "
                f"pos={position} "
                f"sell={best_bid}x{reduce_size}"
            )

            return True, pending_buys, pending_sells

        if position <= -self.HG_HARD_POSITION_LIMIT:
            desired_reduce = abs(position) - self.HG_EMERGENCY_TARGET_POSITION

            reduce_size = min(
                desired_reduce,
                self.HG_MAX_EMERGENCY_TRADE,
                best_ask_volume,
                self._buy_room(position, pending_buys),
            )

            if reduce_size > 0:
                orders.append(Order(self.HG_PRODUCT, best_ask, reduce_size))
                pending_buys += reduce_size

            logger.print(
                f"[HG EMERGENCY SHORT] "
                f"pos={position} "
                f"buy={best_ask}x{reduce_size}"
            )

            return True, pending_buys, pending_sells

        return False, pending_buys, pending_sells

    # -------------------------------------------------------------------------
    # HYDROGEL strategy
    # -------------------------------------------------------------------------

    def _trade_hydrogel(
        self,
        state: TradingState,
        data: Dict[str, Any],
    ) -> Tuple[List[Order], Dict[str, Any]]:
        depth = state.order_depths.get(self.HG_PRODUCT)

        if depth is None:
            return [], data

        book = self._safe_book(depth)

        if book is None:
            return [], data

        best_bid, best_bid_volume, best_ask, best_ask_volume = book

        orders: List[Order] = []

        position = state.position.get(self.HG_PRODUCT, 0)

        pending_buys = 0
        pending_sells = 0

        raw_mid = (best_bid + best_ask) / 2.0

        # ---------------------------------------------------------------------
        # 1. Base mean / fair value estimate
        # ---------------------------------------------------------------------

        weighted_mid = self._weighted_mid_price(
            best_bid=best_bid,
            best_bid_volume=best_bid_volume,
            best_ask=best_ask,
            best_ask_volume=best_ask_volume,
        )

        hg_fair_ema = data["hg_fair_ema"]

        if hg_fair_ema is None:
            hg_fair_ema = float(weighted_mid) if weighted_mid is not None else self.HG_INITIAL_FV

        hg_fair_ema = self._ewma(
            previous=hg_fair_ema,
            value=weighted_mid,
            alpha=self.HG_EMA_ALPHA,
        )

        base_fair = hg_fair_ema

        # ---------------------------------------------------------------------
        # 1b. Wall-aware fair value adjustment
        # ---------------------------------------------------------------------

        (
            fair,
            wall_mode,
            bid_wall_price,
            ask_wall_price,
            bid_wall_strength,
            ask_wall_strength,
            wall_adjustment,
        ) = self._apply_wall_adjustment(
            depth=depth,
            base_fair=base_fair,
        )

        # ---------------------------------------------------------------------
        # 2. Variance / volatility estimate
        # ---------------------------------------------------------------------

        hg_vol_ema = data["hg_vol_ema"]

        previous_mid = data.get("hg_prev_mid", None)

        if previous_mid is not None:
            absolute_move = abs(raw_mid - previous_mid)

            hg_vol_ema = self._ewma(
                previous=hg_vol_ema,
                value=absolute_move,
                alpha=self.HG_VOL_EMA_ALPHA,
            )

        # ---------------------------------------------------------------------
        # 3. Stationarity stress estimate
        # ---------------------------------------------------------------------

        stationarity_error = abs(raw_mid - fair)
        stationarity_z = stationarity_error / max(1.0, hg_vol_ema)

        hg_stress_ema = data["hg_stress_ema"]

        hg_stress_ema = self._ewma(
            previous=hg_stress_ema,
            value=stationarity_z,
            alpha=self.HG_STRESS_EMA_ALPHA,
        )

        # ---------------------------------------------------------------------
        # 4. Autocorrelation / order imbalance signal
        # ---------------------------------------------------------------------

        order_imbalance = 0.0

        if best_bid_volume + best_ask_volume > 0:
            order_imbalance = (
                best_bid_volume - best_ask_volume
            ) / (best_bid_volume + best_ask_volume)

        # ---------------------------------------------------------------------
        # 5. Inventory skew
        # ---------------------------------------------------------------------

        inventory_ratio = position / self.HG_POSITION_LIMIT
        inventory_skew = inventory_ratio * self.HG_INVENTORY_SKEW_TICKS

        # Long inventory lowers fair_for_quotes.
        # Short inventory raises fair_for_quotes.
        fair_for_quotes = fair - inventory_skew

        bid_price = math.floor(fair_for_quotes - self.HG_BID_OFFSET)
        ask_price = math.ceil(fair_for_quotes + self.HG_ASK_OFFSET)

        # Maker-only guard.
        bid_price = min(bid_price, best_ask - 1)
        ask_price = max(ask_price, best_bid + 1)

        if bid_price >= ask_price:
            bid_price = ask_price - 1

        # ---------------------------------------------------------------------
        # 6. Size using volatility, stationarity, and inventory
        # ---------------------------------------------------------------------

        dynamic_quote_size = self._dynamic_quote_size(
            hg_vol_ema=hg_vol_ema,
            hg_stress_ema=hg_stress_ema,
            position=position,
        )

        bid_quantity = min(
            dynamic_quote_size,
            self._buy_room(position, pending_buys),
        )

        ask_quantity = min(
            dynamic_quote_size,
            self._sell_room(position, pending_sells),
        )

        # ---------------------------------------------------------------------
        # 7. Pull adverse side using order imbalance
        # ---------------------------------------------------------------------

        if order_imbalance > self.HG_OIM_THRESHOLD:
            # Bullish imbalance. Avoid selling.
            ask_quantity = 0
        elif order_imbalance < -self.HG_OIM_THRESHOLD:
            # Bearish imbalance. Avoid buying.
            bid_quantity = 0

        # ---------------------------------------------------------------------
        # 8. Stationarity-aware side filter
        #
        # If price is stretched above adjusted fair, do not buy high.
        # If price is stretched below adjusted fair, do not sell low.
        # ---------------------------------------------------------------------

        if hg_stress_ema >= self.HG_TARGET_STRESS_Z:
            if raw_mid > fair:
                bid_quantity = 0
            elif raw_mid < fair:
                ask_quantity = 0

        # ---------------------------------------------------------------------
        # 9. Soft inventory guard
        # ---------------------------------------------------------------------

        if position >= self.HG_SOFT_POSITION_LIMIT:
            # Too long. Stop buying.
            bid_quantity = 0

        if position <= -self.HG_SOFT_POSITION_LIMIT:
            # Too short. Stop selling.
            ask_quantity = 0

        # ---------------------------------------------------------------------
        # 10. Volatility kill switch
        # ---------------------------------------------------------------------

        if hg_vol_ema >= self.HG_KILL_VOL:
            if position > 0:
                bid_quantity = 0
            elif position < 0:
                ask_quantity = 0
            else:
                bid_quantity = 0
                ask_quantity = 0

        # ---------------------------------------------------------------------
        # 11. Stationarity kill switch
        # ---------------------------------------------------------------------

        if hg_stress_ema >= self.HG_KILL_STRESS_Z:
            if position > 0:
                bid_quantity = 0
            elif position < 0:
                ask_quantity = 0
            else:
                bid_quantity = 0
                ask_quantity = 0

        # ---------------------------------------------------------------------
        # 12. Emergency inventory unwind
        # ---------------------------------------------------------------------

        did_unwind, pending_buys, pending_sells = self._maybe_emergency_unwind(
            orders=orders,
            position=position,
            best_bid=best_bid,
            best_bid_volume=best_bid_volume,
            best_ask=best_ask,
            best_ask_volume=best_ask_volume,
            pending_buys=pending_buys,
            pending_sells=pending_sells,
        )

        if did_unwind:
            data = self._save_hydrogel_state(
                data=data,
                hg_fair_ema=hg_fair_ema,
                raw_mid=raw_mid,
                hg_vol_ema=hg_vol_ema,
                hg_stress_ema=hg_stress_ema,
            )

            return orders, data

        # ---------------------------------------------------------------------
        # 13. Place passive maker quotes
        # ---------------------------------------------------------------------

        if bid_quantity > 0:
            pending_buys = self._place_buy(
                orders=orders,
                price=bid_price,
                desired_quantity=bid_quantity,
                position=position,
                pending_buys=pending_buys,
            )

        if ask_quantity > 0:
            pending_sells = self._place_sell(
                orders=orders,
                price=ask_price,
                desired_quantity=ask_quantity,
                position=position,
                pending_sells=pending_sells,
            )

        logger.print(
            f"[HG] "
            f"ts={state.timestamp} "
            f"pos={position} "
            f"base_fair={base_fair:.2f} "
            f"fair={fair:.2f} "
            f"fair_q={fair_for_quotes:.2f} "
            f"wall_adj={wall_adjustment:.2f} "
            f"wall_mode={wall_mode} "
            f"bid_wall={bid_wall_price} "
            f"ask_wall={ask_wall_price} "
            f"bid_ws={bid_wall_strength:.2f} "
            f"ask_ws={ask_wall_strength:.2f} "
            f"mid={raw_mid:.2f} "
            f"vol={hg_vol_ema:.2f} "
            f"stress_z={hg_stress_ema:.2f} "
            f"oim={order_imbalance:.2f} "
            f"size={dynamic_quote_size} "
            f"bid={bid_price}x{bid_quantity} "
            f"ask={ask_price}x{ask_quantity}"
        )

        data = self._save_hydrogel_state(
            data=data,
            hg_fair_ema=hg_fair_ema,
            raw_mid=raw_mid,
            hg_vol_ema=hg_vol_ema,
            hg_stress_ema=hg_stress_ema,
        )

        return orders, data

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)

        result: Dict[Symbol, List[Order]] = {}

        if self.HG_PRODUCT in state.order_depths:
            hydrogel_orders, data = self._trade_hydrogel(state, data)
            result[self.HG_PRODUCT] = hydrogel_orders

        # Explicitly keep all other Round 3 products disabled for now.
        for product in self.DISABLED_PRODUCTS:
            if product in state.order_depths:
                result[product] = []

        trader_data = jsonpickle.encode(data)

        conversions = 0

        logger.flush(
            state=state,
            orders=result,
            conversions=conversions,
            trader_data=trader_data,
        )

        return result, conversions, trader_data