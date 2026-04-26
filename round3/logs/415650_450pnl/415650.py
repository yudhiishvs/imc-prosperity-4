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
    HYDROGEL_PACK wall-aware adaptive market maker.

    This version keeps the high-PnL wall-aware base structure and original
    stationarity behavior, while adding safety protections:

        1. Keep wall-aware fair value.
        2. Keep original stationarity side behavior.
        3. Add inventory-reduction override.
        4. Add danger-state logging.
        5. Add hard fail-safe protections.
        6. Do not soften every signal.

    Main rule:
        If too long:
            never allow more buying
            always allow selling

        If too short:
            never allow more selling
            always allow buying
    """

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

    # Soft limit: stop adding risk, but force inventory-reducing quote.
    HG_SOFT_POSITION_LIMIT = 140

    # Hard limit: emergency reduce inventory.
    HG_HARD_POSITION_LIMIT = 190

    # Emergency unwind target.
    HG_EMERGENCY_TARGET_POSITION = 150

    # Maximum size for one emergency unwind order.
    HG_MAX_EMERGENCY_TRADE = 30

    # Minimum passive quote size when we need to reduce inventory.
    HG_MIN_REDUCING_QUOTE_SIZE = 4

    # -------------------------------------------------------------------------
    # Fair value, volatility, and stationarity parameters
    # -------------------------------------------------------------------------

    HG_INITIAL_FV = 9900.0

    # Slow fair-value tracker.
    HG_EMA_ALPHA = 0.005

    # Faster volatility tracker.
    HG_VOL_EMA_ALPHA = 0.10

    # Stationarity stress tracker.
    HG_STRESS_EMA_ALPHA = 0.05

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
    WALL_BLEND = 0.25

    # -------------------------------------------------------------------------
    # Quoting parameters
    # -------------------------------------------------------------------------

    HG_BID_OFFSET = 1
    HG_ASK_OFFSET = 1

    HG_BASE_QUOTE_SIZE = 20
    HG_MIN_QUOTE_SIZE = 4

    # Kept unchanged from the benchmark wall-aware version.
    HG_INVENTORY_SKEW_TICKS = 6.0

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
    # Original behavior:
    #     If stress >= target, remove the bad side.
    #     If stress >= kill, stop fresh risk and only reduce inventory.
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
        if not orders:
            return None, 0, 0.0

        levels: List[Tuple[int, int]] = []

        for price, volume in orders.items():
            visible_volume = abs(volume) if is_sell_side else volume

            if price <= 0 or visible_volume <= 0:
                continue

            levels.append((price, visible_volume))

        if len(levels) < 2:
            return None, 0, 0.0

        avg_volume = sum(volume for _, volume in levels) / len(levels)

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
        bid_wall_price, _, bid_wall_strength = self._largest_qualified_wall(
            orders=depth.buy_orders,
            is_sell_side=False,
        )

        ask_wall_price, _, ask_wall_strength = self._largest_qualified_wall(
            orders=depth.sell_orders,
            is_sell_side=True,
        )

        if bid_wall_price is not None and ask_wall_price is not None:
            if bid_wall_price < ask_wall_price:
                return (
                    (bid_wall_price + ask_wall_price) / 2.0,
                    "BOTH",
                    bid_wall_price,
                    ask_wall_price,
                    bid_wall_strength,
                    ask_wall_strength,
                )

        if bid_wall_price is not None:
            return (
                max(base_fair, float(bid_wall_price)),
                "BID_ONLY",
                bid_wall_price,
                None,
                bid_wall_strength,
                0.0,
            )

        if ask_wall_price is not None:
            return (
                min(base_fair, float(ask_wall_price)),
                "ASK_ONLY",
                None,
                ask_wall_price,
                0.0,
                ask_wall_strength,
            )

        return base_fair, "NONE", None, None, 0.0, 0.0

    def _apply_wall_adjustment(
        self,
        depth: OrderDepth,
        base_fair: float,
    ) -> Tuple[float, str, Optional[int], Optional[int], float, float, float]:
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
        return max(self.HG_MIN_QUOTE_SIZE, min(self.HG_BASE_QUOTE_SIZE, final_size))

    # -------------------------------------------------------------------------
    # Safety helper functions
    # -------------------------------------------------------------------------

    def _restore_reducing_sell_size(
        self,
        current_quantity: int,
        dynamic_quote_size: int,
        position: int,
        pending_sells: int,
    ) -> int:
        """
        If we are too long, we must always allow selling.
        """
        desired = max(
            current_quantity,
            self.HG_MIN_REDUCING_QUOTE_SIZE,
            dynamic_quote_size,
        )

        return min(desired, self._sell_room(position, pending_sells))

    def _restore_reducing_buy_size(
        self,
        current_quantity: int,
        dynamic_quote_size: int,
        position: int,
        pending_buys: int,
    ) -> int:
        """
        If we are too short, we must always allow buying.
        """
        desired = max(
            current_quantity,
            self.HG_MIN_REDUCING_QUOTE_SIZE,
            dynamic_quote_size,
        )

        return min(desired, self._buy_room(position, pending_buys))

    def _danger_flags(
        self,
        position: int,
        hg_vol_ema: float,
        hg_stress_ema: float,
        wall_adjustment: float,
        bid_quantity: int,
        ask_quantity: int,
    ) -> str:
        flags: List[str] = []

        if position >= self.HG_SOFT_POSITION_LIMIT:
            flags.append("SOFT_LONG")
        if position <= -self.HG_SOFT_POSITION_LIMIT:
            flags.append("SOFT_SHORT")
        if position >= self.HG_HARD_POSITION_LIMIT:
            flags.append("HARD_LONG")
        if position <= -self.HG_HARD_POSITION_LIMIT:
            flags.append("HARD_SHORT")

        if hg_vol_ema >= self.HG_KILL_VOL:
            flags.append("VOL_KILL")
        elif hg_vol_ema >= self.HG_EXTREME_VOL:
            flags.append("VOL_EXTREME")

        if hg_stress_ema >= self.HG_KILL_STRESS_Z:
            flags.append("STRESS_KILL")
        elif hg_stress_ema >= self.HG_HIGH_STRESS_Z:
            flags.append("STRESS_HIGH")
        elif hg_stress_ema >= self.HG_TARGET_STRESS_Z:
            flags.append("STRESS_TARGET")

        if abs(wall_adjustment) >= 0.95:
            flags.append("WALL_MAXISH")

        if bid_quantity == 0 and ask_quantity == 0:
            flags.append("NO_QUOTES")
        elif bid_quantity == 0:
            flags.append("BID_OFF")
        elif ask_quantity == 0:
            flags.append("ASK_OFF")

        return "|".join(flags) if flags else "OK"

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
                f"[HG_EMERGENCY_LONG] "
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
                f"[HG_EMERGENCY_SHORT] "
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
        book = self._safe_book(depth)

        if depth is None or book is None:
            return [], data

        best_bid, best_bid_volume, best_ask, best_ask_volume = book

        orders: List[Order] = []
        position = state.position.get(self.HG_PRODUCT, 0)

        pending_buys = 0
        pending_sells = 0

        raw_mid = (best_bid + best_ask) / 2.0

        # ---------------------------------------------------------------------
        # 1. Base fair value: EMA(volume-weighted mid)
        # ---------------------------------------------------------------------

        weighted_mid = self._weighted_mid_price(
            best_bid=best_bid,
            best_bid_volume=best_bid_volume,
            best_ask=best_ask,
            best_ask_volume=best_ask_volume,
        )

        hg_fair_ema = data["hg_fair_ema"]

        if hg_fair_ema is None:
            hg_fair_ema = (
                float(weighted_mid)
                if weighted_mid is not None
                else self.HG_INITIAL_FV
            )

        hg_fair_ema = self._ewma(
            previous=hg_fair_ema,
            value=weighted_mid,
            alpha=self.HG_EMA_ALPHA,
        )

        base_fair = hg_fair_ema

        # ---------------------------------------------------------------------
        # 2. Wall-aware fair value adjustment
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
        # 3. Volatility estimate
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
        # 4. Stationarity stress
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
        # 5. Order imbalance
        # ---------------------------------------------------------------------

        order_imbalance = 0.0

        if best_bid_volume + best_ask_volume > 0:
            order_imbalance = (
                best_bid_volume - best_ask_volume
            ) / (best_bid_volume + best_ask_volume)

        # ---------------------------------------------------------------------
        # 6. Inventory skew
        # ---------------------------------------------------------------------

        inventory_ratio = position / self.HG_POSITION_LIMIT
        inventory_skew = inventory_ratio * self.HG_INVENTORY_SKEW_TICKS
        fair_for_quotes = fair - inventory_skew

        bid_price = math.floor(fair_for_quotes - self.HG_BID_OFFSET)
        ask_price = math.ceil(fair_for_quotes + self.HG_ASK_OFFSET)

        # Maker-only guard.
        bid_price = min(bid_price, best_ask - 1)
        ask_price = max(ask_price, best_bid + 1)

        if bid_price >= ask_price:
            bid_price = ask_price - 1

        # Hard fail-safe: never place invalid prices.
        if bid_price <= 0 or ask_price <= 0 or bid_price >= ask_price:
            logger.print(
                f"[HG_FAILSAFE_BAD_QUOTES] "
                f"ts={state.timestamp} "
                f"bid={bid_price} "
                f"ask={ask_price} "
                f"best_bid={best_bid} "
                f"best_ask={best_ask}"
            )

            data = self._save_hydrogel_state(
                data=data,
                hg_fair_ema=hg_fair_ema,
                raw_mid=raw_mid,
                hg_vol_ema=hg_vol_ema,
                hg_stress_ema=hg_stress_ema,
            )

            return [], data

        # ---------------------------------------------------------------------
        # 7. Base size
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
        # 8. OIM quote pulling
        # ---------------------------------------------------------------------

        if order_imbalance > self.HG_OIM_THRESHOLD:
            # Bullish imbalance. Avoid selling.
            ask_quantity = 0

        elif order_imbalance < -self.HG_OIM_THRESHOLD:
            # Bearish imbalance. Avoid buying.
            bid_quantity = 0

        # ---------------------------------------------------------------------
        # 9. Original stationarity behavior
        #
        # Keep original behavior:
        #   If price is above fair and stress is high, do not buy high.
        #   If price is below fair and stress is high, do not sell low.
        #
        # This is NOT softened here.
        # ---------------------------------------------------------------------

        if hg_stress_ema >= self.HG_TARGET_STRESS_Z:
            if raw_mid > fair:
                bid_quantity = 0
            elif raw_mid < fair:
                ask_quantity = 0

        # ---------------------------------------------------------------------
        # 10. Volatility / stationarity fail-safe:
        #     only reduce inventory, never add fresh inventory.
        # ---------------------------------------------------------------------

        if hg_vol_ema >= self.HG_KILL_VOL or hg_stress_ema >= self.HG_KILL_STRESS_Z:
            if position > 0:
                # Long: only sell/reduce.
                bid_quantity = 0
                ask_quantity = self._restore_reducing_sell_size(
                    current_quantity=ask_quantity,
                    dynamic_quote_size=dynamic_quote_size,
                    position=position,
                    pending_sells=pending_sells,
                )
            elif position < 0:
                # Short: only buy/reduce.
                ask_quantity = 0
                bid_quantity = self._restore_reducing_buy_size(
                    current_quantity=bid_quantity,
                    dynamic_quote_size=dynamic_quote_size,
                    position=position,
                    pending_buys=pending_buys,
                )
            else:
                # No inventory: do not initiate new risk in kill state.
                bid_quantity = 0
                ask_quantity = 0

        # ---------------------------------------------------------------------
        # 11. Inventory-reduction override:
        #     This is the key safety patch.
        #
        # If too long:
        #     never allow more buying
        #     always allow selling
        #
        # If too short:
        #     never allow more selling
        #     always allow buying
        # ---------------------------------------------------------------------

        if position >= self.HG_SOFT_POSITION_LIMIT:
            bid_quantity = 0
            ask_quantity = self._restore_reducing_sell_size(
                current_quantity=ask_quantity,
                dynamic_quote_size=dynamic_quote_size,
                position=position,
                pending_sells=pending_sells,
            )

        if position <= -self.HG_SOFT_POSITION_LIMIT:
            ask_quantity = 0
            bid_quantity = self._restore_reducing_buy_size(
                current_quantity=bid_quantity,
                dynamic_quote_size=dynamic_quote_size,
                position=position,
                pending_buys=pending_buys,
            )

        # ---------------------------------------------------------------------
        # 12. Emergency unwind near hard limit
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

        danger_flags = self._danger_flags(
            position=position,
            hg_vol_ema=hg_vol_ema,
            hg_stress_ema=hg_stress_ema,
            wall_adjustment=wall_adjustment,
            bid_quantity=bid_quantity,
            ask_quantity=ask_quantity,
        )

        logger.print(
            f"[HG] "
            f"ts={state.timestamp} "
            f"flags={danger_flags} "
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
            f"inst_z={stationarity_z:.2f} "
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
    # Main
    # -------------------------------------------------------------------------

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[Symbol, List[Order]] = {}

        if self.HG_PRODUCT in state.order_depths:
            hydrogel_orders, data = self._trade_hydrogel(state, data)
            result[self.HG_PRODUCT] = hydrogel_orders

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