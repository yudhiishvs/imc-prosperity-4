import json
import csv
from io import StringIO

LOG_FILE = "Phase_2/378034/378034.log"

def main():
    print("Loading log file...")
    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    print("Keys found in log:", list(data.keys()))
    
    # Extract historical trades
    trade_history = data.get("tradeHistory", [])
    print(f"\nTotal historical trades logged: {len(trade_history)}")
    
    hg_market_trades = [t for t in trade_history if t.get("symbol") == "HYDROGEL_PACK"]
    print(f"Total HYDROGEL_PACK market trades: {len(hg_market_trades)}")

    # The lambda logs (sandboxLogs) contain our bot's standard output
    sandbox_logs = data.get("logs", {})
    # Wait, data["logs"] might be a dict or string? Let's check type.
    if isinstance(sandbox_logs, dict):
        print("\nLogs type: dict. Keys:", list(sandbox_logs.keys())[:5])
        
    elif isinstance(sandbox_logs, list):
        
        # Process activitiesLog to build a dictionary of market state per timestamp
        activities = data.get("activitiesLog", "").strip().split("\n")
        market_state = {}
        if activities and "day;timestamp" in activities[0]:
            header = activities[0].split(";")
            for line in activities[1:]:
                if not line.strip(): continue
                parts = line.split(";")
                ts = int(parts[1])
                product = parts[2]
                if product == "HYDROGEL_PACK":
                    best_bid = int(parts[3]) if parts[3] else 0
                    best_ask = int(parts[9]) if parts[9] else 0
                    market_state[ts] = {"best_bid": best_bid, "best_ask": best_ask}

        # Process market trades
        hg_market_trades_by_ts = {}
        for t in hg_market_trades:
            ts = t.get("timestamp")
            if ts not in hg_market_trades_by_ts:
                hg_market_trades_by_ts[ts] = []
            hg_market_trades_by_ts[ts].append(t)

        print(f"\nFound {len(sandbox_logs)} log entries.")
        
        matches = 0
        missed_fills = 0
        quoted_at_best_bid = 0
        quoted_inside_spread = 0

        for entry in sandbox_logs:
            timestamp = entry.get("timestamp")
            lambda_log = entry.get("lambdaLog", "")
            
            # Find our bot's print statement
            hg_line = None
            if "[HG]" in lambda_log:
                for line in lambda_log.split("\n"):
                    if "[HG]" in line:
                        hg_line = line
                        break
            
            if hg_line:
                # Format: [HG] ts=0  pos=0  fair=9900.00  oim=0.00  bid=9999x100  ask=10001x100
                try:
                    parts = hg_line.split("  ")
                    bid_part = [p for p in parts if p.startswith("bid=")][0]
                    ask_part = [p for p in parts if p.startswith("ask=")][0]
                    my_bid = int(bid_part.split("=")[1].split("x")[0])
                    my_ask = int(ask_part.split("=")[1].split("x")[0])
                    
                    state = market_state.get(timestamp, {})
                    best_bid = state.get("best_bid", 0)
                    best_ask = state.get("best_ask", 0)
                    
                    if my_bid == best_bid:
                        quoted_at_best_bid += 1
                    elif best_bid < my_bid < best_ask:
                        quoted_inside_spread += 1
                        
                    # Check if any trades occurred at our price
                    trades = hg_market_trades_by_ts.get(timestamp, [])
                    for t in trades:
                        t_price = t.get("price")
                        if t_price == my_bid or t_price == my_ask:
                            missed_fills += 1
                            # print(f"TS {timestamp}: Trade at {t_price}. My Bid: {my_bid}, My Ask: {my_ask}, Best Bid: {best_bid}")
                            
                except Exception as e:
                    pass
                        
        print(f"\nStats:")
        print(f"Times quoted EXACTLY at market best bid: {quoted_at_best_bid}")
        print(f"Times quoted INSIDE the spread (penny-jump): {quoted_inside_spread}")
        print(f"Missed Fills (trade occurred at our quote price but we weren't filled): {missed_fills}")
    else:
        print("\nLogs type:", type(sandbox_logs))
        
    print("\nParsing complete.")

if __name__ == "__main__":
    main()
