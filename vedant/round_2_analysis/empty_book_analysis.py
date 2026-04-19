import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Setup paths
repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "data" / "ROUND_2"
out_dir = repo_root / "vedant" / "round_2_analysis"
out_dir.mkdir(parents=True, exist_ok=True)

# We want to also drop these in the gemini artifacts folder so they can be embedded in the walkthrough.
gemini_artifacts_dir = Path("/Users/vedant/.gemini/antigravity/brain/40238af2-ca9f-40a5-b7dc-1bb1d5b3871c/artifacts")
gemini_artifacts_dir.mkdir(parents=True, exist_ok=True)

days = ["-1", "0", "1"]

for day in days:
    csv_path = data_dir / f"prices_round_2_day_{day}.csv"
    if not csv_path.exists():
        print(f"Skipping day {day}: {csv_path} not found.")
        continue
        
    df = pd.read_csv(csv_path, sep=";")
    df = df[df["product"] == "ASH_COATED_OSMIUM"].copy()
    
    # Sort by timestamp
    df = df.sort_values("timestamp")
    
    pd.set_option('mode.chained_assignment', None)
    
    # Calculate L2 mid price safely
    df["l2_mid"] = (df["bid_price_2"] + df["ask_price_2"]) / 2.0
    
    # Identify empty book events. If volume is NaN (or 0), the side is effectively empty.
    df["empty_bid"] = df["bid_volume_1"].isna() | (df["bid_volume_1"] == 0)
    df["empty_ask"] = df["ask_volume_1"].isna() | (df["ask_volume_1"] == 0)
    df["empty_any"] = df["empty_bid"] | df["empty_ask"]
    
    # Create an interpolated dataframe for seamless line plotting and dot placement
    timeseries_cols = ["mid_price", "bid_price_1", "ask_price_1", "bid_price_2", "ask_price_2", "l2_mid"]
    df_interp = df.copy()
    import numpy as np
    for col in timeseries_cols:
        # Crucial: Replace literal zeros with NaN first so they actually get interpolated across
        df_interp[col] = df_interp[col].replace(0, np.nan)
        df_interp[col] = df_interp[col].interpolate(method='linear').bfill().ffill()
    
    # Set up the plot
    plt.figure(figsize=(24, 12))
    
    # Plot L2 first so it's in the background, using interpolated lines
    plt.plot(df_interp["timestamp"], df_interp["bid_price_2"], color="purple", label="L2 Bid", linewidth=0.5, alpha=0.3)
    plt.plot(df_interp["timestamp"], df_interp["ask_price_2"], color="purple", label="L2 Ask", linewidth=0.5, alpha=0.3)
    plt.plot(df_interp["timestamp"], df_interp["l2_mid"], color="purple", label="L2 Mid", linewidth=1.0, linestyle="--", alpha=0.6)
    
    # L1 Bids & Asks, using interpolated lines
    plt.plot(df_interp["timestamp"], df_interp["bid_price_1"], color="green", label="L1 Bid", linewidth=1.5, alpha=0.8)
    plt.plot(df_interp["timestamp"], df_interp["ask_price_1"], color="red", label="L1 Ask", linewidth=1.5, alpha=0.8)
    
    # L1 Mid
    plt.plot(df_interp["timestamp"], df_interp["mid_price"], color="lightblue", label="L1 Mid", linewidth=2.0)
    
    # Highlight Empty Book Ticks directly on the interpolated timeseries
    empty_points_interp = df_interp[df_interp["empty_any"]]
    if not empty_points_interp.empty:
        # Plot black dots directly on all existing data points at these timestamps
        for idx, col in enumerate(timeseries_cols):
            label = "Empty State (Black Dot)" if idx == 0 else ""
            plt.scatter(empty_points_interp["timestamp"], empty_points_interp[col], color="black", s=25, zorder=5, label=label)
            
    plt.title(f"ASH_COATED_OSMIUM Orderbook Extreme/Empty States - Round 2 Day {day}", fontsize=18)
    plt.xlabel("Timestamp (ms)", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    
    # Deduplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    out_file = out_dir / f"osmium_empty_book_day_{day}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    
    # Copy to artifacts dir for rendering in walkthrough
    artifact_file = gemini_artifacts_dir / f"osmium_empty_book_day_{day}.png"
    shutil.copy2(out_file, artifact_file)
    
    print(f"Day {day}: Found {len(empty_points_interp)} empty orderbook ticks. Saved plot.")

print("Analysis completely finished.")
