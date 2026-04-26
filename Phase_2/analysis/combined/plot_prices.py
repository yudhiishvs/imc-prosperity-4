import pandas as pd
import matplotlib.pyplot as plt
import os

def create_plot():
    # Define file paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "../../data/ROUND_3")
    files = [
        "prices_round_3_day_0.csv",
        "prices_round_3_day_1.csv",
        "prices_round_3_day_2.csv"
    ]

    all_data = []

    print("Loading data...")
    for i, f in enumerate(files):
        path = os.path.join(base_path, f)
        if os.path.exists(path):
            print(f"Reading {path}...")
            df = pd.read_csv(path, sep=';')
            # Adjust timestamp for continuity (1,000,000 ticks per day)
            df['timestamp'] = df['timestamp'] + i * 1000000
            all_data.append(df)
        else:
            print(f"Warning: {path} not found.")

    if not all_data:
        print("No data found to plot. Please ensure the CSV files are in the 'data/ROUND_3' directory.")
        return

    print("Combining data...")
    combined_df = pd.concat(all_data)

    # Pivot the data: timestamp as index, products as columns, mid_price as values
    print("Pivoting data for plotting...")
    plot_df = combined_df.pivot(index='timestamp', columns='product', values='mid_price')

    # Plot 1: HYDROGEL_PACK
    print("Generating Hydrogel Pack graph...")
    plt.figure(figsize=(16, 9))
    if 'HYDROGEL_PACK' in plot_df.columns:
        plt.plot(plot_df.index, plot_df['HYDROGEL_PACK'], label='HYDROGEL_PACK', linewidth=1.5, color='blue')
        
        # Logically chosen bounds: min/max with a small buffer
        y_min = plot_df['HYDROGEL_PACK'].min() * 0.995
        y_max = plot_df['HYDROGEL_PACK'].max() * 1.005
        plt.ylim(y_min, y_max)
        
        plt.title('Hydrogel Pack Price - Round 3 (Days 0, 1, 2)', fontsize=16)
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel('Mid Price', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(script_dir, "../../plots/hydrogel/hydrogel_pack_price.png")
        plt.savefig(save_path, dpi=300)
        print(f"Hydrogel Pack graph saved as {save_path}")
    else:
        print("HYDROGEL_PACK not found in data.")

    # Plot 2: VELVETFRUIT_EXTRACT and its options (VEV_*)
    print("Generating Velvetfruit Extract and Options graph...")
    plt.figure(figsize=(16, 9))
    
    option_products = [p for p in plot_df.columns if p.startswith('VEV_') or p == 'VELVETFRUIT_EXTRACT']
    
    if option_products:
        # Use two y-axes if necessary, but the user asked for one graph.
        # However, Velvetfruit is ~5000 and options are ~0-1200.
        # I'll use a log scale or a secondary y-axis if it's too disparate.
        # Let's try plotting them together first; if disparate, I'll use subplots.
        
        fig, ax1 = plt.subplots(figsize=(16, 9))
        
        # Primary axis for Velvetfruit Extract
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Velvetfruit Extract Price', color='tab:red')
        ax1.plot(plot_df.index, plot_df['VELVETFRUIT_EXTRACT'], label='VELVETFRUIT_EXTRACT', color='tab:red', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        # Secondary axis for options
        ax2 = ax1.twinx()
        ax2.set_ylabel('Options Price (VEV_*)', color='tab:blue')
        
        for p in [p for p in option_products if p.startswith('VEV_')]:
            ax2.plot(plot_df.index, plot_df[p], label=p, alpha=0.7)
            
        plt.title('Velvetfruit Extract and Options - Round 3 (Days 0, 1, 2)', fontsize=16)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(script_dir, "../../plots/velvetfruit/velvetfruit_options_combined.png")
        plt.savefig(save_path, dpi=300)
        print(f"Velvetfruit and Options graph saved as {save_path}")
    else:
        print("Velvetfruit or Options not found in data.")

if __name__ == "__main__":
    create_plot()
