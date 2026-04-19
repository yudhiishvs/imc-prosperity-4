import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_day(day):
    file_path = f'/Users/vedant/Quant/Prosperity4/imc-prosperity-4/data/ROUND_2/prices_round_2_day_{day}.csv'
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    # OS permissions issue workaround for matplotlib
    os.environ['MPLCONFIGDIR'] = os.getcwd() + '/.matplotlib_cache'
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

    df = pd.read_csv(file_path, sep=';')
    
    osmium = df[df['product'] == 'ASH_COATED_OSMIUM']
    pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT']
    
    if osmium.empty and pepper.empty:
        print(f"No data for Osmium or Pepper in day {day}.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Osmium Mid Price', color=color)
    ax1.plot(osmium['timestamp'], osmium['mid_price'], color=color, label='ASH_COATED_OSMIUM')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Pepper Root Mid Price', color=color)
    ax2.plot(pepper['timestamp'], pepper['mid_price'], color=color, label='INTARIAN_PEPPER_ROOT')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Mid Prices - Round 2 Day {day}')
    fig.tight_layout()
    
    output_dir = '/Users/vedant/Quant/Prosperity4/imc-prosperity-4/vedant/plots'
    os.makedirs(output_dir, exist_ok=True)
    out_file = f'{output_dir}/mid_prices_day_{day}.png'
    plt.savefig(out_file)
    plt.close()
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    days = [-1, 0, 1]
    for d in days:
        plot_day(d)
