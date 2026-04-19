import json
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    log_file = '/Users/vedant/Quant/Prosperity4/imc-prosperity-4/hold_1_unit/296479.log'
    print(f"Loading {log_file}...")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
        
    csv_data = data['activitiesLog']
    df = pd.read_csv(io.StringIO(csv_data), sep=';')
    
    osmium = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    osmium.sort_values(by='timestamp', inplace=True)
    
    # At t=0, we hit the best ask. Looking at the log, the first ask at t=0 was 10016.
    # From t=100 onwards, PnL reflects that purchase.
    buy_price = 10016.0
    
    # Calculate True Continuous FV
    osmium['true_fv'] = np.where(osmium['timestamp'] >= 100, osmium['profit_and_loss'] + buy_price, np.nan)
    
    # Drop t=0 where we don't have PnL on the position yet
    osmium = osmium.dropna(subset=['true_fv'])
    
    fv = osmium['true_fv'].values
    
    mean_fv = np.mean(fv)
    std_fv = np.std(fv)
    print(f"=== TRUE FV STATISTICS ===")
    print(f"Mean FV: {mean_fv:.4f}")
    print(f"Std Dev FV: {std_fv:.4f}")
    
    # Autocorrelation (lag 1)
    fv_shifted = fv[:-1]
    fv_current = fv[1:]
    lag1_corr = np.corrcoef(fv_shifted, fv_current)[0, 1]
    print(f"Lag-1 Autocorrelation: {lag1_corr:.4f}")
    
    # Variance Ratio (lag 10) for mean reversion check
    lag10_corr = np.corrcoef(fv[:-10], fv[10:])[0, 1]
    print(f"Lag-10 Autocorrelation: {lag10_corr:.4f}")
    
    # ACF plotting
    plt.figure(figsize=(10, 6))
    plt.plot(osmium['timestamp'], fv, label='Continuous True FV', color='blue', alpha=0.7)
    plt.plot(osmium['timestamp'], osmium['mid_price'], label='Quantized Mid Price', color='red', alpha=0.5, linestyle='--')
    plt.axhline(10000, color='black', linestyle=':', label='Theoretical Mean (10000)')
    plt.title('ASH_COATED_OSMIUM: Continuous True FV vs Mid Price')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('fv_analysis.png')
    plt.close()
    print("Plot saved to fv_analysis.png")

if __name__ == "__main__":
    main()
