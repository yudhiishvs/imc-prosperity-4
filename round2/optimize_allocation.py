import numpy as np
from scipy.optimize import brentq
import sys

def optimize_allocation(total_budget=95.0):
    """
    Solves for the optimal balance between Research (x) and Scale (y = B-x).
    Objective: Maximize Research(x) * Scale(y)
    Equation: (B - x)/(1 + x) - ln(1 + x) = 0
    """
    B = total_budget
    g = lambda x: (B - x)/(1 + x) - np.log(1 + x)
    
    try:
        # Solve for x in the range [0, B]
        x_opt = brentq(g, 0, B)
        scale_opt = B - x_opt
        return x_opt, scale_opt
    except Exception as e:
        print(f"Error solving optimization: {e}")
        return None, None

if __name__ == "__main__":
    BUDGET_XIREC = 50000
    LN_101 = np.log(101)  # approx 4.6151
    
    print(f"{'Budget (B)':>10} | {'Res (x)':>10} | {'Scale (y)':>10} | {'Speed (z)':>10} | {'Final PnL Score (XIREC)':>30}")
    print("-" * 88)
    
    # B is the combined allocation to Research and Scale (0 to 100)
    # The remaining (100 - B) is allocated to Speed (z)
    for b in range(5, 105, 5):
        B = float(b)
        if B > 100: B = 100.0
        
        x_opt, y_opt = optimize_allocation(B)
        
        if x_opt is not None:
            # 1. Research outcome formula: 200,000 * ln(1+x) / ln(101)
            research_val = 200000 * np.log(1 + x_opt) / LN_101
            
            # 2. Scale outcome formula: 7 * y / 100
            scale_val = 7 * y_opt / 100.0
            
            # 3. Component 3: Speed z
            z_pct = 100.0 - B
            
            # Gross PnL (pre-speed-multiplier and pre-deduction)
            gross_pre_mult = research_val * scale_val
            
            # PnL = (Research * Scale * Speed_Multiplier) - Budget_Used
            # We treat the budget used as the full 50,000 since we allocate 100% across the three pillars.
            # Speed_Multiplier is unknown (rank-based), so we note it as 's' or 'z_mult'.
            
            if gross_pre_mult > 0:
                pnl_str = f"({gross_pre_mult:,.0f} * s) - 50,000"
            else:
                pnl_str = "0 - 50,000"
                
            print(f"{B:>9.1f}% | {x_opt:>9.2f}% | {y_opt:>9.2f}% | {z_pct:>9.1f}% | {pnl_str:>30}")
        
    print("-" * 88)
    print("Note: 's' is the rank-based Speed multiplier (0.1 to 0.9).")
    print("Optimization maximizes Research * Scale for the available non-Speed budget.")
