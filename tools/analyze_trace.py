import pandas as pd
import numpy as np

df = pd.read_csv("results/diagnostics/ua_adc_trace_nominal.csv")

print("="*80)
print("ANALYSIS OF NOMINAL TRACE")
print("="*80)

# Check per drone: drone 0 is top drone (no downwash above it), drone 1 is middle, drone 2 is bottom
for d in [0, 1, 2]:
    sub = df[df["drone_id"] == d].copy()
    print(f"\n--- DRONE {d} ---")
    
    # a. Sign check
    # Check where F_true, F_assumed, F_hat have same sign
    # Note: for drone 0, forces may be near 0
    # For drones 1 and 2, non-zero downwash exists
    signs_agree = (np.sign(sub["F_hat"]) == np.sign(sub["F_true"])) & (np.sign(sub["F_hat"]) == np.sign(sub["F_assumed"]))
    frac_agree = signs_agree.mean()
    disagree_steps = sub[~signs_agree]["step"].tolist()
    first_disagree = disagree_steps[0] if disagree_steps else None
    print(f"a. Fraction of steps where signs of (F_hat, F_true, F_assumed) agree: {frac_agree:.4f} ({signs_agree.sum()}/{len(sub)})")
    print(f"   First step where they disagree: {first_disagree}")
    if first_disagree is not None:
        row_dis = sub[sub["step"] == first_disagree].iloc[0]
        print(f"   At step {first_disagree}: F_hat={row_dis['F_hat']:.6f}, F_true={row_dis['F_true']:.6f}, F_assumed={row_dis['F_assumed']:.6f}")
        
    # b. Magnitude check at t = 0, 1, 2, 5, 10
    print("b. Magnitude check at t in [0, 1, 2, 5, 10] s:")
    for t_target in [0.0, 1.0, 2.0, 5.0, 10.0]:
        # closest step
        row = sub.iloc[(sub["t"] - t_target).abs().argsort()[:1]].iloc[0]
        print(f"   t={row['t']:5.2f}s (step {int(row['step']):3d}): F_hat = {row['F_hat']:10.6f} N, F_true = {row['F_true']:10.6f} N, F_assumed = {row['F_assumed']:10.6f} N, theta = {row['theta']:8.4f}, P = {row['P']:10.4f}, conf = {row['confidence']:.4f}, alpha = {row['blend_weight']:.5f}")

    # c. Residual check
    mean_res = sub["residual"].mean()
    std_res = sub["residual"].std()
    print(f"c. Residual: mean = {mean_res:10.6f} m/s^2, std = {std_res:10.6f} m/s^2, min = {sub['residual'].min():10.6f}, max = {sub['residual'].max():10.6f}")

print("\n" + "="*80)
print("OVERALL SUMMARY FOR DRONES 1 & 2 (INGESTING DOWNWASH)")
print("="*80)
sub12 = df[df["drone_id"].isin([1, 2])]
signs_agree12 = (np.sign(sub12["F_hat"]) == np.sign(sub12["F_true"])) & (np.sign(sub12["F_hat"]) == np.sign(sub12["F_assumed"]))
print(f"Combined Drones 1 & 2 sign agreement: {signs_agree12.mean():.4f}")
print(f"Combined Drones 1 & 2 residual mean: {sub12['residual'].mean():.6f} m/s^2, std: {sub12['residual'].std():.6f} m/s^2")

