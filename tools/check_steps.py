import pandas as pd
import numpy as np

df = pd.read_csv("results/diagnostics/ua_adc/ua_adc_trace_nominal.csv")

d0 = df[df["drone_id"] == 0]
d1 = df[df["drone_id"] == 1]
d2 = df[df["drone_id"] == 2]

# Check values for step 1 and step 100
step1_d0 = d0[d0["step"] == 1].iloc[0]
step1_d1 = d1[d1["step"] == 1].iloc[0]
step1_d2 = d2[d2["step"] == 1].iloc[0]

step100_d0 = d0[d0["step"] == 100].iloc[0]
step100_d1 = d1[d1["step"] == 100].iloc[0]
step100_d2 = d2[d2["step"] == 100].iloc[0]

print("Step 1 values:")
print(f"Drone 0: P={step1_d0['P']:.6f}, c={step1_d0['confidence']:.6f}, alpha={step1_d0['blend_weight']:.6f}")
print(f"Drone 1: P={step1_d1['P']:.6f}, c={step1_d1['confidence']:.6f}, alpha={step1_d1['blend_weight']:.6f}")
print(f"Drone 2: P={step1_d2['P']:.6f}, c={step1_d2['confidence']:.6f}, alpha={step1_d2['blend_weight']:.6f}")

print("\nStep 100 values:")
print(f"Drone 0: P={step100_d0['P']:.6f}, c={step100_d0['confidence']:.6f}, alpha={step100_d0['blend_weight']:.6f}")
print(f"Drone 1: P={step100_d1['P']:.6f}, c={step100_d1['confidence']:.6f}, alpha={step100_d1['blend_weight']:.6f}")
print(f"Drone 2: P={step100_d2['P']:.6f}, c={step100_d2['confidence']:.6f}, alpha={step100_d2['blend_weight']:.6f}")
