import pandas as pd

# Load your results
df = pd.read_csv('eeg_results_summary.csv')

# 1. Define your food categories
# (Make sure these names exactly match the names in your CSV)
liquid_foods = ['the_vert', 'coke', 'lait', 'milo'] 
solid_foods = ['bonbon_acide', 'chocolat', 'chocolat_menthe', 'algues']

# 2. Isolate the baselines
water_baseline = df[df['Food'] == 'eau'].set_index('Participant')[['FAA_Happiness', 'Calmness_Ratio']]
water_baseline = water_baseline.rename(columns={'FAA_Happiness': 'Baseline_FAA', 'Calmness_Ratio': 'Baseline_Calmness'})

cracker_baseline = df[df['Food'] == 'cracker'].set_index('Participant')[['FAA_Happiness', 'Calmness_Ratio']]
cracker_baseline = cracker_baseline.rename(columns={'FAA_Happiness': 'Baseline_FAA', 'Calmness_Ratio': 'Baseline_Calmness'})

# 3. Split the main dataframe into liquids and solids
# We include the baseline itself in the list so it gets processed too
df_liquids = df[df['Food'].isin(liquid_foods + ['eau'])].copy()
df_solids = df[df['Food'].isin(solid_foods + ['cracker'])].copy()

# 4. Merge the correct baseline into each dataframe
df_liquids = df_liquids.merge(water_baseline, on='Participant', how='left')
df_solids = df_solids.merge(cracker_baseline, on='Participant', how='left')

# 5. Calculate Delta (Change from baseline)
df_liquids['Delta_FAA'] = df_liquids['FAA_Happiness'] - df_liquids['Baseline_FAA']
df_liquids['Delta_Calmness'] = df_liquids['Calmness_Ratio'] - df_liquids['Baseline_Calmness']

df_solids['Delta_FAA'] = df_solids['FAA_Happiness'] - df_solids['Baseline_FAA']
df_solids['Delta_Calmness'] = df_solids['Calmness_Ratio'] - df_solids['Baseline_Calmness']

# Set the baseline foods' Delta to 0 (since water - water = 0)
df_liquids.loc[df_liquids['Food'] == 'eau', ['Delta_FAA', 'Delta_Calmness']] = 0
df_solids.loc[df_solids['Food'] == 'cracker', ['Delta_FAA', 'Delta_Calmness']] = 0

# 6. Combine the data back together
df_corrected = pd.concat([df_liquids, df_solids], ignore_index=True)

# 7. Calculate Group Averages (Excluding the baselines from the final average)
# We only want to average the actual target foods
foods_to_average = liquid_foods + solid_foods
df_analysis = df_corrected[df_corrected['Food'].isin(foods_to_average)]

print("\n--- TRUE GROUP AVERAGES (Relative to Baselines) ---")
grouped = df_analysis.groupby('Food')[['Delta_FAA', 'Delta_Calmness']].mean()
# Sort by Delta_Calmness descending (most calming at the top)
print(grouped.sort_values(by='Delta_Calmness', ascending=False))

# Save the final corrected data
df_corrected.to_csv('eeg_corrected_dual_baseline.csv', index=False)
print("\nCorrected data saved to 'eeg_corrected_dual_baseline.csv'")