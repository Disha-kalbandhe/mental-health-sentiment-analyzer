import pandas as pd

# Read datasets with correct column names
df1 = pd.read_csv("data/dataset1.csv")  # columns: Post, Label
df2 = pd.read_csv("data/dataset2.csv")  # columns: text, class
df3 = pd.read_csv("data/dataset3.csv")  # columns: Tweet, Suicide

# Rename necessary columns
df1 = df1[['Post', 'Label']].rename(columns={'Post': 'text', 'Label': 'label'})
df2 = df2[['text', 'class']].rename(columns={'class': 'label'})
df3 = df3[['Tweet', 'Suicide']].rename(columns={'Tweet': 'text', 'Suicide': 'label'})

# Combine all datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

# Normalize labels
df['label'] = df['label'].str.lower().str.strip()
df['label'] = df['label'].replace({
    'suicide': 'suicidal',
    'non-suicide': 'non-suicidal',
    'not suicide post': 'non-suicidal',
    'potential suicide post': 'suicidal',
    'non-suicide post': 'non-suicidal',
    'non suicide': 'non-suicidal'
})

# Drop empty or NaN rows
df = df.dropna()

# Optional: See stats
print("✅ Merged shape:", df.shape)
print(df['label'].value_counts())

# Save merged dataset
df.to_csv("data/merged_clean_dataset.csv", index=False)
print("✅ Saved to data/merged_clean_dataset.csv")
