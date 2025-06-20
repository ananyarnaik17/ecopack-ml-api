import pandas as pd

# Load the dataset
file_path = 'final_dim_extract_data.csv'
df = pd.read_csv(file_path)

# Select relevant columns
selected_columns = ['ASIN', 'PT', 'height cm', 'width cm', 'length cm']
df = df[selected_columns]

# Drop rows where any of the dimension fields are missing
df_cleaned = df.dropna(subset=['height cm', 'width cm', 'length cm'])

# Reset index
df_cleaned.reset_index(drop=True, inplace=True)

# Show cleaned dataset details
print("Cleaned Dataset Shape:", df_cleaned.shape)
print("Sample Cleaned Records:")
print(df_cleaned.head())

# Save the cleaned dataset for ML model
df_cleaned.to_csv('cleaned_packaging_data.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_packaging_data.csv'")
