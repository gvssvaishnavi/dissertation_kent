import pandas as pd

# Load the GPR data
gpr_data = pd.read_excel('data_gpr_export.xls')  # Replace with the actual file path

# Display the first few rows to understand the structure
print(gpr_data.head())

# Convert the 'month' column to datetime format
gpr_data['month'] = pd.to_datetime(gpr_data['month'], format='%m-%d-%Y')

# Set the 'month' column as the index
gpr_data.set_index('month', inplace=True)

# Select only the numeric columns for aggregation
numeric_columns = gpr_data.select_dtypes(include='number').columns
gpr_data_numeric = gpr_data[numeric_columns]

# Resample the data to annual frequency and take the mean of each year
gpr_annual = gpr_data_numeric.resample('A').mean()

# Reset the index to get 'Year' as a column
gpr_annual.reset_index(inplace=True)
gpr_annual['Year'] = gpr_annual['month'].dt.year

# Drop the original 'month' column
gpr_annual.drop(columns=['month'], inplace=True)

# Select relevant columns for your analysis
# Example: Assuming 'GPR', 'GPRT', and 'GPRA' are relevant for your analysis
gpr_annual = gpr_annual[['Year', 'GPR', 'GPRT', 'GPRA']]

# Save the cleaned and aggregated GPR data
gpr_annual.to_csv('cleaned_gpr_data_annual.csv', index=False)

print("Cleaned and aggregated GPR data saved to 'cleaned_gpr_data_annual.csv'.")
