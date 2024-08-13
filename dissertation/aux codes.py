# Load the necessary libraries
import pandas as pd

# Load the INF data file
inf_file_path = 'inf.csv'
inf_data = pd.read_csv(inf_file_path)

# Display the first few rows of the INF data to understand its structure
print(inf_data.head())

# Filter relevant rows for Eurozone countries
# Assuming we have a list of Eurozone country codes
eurozone_countries = ['AUT', 'BEL', 'CYP', 'DEU', 'ESP', 'EST', 'FIN', 'FRA', 'GRC', 'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN']
inf_data = inf_data[inf_data['Country Code'].isin(eurozone_countries)]

# Transform the data to have 'Year' and 'INF' columns
inf_data = inf_data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                         var_name='Year', value_name='INF')

# Drop unnecessary columns
inf_data = inf_data[['Year', 'Country Code', 'INF']]

# Convert 'Year' to numeric
inf_data['Year'] = pd.to_numeric(inf_data['Year'], errors='coerce')

# Handle missing values by dropping rows with NaN values
inf_data = inf_data.dropna()

# Aggregate INF data by year (taking the mean for each year)
inf_yearly = inf_data.groupby('Year')['INF'].mean().reset_index()

# Save the cleaned INF data to a new file
cleaned_inf_path = 'cleaned_inf_data.xlsx'
inf_yearly.to_excel(cleaned_inf_path, index=False)

print(f"Cleaned INF data saved to {cleaned_inf_path}")

# Display the cleaned INF data
inf_yearly.head()
