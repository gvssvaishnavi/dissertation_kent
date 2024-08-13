import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
bank_data = pd.read_excel('prepared_data.xlsx')
gdp_data = pd.read_excel('cleaned_gdp_data.xlsx')
inf_data = pd.read_excel('cleaned_inf_data.xlsx')
gpr_data = pd.read_excel('cleaned_gpr_data.xlsx')

# Merge bank data with GDP data
merged_data = pd.merge(bank_data, gdp_data, on=['Year'], how='left')

# Merge the resulting data with INF data
merged_data = pd.merge(merged_data, inf_data, on=['Year'], how='left')

# Merge the resulting data with GPR data
final_data = pd.merge(merged_data, gpr_data, on=['Year'], how='left')

# Save the merged data
final_data.to_excel('final_merged_data.xlsx', index=False)

# Compute the logarithm of necessary variables
log_vars = ['Total Assets EUR', 'Equity to Assets Ratio', 'Loan Ratio', 'LLP', 'Deposit Ratio']
for var in log_vars:
    # Filter out non-positive values before applying log transformation
    valid_indices = final_data[var] > 0
    final_data.loc[valid_indices, f'Ln_{var}'] = np.log(final_data.loc[valid_indices, var])

# Replace infinite values resulting from log transformation
for var in log_vars:
    final_data[f'Ln_{var}'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Check the columns in the final_data DataFrame
print(final_data.columns)

# List of all relevant variables including the dependent variables
all_vars = [f'Ln_{var}' for var in log_vars] + ['GPR', 'GDP', 'INF', 'Z-Score', 'NPL', 'Standard Deviation of ROAA']

# Drop rows with any NaN values in the independent variables or dependent variables
final_data = final_data.dropna(subset=all_vars)

# Save the transformed data
final_data.to_csv('transformed_data.csv', index=False)

# Define the dependent and independent variables
dependent_vars = ['Z-Score', 'Standard Deviation of ROAA', 'NPL']
independent_vars = ['GPR', 'Ln_Total Assets EUR', 'Ln_Equity to Assets Ratio', 'Ln_Loan Ratio', 'Ln_LLP', 'Ln_Deposit Ratio', 'GDP', 'INF']

# Run regressions
results = {}
for dep_var in dependent_vars:
    if dep_var in final_data.columns:  # Check if the dependent variable exists
        X = final_data[independent_vars]
        y = final_data[dep_var]
        X = sm.add_constant(X)  # add constant term
        model = sm.OLS(y, X).fit()
        results[dep_var] = model.summary()

# Display regression results
for dep_var, result in results.items():
    print(f'Results for {dep_var}:')
    print(result)
    print('\n')

# Save regression results to files
for dep_var, result in results.items():
    with open(f'regression_results_{dep_var}.txt', 'w') as file:
        file.write(str(result))

# Subsample analysis based on profitability
# Define the median ROA and ROE to split the dataset
median_roa = final_data['ROAA'].median()
median_roe = final_data['ROAE'].median()

# Split the data based on median ROA
low_roa = final_data[final_data['ROAA'] < median_roa]
high_roa = final_data[final_data['ROAA'] >= median_roa]

# Split the data based on median ROE
low_roe = final_data[final_data['ROAE'] < median_roe]
high_roe = final_data[final_data['ROAE'] >= median_roe]

# Define a function to run regression on subsamples
def run_regression(subsample, dep_vars, indep_vars):
    subsample_results = {}
    for dep_var in dep_vars:
        if dep_var in subsample.columns:  # Check if the dependent variable exists in the subsample
            X = subsample[indep_vars]
            y = subsample[dep_var]
            X = sm.add_constant(X)  # add constant term
            model = sm.OLS(y, X).fit()
            subsample_results[dep_var] = model.summary()
    return subsample_results

# Run regression for low and high ROA subsamples
low_roa_results = run_regression(low_roa, dependent_vars, independent_vars)
high_roa_results = run_regression(high_roa, dependent_vars, independent_vars)

# Run regression for low and high ROE subsamples
low_roe_results = run_regression(low_roe, dependent_vars, independent_vars)
high_roe_results = run_regression(high_roe, dependent_vars, independent_vars)

# Save subsample regression results to files
for dep_var, result in low_roa_results.items():
    with open(f'low_roa_regression_results_{dep_var}.txt', 'w') as file:
        file.write(str(result))

for dep_var, result in high_roa_results.items():
    with open(f'high_roa_regression_results_{dep_var}.txt', 'w') as file:
        file.write(str(result))

for dep_var, result in low_roe_results.items():
    with open(f'low_roe_regression_results_{dep_var}.txt', 'w') as file:
        file.write(str(result))

for dep_var, result in high_roe_results.items():
    with open(f'high_roe_regression_results_{dep_var}.txt', 'w') as file:
        file.write(str(result))

plt.switch_backend('agg')  # Use 'agg' backend to avoid display issues

# (Assume the code before plotting is unchanged)

# Create ROAA and ROAE groups explicitly
final_data['ROAA Group'] = np.where(final_data['ROAA'] >= median_roa, 'High', 'Low')
final_data['ROAE Group'] = np.where(final_data['ROAE'] >= median_roe, 'High', 'Low')

# Plotting

# Histogram of Z-Scores
plt.figure(figsize=(10, 6))
sns.histplot(final_data['Z-Score'].dropna(), bins=30, kde=True)
plt.title('Distribution of Z-Scores')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.savefig('zscore_histogram.png')

# Scatter Plot: Z-Score vs. GPR
plt.figure(figsize=(10, 6))
sns.scatterplot(data=final_data, x='GPR', y='Z-Score')
plt.title('Z-Score vs. Geopolitical Risk (GPR)')
plt.xlabel('Geopolitical Risk (GPR)')
plt.ylabel('Z-Score')
plt.savefig('zscore_vs_gpr_scatter.png')

# Box Plot: Z-Score by High/Low ROAA
plt.figure(figsize=(10, 6))
sns.boxplot(x='ROAA Group', y='Z-Score', data=final_data)
plt.title('Z-Score by ROAA Group')
plt.xlabel('ROAA Group (High/Low)')
plt.ylabel('Z-Score')
plt.savefig('zscore_by_roaa_boxplot.png')

# Box Plot: Z-Score by High/Low ROAE
plt.figure(figsize=(10, 6))
sns.boxplot(x='ROAE Group', y='Z-Score', data=final_data)
plt.title('Z-Score by ROAE Group')
plt.xlabel('ROAE Group (High/Low)')
plt.ylabel('Z-Score')
plt.savefig('zscore_by_roe_boxplot.png')

# Heatmap of Correlation Matrix
plt.figure(figsize=(12, 8))
corr_matrix = final_data[all_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix_heatmap.png')

# Save and close all figures
plt.close('all')

# Calculate the standard deviation of GPR
gpr_std = final_data['GPR'].std()

# Create a new variable for a one standard deviation change in GPR
final_data['GPR_Std'] = final_data['GPR'] / gpr_std



# Define the dependent variables
dependent_vars = ['Z-Score', 'ROAA', 'NPL', 'ROAE']  # Example variables

# Independent variables including the standardized GPR
independent_vars = ['GPR_Std', 'Ln_Total Assets EUR', 'Ln_Equity to Assets Ratio', 'Ln_Loan Ratio', 'Ln_LLP', 'Ln_Deposit Ratio', 'GDP', 'INF']

# Run the regressions
results = {}
for dep_var in dependent_vars:
    X = final_data[independent_vars]
    y = final_data[dep_var]
    X = sm.add_constant(X)  # add constant term
    model = sm.OLS(y, X).fit()
    results[dep_var] = model.summary()

# Display regression results
for dep_var, result in results.items():
    print(f'Results for {dep_var}:')
    print(result)
    print('\n')
