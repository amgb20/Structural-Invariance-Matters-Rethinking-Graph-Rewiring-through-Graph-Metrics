import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Display the structured data
import ace_tools_open as tools

# Load the CSV file
file_path = "test_imdb.csv"
df = pd.read_csv(file_path, index_col=0)

# Extract metric names from the second column
metrics = df.iloc[:, 0].values  # Assuming metric names are in the first column

# Function to extract mean and std from "mean ± std" format
def extract_mean_std(value):
    match = re.match(r"([-\d.]+)\s*±\s*([-\d.]+)", str(value))
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        return None, None  # Return None if format doesn't match

# Create separate DataFrames for mean and std
mean_df = pd.DataFrame(index=metrics)
std_df = pd.DataFrame(index=metrics)

# Extract mean and std values for each rewiring method
for column in df.columns[1:]:  # Skip first column (metrics)
    mean_df[column], std_df[column] = zip(*df[column].apply(extract_mean_std))

# Apply logarithm transformation (handling zeros by adding a small constant)
log_mean_df = np.log10(mean_df.replace(0, np.nan))  # Replace zeros with NaN to avoid log(0)

tools.display_dataframe_to_user(name="Mean Values", dataframe=mean_df)

# Plot bar chart for mean values
plt.figure(figsize=(12, 6))
log_mean_df.plot(kind='bar', figsize=(12, 6), alpha=0.75, edgecolor='black')
plt.title("Logarithmic Comparison of Mean Values Across Rewiring Methods")
plt.ylabel("Log Mean Value")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Rewiring Methods")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
