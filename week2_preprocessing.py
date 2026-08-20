import pandas as pd

# Load the logistics dataset
data = pd.read_csv("Dataset/logistics_data.csv")

# Display the first five records
print("First five records:")
print(data.head())

# Display dataset information
print("\nDataset information:")
print(data.info())

# Check the number of rows and columns
print("\nDataset shape:")
print(data.shape)

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Display basic statistical information
print("\nStatistical summary:")
print(data.describe())


# -----------------------------------------
# Missing Value Handling
# -----------------------------------------

# Create a copy for preprocessing demonstration
cleaned_data = data.copy()

# Introduce a missing value for demonstration
cleaned_data.loc[2, "Shipment_Weight_kg"] = None

print("\nMissing values before handling:")
print(cleaned_data.isnull().sum())

# Fill the missing numerical value using the median
median_weight = cleaned_data["Shipment_Weight_kg"].median()
cleaned_data["Shipment_Weight_kg"] = cleaned_data["Shipment_Weight_kg"].fillna(median_weight)

print("\nMissing values after handling:")
print(cleaned_data.isnull().sum())


# -----------------------------------------
# Outlier Detection using IQR
# -----------------------------------------

numerical_columns = [
    "Distance_km",
    "Shipment_Weight_kg",
    "Expected_Delivery_Days",
    "Delivery_Time_Days",
    "Transportation_Cost"
]

print("\nOutlier Detection using IQR:")

for column in numerical_columns:
    Q1 = cleaned_data[column].quantile(0.25)
    Q3 = cleaned_data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = cleaned_data[
        (cleaned_data[column] < lower_bound) |
        (cleaned_data[column] > upper_bound)
    ]

    print(f"\n{column}:")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    print(f"Number of Outliers: {len(outliers)}")

    # -----------------------------------------
# Min-Max Normalization
# -----------------------------------------

from sklearn.preprocessing import MinMaxScaler

# Select numerical columns for normalization
columns_to_normalize = [
    "Distance_km",
    "Shipment_Weight_kg",
    "Expected_Delivery_Days",
    "Delivery_Time_Days",
    "Transportation_Cost"
]

# Create the scaler
scaler = MinMaxScaler()

# Apply Min-Max normalization
normalized_data = cleaned_data.copy()
normalized_data[columns_to_normalize] = scaler.fit_transform(
    normalized_data[columns_to_normalize]
)

print("\nNormalized data:")
print(normalized_data[columns_to_normalize].head())


# -----------------------------------------
# Complete Preprocessing Pipeline
# -----------------------------------------

print("\n========== FINAL PREPROCESSING PIPELINE ==========")

# Start with a copy of the original dataset
final_data = data.copy()

# Step 1: Handle missing values
# Median imputation is used for the numerical column
final_data.loc[2, "Shipment_Weight_kg"] = None
final_data["Shipment_Weight_kg"] = final_data["Shipment_Weight_kg"].fillna(
    final_data["Shipment_Weight_kg"].median()
)

# Step 2: Detect outliers using IQR
for column in numerical_columns:
    Q1 = final_data[column].quantile(0.25)
    Q3 = final_data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count = (
        (final_data[column] < lower_bound) |
        (final_data[column] > upper_bound)
    ).sum()

    print(f"{column}: {outlier_count} outliers")

# Step 3: Normalize numerical variables
final_normalized_data = final_data.copy()

final_normalized_data[columns_to_normalize] = scaler.fit_transform(
    final_normalized_data[columns_to_normalize]
)

print("\nFinal normalized dataset:")
print(final_normalized_data.head())

print("\nFinal dataset shape:", final_normalized_data.shape)

print("\nFinal missing values:")
print(final_normalized_data.isnull().sum())