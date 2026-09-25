import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# WEEK 3 - ADVANCED DATA ANALYSIS
# -----------------------------------------

# Load logistics dataset
data = pd.read_csv("Dataset/logistics_data.csv")

print("First five records:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
print(data.info())

print("\nStatistical summary:")
print(data.describe())


# -----------------------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------------------

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# Central tendency measures
print("\nMean:")
print(data[
    [
        "Distance_km",
        "Shipment_Weight_kg",
        "Expected_Delivery_Days",
        "Delivery_Time_Days",
        "Transportation_Cost"
    ]
].mean())

print("\nMedian:")
print(data[
    [
        "Distance_km",
        "Shipment_Weight_kg",
        "Expected_Delivery_Days",
        "Delivery_Time_Days",
        "Transportation_Cost"
    ]
].median())

print("\nMode:")
print(data[
    [
        "Distance_km",
        "Shipment_Weight_kg",
        "Expected_Delivery_Days",
        "Delivery_Time_Days",
        "Transportation_Cost"
    ]
].mode().iloc[0])

# Distribution of categorical variables
print("\nVehicle Type Distribution:")
print(data["Vehicle_Type"].value_counts())

print("\nOn-Time Delivery Distribution:")
print(data["On_Time"].value_counts())

# Correlation analysis
print("\nCorrelation Matrix:")
correlation_matrix = data[
    [
        "Distance_km",
        "Shipment_Weight_kg",
        "Expected_Delivery_Days",
        "Delivery_Time_Days",
        "Transportation_Cost"
    ]
].corr()

print(correlation_matrix)

# -----------------------------------------
# Visualization 1: Delivery Time Distribution
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    data["Delivery_Time_Days"],
    bins=5,
    edgecolor="black"
)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")

plt.tight_layout()
plt.show()


# -----------------------------------------
# Visualization 2: Transportation Cost Distribution
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    data["Transportation_Cost"],
    bins=6,
    edgecolor="black"
)

plt.title("Distribution of Transportation Cost")
plt.xlabel("Transportation Cost (₹)")
plt.ylabel("Number of Shipments")

plt.tight_layout()
plt.show()

# -----------------------------------------
# Visualization 3: Vehicle Type Distribution
# -----------------------------------------

vehicle_counts = data["Vehicle_Type"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    vehicle_counts.index,
    vehicle_counts.values,
    edgecolor="black"
)

plt.title("Distribution of Vehicle Types")
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Shipments")

plt.tight_layout()
plt.show()

# -----------------------------------------
# Visualization 4: On-Time Delivery
# -----------------------------------------

on_time_counts = data["On_Time"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    on_time_counts.index,
    on_time_counts.values,
    edgecolor="black"
)

plt.title("On-Time vs Delayed Shipments")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Shipments")

plt.tight_layout()
plt.show()

# -----------------------------------------
# Visualization 5: Correlation Heatmap
# -----------------------------------------

correlation_matrix = data[
    [
        "Distance_km",
        "Shipment_Weight_kg",
        "Expected_Delivery_Days",
        "Delivery_Time_Days",
        "Transportation_Cost"
    ]
].corr()

plt.figure(figsize=(10, 7))

plt.imshow(correlation_matrix, cmap="coolwarm")

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# -----------------------------------------
# Visualization 6: Delivery Time vs Cost
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Delivery_Time_Days"],
    data["Transportation_Cost"]
)

plt.title("Delivery Time vs Transportation Cost")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Transportation Cost (₹)")

plt.tight_layout()
plt.show()

# -----------------------------------------
# Visualization 7: Distance vs Transportation Cost
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Distance_km"],
    data["Transportation_Cost"]
)

plt.title("Distance vs Transportation Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost (₹)")

plt.tight_layout()
plt.show()

# -----------------------------------------
# Visualization 8: Expected vs Actual Delivery
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Expected_Delivery_Days"],
    data["Delivery_Time_Days"]
)

plt.title("Expected Delivery Time vs Actual Delivery Time")
plt.xlabel("Expected Delivery Time (Days)")
plt.ylabel("Actual Delivery Time (Days)")

plt.tight_layout()
plt.show()