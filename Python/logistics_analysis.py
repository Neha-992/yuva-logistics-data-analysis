import pandas as pd

# Load the logistics dataset
data = pd.read_csv("Dataset/logistics_data.csv")

# Display the first five records
print("First five records:")
print(data.head())

# Display information about the dataset
print("\nDataset information:")
print(data.info())

# Calculate average delivery time
average_delivery_time = data["Delivery_Time_Days"].mean()

# Calculate average transportation cost
average_transportation_cost = data["Transportation_Cost"].mean()

# Calculate average delivery distance
average_delivery_distance = data["Distance_km"].mean()

print("\nAverage Delivery Time:", average_delivery_time)
print("Average Transportation Cost:", average_transportation_cost)
print("Average Delivery Distance:", average_delivery_distance)

# Calculate On-Time Delivery Rate
on_time_deliveries = (data["On_Time"] == "Yes").sum()
total_deliveries = len(data)

on_time_delivery_rate = (on_time_deliveries / total_deliveries) * 100

print("On-Time Delivery Rate:", on_time_delivery_rate, "%")

# Calculate Delivery Delay Rate
delayed_deliveries = (data["On_Time"] == "No").sum()

delivery_delay_rate = (delayed_deliveries / total_deliveries) * 100

print("Delivery Delay Rate:", delivery_delay_rate, "%")

import matplotlib.pyplot as plt

# Create a chart showing on-time and delayed deliveries
delivery_status = data["On_Time"].value_counts()

delivery_status.plot(kind="bar")

plt.title("On-Time vs Delayed Deliveries")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Shipments")

plt.tight_layout()
plt.show()