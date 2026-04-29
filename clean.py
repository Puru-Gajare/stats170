import pandas as pd
import matplotlib.pyplot as plt

#load and clean
df = pd.read_csv("housing.csv")
df = df.dropna()
df = df.drop_duplicates()
df = df.drop(columns=["ocean_proximity"])

# Define X and y
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Get metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)


# Select a few rows
sample = df.head(5)

# Create figure
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')

# Create table
table = ax.table(
    cellText=sample.values,
    colLabels=sample.columns,
    loc='center'
)

# Style
table.auto_set_font_size(False)
table.set_fontsize(8)
table.auto_set_column_width(col=list(range(len(sample.columns))))

# Save as image
plt.savefig("cleaned_data_table.png", bbox_inches='tight')
plt.show()