import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv("dataset/car_data.csv")

# Remove ID column
df = df.drop("Car ID", axis=1)

# Separate input and output
X = df.drop("Price", axis=1)
y = df["Price"]


# Find categorical columns
categorical_columns = X.select_dtypes(include="object").columns


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)


# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# Train
pipeline.fit(X_train, y_train)


# Test
prediction = pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, prediction))
print("R2 Score:", r2_score(y_test, prediction))


# Save model
joblib.dump(
    pipeline,
    "model/car_price_model.pkl"
)

print("Model saved successfully!")