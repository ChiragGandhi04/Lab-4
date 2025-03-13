import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("D:/lab-4/Fish.csv")

# Select features and target variable
X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Weight']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train regression model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Evaluate model
predictions = regressor.predict(X_test)
print("Regression MAE:", mean_absolute_error(y_test, predictions))

# Save model and scaler
pickle.dump(regressor, open("regressor.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
