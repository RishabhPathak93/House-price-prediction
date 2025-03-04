# train_model.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from tqdm import tqdm

# Load dataset
print("Loading dataset...")
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target

# Add more features (for demonstration, we'll add some polynomial features)
print("Adding polynomial features...")
df['AveRoomsSq'] = df['AveRooms'] ** 2
df['AveBedrmsSq'] = df['AveBedrms'] ** 2
df['PopulationSq'] = df['Population'] ** 2
df['HouseAgeSq'] = df['HouseAge'] ** 2
df['AveOccupSq'] = df['AveOccup'] ** 2

# Split data
print("Splitting data into training and testing sets...")
X = df.drop('PRICE', axis=1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training the model...")
model = LinearRegression()

# Use tqdm to show progress during training
for _ in tqdm(range(100), desc="Training Progress"):
    model.fit(X_train, y_train)

# Save the model
print("Saving the model...")
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")