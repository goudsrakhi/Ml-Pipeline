import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error

# Load the test data
data = pd.read_csv('data/dataset.csv')
X_test = data[['feature1', 'feature2']]  # Replace with your features
y_test = data['target']                 # Replace with your target variable

# Load the trained model
with open('model/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
