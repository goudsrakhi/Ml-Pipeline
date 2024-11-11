import pickle

# Load the trained model
with open('model/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict function (replace with actual inputs)
def predict(input_features):
    return model.predict([input_features])

# Example prediction
print(predict([0.5, 0.8]))  # Replace with your input values
