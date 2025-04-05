from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('final_dataset.csv')
categorical_cols = ['region', 'eventType', 'side']

# Initialize LabelEncoders
le_region = LabelEncoder()
le_eventType = LabelEncoder()
le_side = LabelEncoder()

# Fit and transform the encoders
df['region_encoded'] = le_region.fit_transform(df['region'])
df['eventType_encoded'] = le_eventType.fit_transform(df['eventType'])
df['side_encoded'] = le_side.fit_transform(df['side'])

# Load the trained model
with open('models/GradientBoostingRegressor_model.pkl', 'rb') as file:
    model = pickle.load(file)

feature_columns = ['totalGuests', 'region_encoded', 'eventType_encoded', 'side_encoded', 'amount']
target_columns = ['food', 'venue', 'videography', 'decorations', 'entertainment']

df['food_per_guest'] = df['amount'] / df['totalGuests']
X = df[feature_columns + ['food_per_guest']]
y = df[target_columns]

def adjust_predictions_based_on_priority(y_pred, total_amount, total_guests, priorities):
    for i in range(len(y_pred)):
        # Apply user-defined priorities
        priority_weights = np.array(priorities)

        # Set any targets with priority 0 to zero
        y_pred[i] = y_pred[i] * (priority_weights > 0)

        if np.sum(priority_weights) == 0:
            y_pred[i] = np.zeros_like(y_pred[i])
            continue
        
        # Adjust food priority to be higher than venue
        food_priority_index = target_columns.index('food')
        venue_priority_index = target_columns.index('venue')
        
        priority_weights[food_priority_index] += 2  # Boost food priority
        priority_weights[venue_priority_index] -= 1  # Reduce venue priority
        
        # Apply the priority scaling
        scaled_predictions = y_pred[i] * priority_weights
        
        # Reduce food proportionally if totalGuests is low
        if total_guests < 50:  # Example threshold for small guest counts
            scaled_predictions[food_priority_index] = scaled_predictions[food_priority_index] * (total_guests / 100)
        
        # Normalize predictions to match the total amount
        predicted_sum = np.sum(scaled_predictions)
        if predicted_sum != 0:
            scale_factor = total_amount / predicted_sum
            scaled_predictions = scaled_predictions * scale_factor
        
        y_pred[i] = scaled_predictions
    
    return y_pred

def predict_values_with_priorities(model, le_region, le_eventType, le_side, total_guests, region, eventType, side, amount, priorities):
    # Encode categorical features
    region_encoded = le_region.transform([region])[0]
    eventType_encoded = le_eventType.transform([eventType])[0]
    side_encoded = le_side.transform([side])[0]

    # Calculate 'food_per_guest' based on the user input
    food_per_guest = amount / total_guests  # Dynamic calculation

    # Create input data, including food_per_guest
    input_data = np.array([[total_guests, region_encoded, eventType_encoded, side_encoded, amount, food_per_guest]])

    # Predict using the model
    predictions = model.predict(input_data)

    # Adjust predictions based on user priorities
    predictions_adjusted = adjust_predictions_based_on_priority(predictions, amount, total_guests, priorities)

    # Combine results, converting predictions to integers
    results = {key: int(round(value)) for key, value in zip(target_columns, predictions_adjusted[0])}

    return results

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.get_json(force=True)

    # Ensure all required fields are present in the input
    required_fields = ['totalGuests', 'region', 'eventType', 'side', 'amount', 'priorities']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    total_guests = int(data['totalGuests'])
    region = data['region']
    eventType = data['eventType']
    side = data['side']
    amount = float(data['amount'])
    priorities = list(map(int, data['priorities']))

    predictions = predict_values_with_priorities(
        model,
        le_region,
        le_eventType,
        le_side,
        total_guests,
        region,
        eventType,
        side,
        amount,
        priorities
    )

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
