from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import joblib
import numpy as np
from dotenv import load_dotenv
import os

# ===== Load Environment Variables =====
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# ===== MongoDB Client =====
client = MongoClient(MONGO_URI)  # ✅ Use .env MONGO_URI
db = client['ecopack']
collection = db['form_submissions']

# ===== Flask App Setup =====
app = Flask(__name__)
CORS(app)

# ===== Load ML Model and Encoders =====
model = joblib.load('model.pkl')
le_product = joblib.load('le_product.pkl')
le_durability = joblib.load('le_durability.pkl')
le_shipping = joblib.load('le_shipping.pkl')
le_target = joblib.load('le_target.pkl')

# ===== Prediction API =====
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required_fields = ['productName', 'productType', 'dimensions', 'weight', 'durability', 'shippingMethod']

    if not all(field in data and data[field] for field in required_fields):
        return jsonify({'error': 'Please fill all required fields.'}), 400

    try:
        product_type = le_product.transform([data['productType']])[0]
        length = data['dimensions']['length']
        width = data['dimensions']['width']
        height = data['dimensions']['height']
        weight = data['weight']
        durability = le_durability.transform([data['durability']])[0]
        shipping_method = le_shipping.transform([data['shippingMethod']])[0]

        features = np.array([[product_type, length, width, height, weight, durability, shipping_method]])
        prediction_encoded = model.predict(features)[0]
        prediction = le_target.inverse_transform([prediction_encoded])[0]

        collection.insert_one({
            'formData': data,
            'recommendation': prediction
        })

        return jsonify({'recommendation': prediction})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Error during prediction.'}), 500

# ===== Shipping Cost API =====
@app.route('/get-shipping-cost', methods=['POST'])
def get_shipping_cost():
    data = request.get_json()

    weight = data.get('weight')
    shipping_method = data.get('shippingMethod')

    if weight is None or shipping_method is None:
        return jsonify({"error": "Missing weight or shipping method."}), 400

    try:
        weight = float(weight)
    except:
        return jsonify({"error": "Invalid weight provided."}), 400

    base_cost = 50
    delivery_time = ""

    if shipping_method == 'Air':
        cost = base_cost + (20 * weight)
        delivery_time = "2-4 days"
    elif shipping_method == 'Sea':
        cost = base_cost + (10 * weight)
        delivery_time = "10-15 days"
    elif shipping_method == 'Land':
        cost = base_cost + (5 * weight)
        delivery_time = "5-7 days"
    elif shipping_method == 'Local':
        cost = 30
        delivery_time = "Same day"
    else:
        return jsonify({"error": "Invalid shipping method."}), 400

    return jsonify({
        "estimatedCost": round(cost, 2),
        "deliveryTime": delivery_time
    })

if __name__ == '__main__':
    app.run(port=8000, debug=True)
