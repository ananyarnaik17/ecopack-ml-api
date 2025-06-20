from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)

# Load cleaned dataset
df = pd.read_csv('cleaned_packaging_data.csv')

# Prepare features for KNN
X = df[['height cm', 'width cm', 'length cm']].values

# Initialize and train KNN model
knn = NearestNeighbors(n_neighbors=3, algorithm='auto')
knn.fit(X)

@app.route('/recommend', methods=['POST'])
def recommend_package():
    data = request.json
    try:
        user_height = float(data.get('height'))
        user_width = float(data.get('width'))
        user_length = float(data.get('length'))

        user_input = np.array([[user_height, user_width, user_length]])

        # Find nearest neighbors
        distances, indices = knn.kneighbors(user_input)

        # Get recommended packages
        recommendations = []
        for idx in indices[0]:
            product = df.iloc[idx]
            recommendations.append({
                'ASIN': product['ASIN'],
                'Product_Type': product['PT'],
                'Height_cm': product['height cm'],
                'Width_cm': product['width cm'],
                'Length_cm': product['length cm']
            })

        return jsonify({
            'recommendations': recommendations,
            'message': 'Packaging recommendation fetched successfully!'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
