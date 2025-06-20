# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv('synthetic_packaging_dataset.csv')

# Label Encoding for categorical variables
le_product = LabelEncoder()
le_durability = LabelEncoder()
le_shipping = LabelEncoder()
le_package = LabelEncoder()

data['productType'] = le_product.fit_transform(data['productType'])
data['durability'] = le_durability.fit_transform(data['durability'])
data['shippingMethod'] = le_shipping.fit_transform(data['shippingMethod'])
data['recommendedPackage'] = le_package.fit_transform(data['recommendedPackage'])

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'productType': le_product,
        'durability': le_durability,
        'shippingMethod': le_shipping,
        'recommendedPackage': le_package
    }, f)

# Check class distribution
class_counts = data['recommendedPackage'].value_counts()
print("\nOriginal class distribution:\n", class_counts)

# Find the maximum class count
max_count = class_counts.max()

# Balance the dataset using oversampling
balanced_data = pd.DataFrame()
for label in class_counts.index:
    class_samples = data[data['recommendedPackage'] == label]
    if len(class_samples) < max_count:
        class_upsampled = resample(class_samples,
                                   replace=True,  # Sample with replacement
                                   n_samples=max_count,
                                   random_state=42)
        balanced_data = pd.concat([balanced_data, class_upsampled])
    else:
        balanced_data = pd.concat([balanced_data, class_samples])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced class distribution:\n", balanced_data['recommendedPackage'].value_counts())

# Prepare features and target
X = balanced_data[['productType', 'length_cm', 'width_cm', 'height_cm', 'weight_kg', 'durability', 'shippingMethod']]
y = balanced_data['recommendedPackage']

# Train-test split with stratify (since now all classes are balanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, class_weight='balanced')
model.fit(X_train, y_train)

# Save the trained model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel training completed and saved.")

# Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Get the labels present in y_test
unique_labels = np.unique(y_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=unique_labels, target_names=le_package.inverse_transform(unique_labels)))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_package.inverse_transform(unique_labels))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
