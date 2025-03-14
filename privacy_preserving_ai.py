import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from cryptography.fernet import Fernet
import os
import json
import pickle
import logging


# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Generate a key for encryption
def generate_key():
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)
    logging.info("Key generated and saved.")

# Load the previously generated key
def load_key():
    return open("secret.key", "rb").read()

# Encrypt data
def encrypt_data(data, key):
    f = Fernet(key)
    encrypted = f.encrypt(data.encode())
    return encrypted

# Decrypt data
def decrypt_data(encrypted_data, key):
    f = Fernet(key)
    decrypted = f.decrypt(encrypted_data).decode()
    return decrypted

# Load dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    logging.info("Dataset loaded successfully.")
    return data

# Preprocess dataset
def preprocess_data(data):
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    logging.info("Missing values imputed.")
    
    return data_imputed

# Split dataset into training and testing sets
def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

# Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Features scaled.")
    return X_train_scaled, X_test_scaled

# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    logging.info("Model trained successfully.")
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    logging.info("Model evaluation completed.")

# Save model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"Model saved to {filename}.")

# Load model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    logging.info(f"Model loaded from {filename}.")
    return model

# Main function
def main():
    # Generate or load encryption key
    if not os.path.exists("secret.key"):
        generate_key()
    key = load_key()

    # Load dataset
    data_path = 'data.csv'
    data = load_dataset(data_path)

    # Simulate sensitive information encryption (just for demonstration)
    sensitive_data = json.dumps(data.iloc[:, 0].tolist())  # Example encryption for the first column
    encrypted_data = encrypt_data(sensitive_data, key)

    # Data preprocessing
    processed_data = preprocess_data(data)

    # Split dataset
    target_col = 'target'  # replace with your target column
    X_train, X_test, y_train, y_test = split_data(processed_data, target_col)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train model
    model = train_model(X_train_scaled, y_train)

    # Evaluate model
    evaluate_model(model, X_test_scaled, y_test)

    # Save model
    save_model(model, 'privacy_preserving_model.pkl')

if __name__ == "__main__":
    main()