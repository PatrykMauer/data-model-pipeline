import os
import smtplib
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import argparse


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def load_dataset(file_path):
    df = pd.read_excel(file_path)
    # Drop unnecessary columns
    df = df.drop(columns=['AUCTION DATE', 'URL', 'ImageName'])
    # Separate features (X) and target (y)
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    return X, y


def notify_performance_drop(mape, baseline_mape):
    message = f"Warning: Model MAPE has exceeded baseline by 10%.\n" \
              f"Model MAPE: {mape}%\nBaseline MAPE: {baseline_mape}%"

    # Get email details from environment variables (safer than hardcoding)
    sender = os.getenv('SMTP_SENDER', 'default_sender@example.com')
    receiver = os.getenv('SMTP_RECEIVER', 'default_receiver@example.com')
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.example.com')
    smtp_port = os.getenv('SMTP_PORT', 587)
    smtp_user = os.getenv('SMTP_USER', 'default_user')
    smtp_password = os.getenv('SMTP_PASSWORD', 'default_password')

    msg = MIMEText(message)
    msg['Subject'] = 'Model Performance Alert'
    msg['From'] = sender
    msg['To'] = receiver

    # Connect to SMTP server and send email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender, receiver, msg.as_string())
        print('Performance alert email sent.')
    except Exception as e:
        print(f'Failed to send email: {e}')


def calculate_baseline(df):
    # Group by artist and calculate the mean price for each artist
    baseline = df.groupby('ARTIST')['PRICE'].mean().reset_index()
    # Merge the baseline back to the dataset to have the baseline prediction
    df = pd.merge(df, baseline, on='ARTIST', how='left',
                  suffixes=('', '_baseline'))
    return df['PRICE_baseline']


def save_mape(mape, filepath='model_training/models/previous_mape.txt'):
    """Save the current MAPE to a file."""
    with open(filepath, 'w') as f:
        f.write(str(mape))


def load_previous_mape(filepath='model_training/models/previous_mape.txt'):
    """Load the previous MAPE from a file if it exists."""
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            return float(f.read().strip())
    return None


def main():
    # Use argparse to accept the file name as an argument
    parser = argparse.ArgumentParser(
        description='Train XGBoost model on auction data.')
    parser.add_argument(
        'input_file', type=str,
        help='Name of the input Excel file (without full path)')
    args = parser.parse_args()

    # Define the base path where the processed files are located
    base_path = Path('data_pipeline/data/processed')

    # Combine the base path with the input file name to get the full path
    dataset_file = base_path / args.input_file

    # Ensure the file exists before proceeding
    if not dataset_file.exists():
        print(f"Error: File {dataset_file} does not exist.")
        return

    # Load the dataset
    X, y = load_dataset(dataset_file)

    # Calculate baseline predictions
    df = pd.read_excel(dataset_file)
    baseline_y_pred = calculate_baseline(df)

    # Evaluate the baseline model performance
    baseline_mape = mean_absolute_percentage_error(y, baseline_y_pred)
    print(f'Baseline MAPE: {baseline_mape}%')

    # Load the previous MAPE from the file (if exists)
    previous_mape = load_previous_mape()

    # Split the data into train and test sets
    train_size = int(0.8 * len(X))  # 80% for training
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize and train the XGBoost model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # Create a directory to save the model if it doesn't exist
    model_output_path = Path('model_training/models')
    model_output_path.mkdir(parents=True, exist_ok=True)

    # Save the trained model
    model_file = model_output_path / 'xgb_model_default_params.joblib'
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Check performance drop and notify if necessary
    if previous_mape is not None and mape > previous_mape * 1.10:
        notify_performance_drop(mape, previous_mape)

    # Save the current MAPE for future comparison
    save_mape(mape)

    # Print evaluation metrics
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'MAPE: {mape}%')
    print(f'R2 Score: {r2}')


if __name__ == '__main__':
    main()
