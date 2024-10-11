import json
import pandas as pd
import subprocess
from flask import Flask, request, jsonify
from pathlib import Path

app = Flask(__name__)


def load_previous_mape(filepath='model_training/models/previous_mape.txt'):
    """Load the previous MAPE from a file if it exists."""
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            return float(f.read().strip())
    return None


def save_mape(mape, filepath='model_training/models/previous_mape.txt'):
    """Save the current MAPE to a file."""
    model_output_path = Path(filepath).parent
    # Create the directory if it doesn't exist
    model_output_path.mkdir(parents=True, exist_ok=True)

    # Save the MAPE to the file
    with open(filepath, 'w') as f:
        f.write(str(mape))


app = Flask(__name__)


app = Flask(__name__)


@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        # Retrieve data and filename from the request
        request_data = request.json
        filename = request_data.get('filename', 'results_2024_05_11.xlsx')
        new_data_json = request_data.get('data')

        # Convert the new data from JSON string to a list of dictionaries
        new_data_list = json.loads(new_data_json)
        print(new_data_list)
        # Convert the list of dictionaries to a DataFrame
        new_data_df = pd.DataFrame(new_data_list)

        # Define the file path (assuming the file is stored in 'data_pipeline/data/processed/')
        file_path = Path(f'data_pipeline/data/raw/{filename}')

        # Check if the file exists and append the data
        if file_path.exists():
            # Load existing data
            existing_data_df = pd.read_excel(file_path)

            # Append new data to the existing data
            combined_data_df = pd.concat(
                [existing_data_df, new_data_df],
                ignore_index=True)
        else:
            # If the file doesn't exist, just use the new data
            combined_data_df = new_data_df

        # Save the combined data back to the file
        combined_data_df.to_excel(file_path, index=False)

        # Optionally, run a data processing shell script or other operations
        # For example, running the script to preprocess the data
        command = ['./data_pipeline/data_processing.sh', filename]
        subprocess.run(command, check=True)

        return jsonify({'message': 'Data appended, processed, and saved successfully'}), 200

    except subprocess.CalledProcessError as e:
        return jsonify(
            {'error': f'Error occurred during processing: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    filename = data.get('filename', 'results_2024_05_11.xlsx')

    try:
        # Load the previous MAPE from the correct location
        previous_mape = load_previous_mape(
            filepath='model_training/models/previous_mape.txt')

        # Run the model training script and pass the filename as an argument
        command = ['python3', 'model_training/train_model.py', filename]

        # Call the model training script and wait for it to complete
        result = subprocess.run(command, check=True,
                                capture_output=True, text=True)

        # Log output from the model training script
        output = result.stdout

        # Extract the current MAPE from the output (assuming MAPE is printed in the stdout)
        current_mape = None
        for line in output.splitlines():
            if "MAPE:" in line:
                current_mape = float(line.split("MAPE:")[
                                     1].strip().replace('%', ''))

        # Prepare the response message
        if current_mape is not None:
            # Compare current MAPE with previous MAPE
            if previous_mape is not None:
                mape_diff = current_mape - previous_mape
                message = f"Model training completed successfully. Previous MAPE: {previous_mape}%. Current MAPE: {current_mape}%. Difference: {mape_diff:.2f}%."
            else:
                message = f"Model training completed successfully. Current MAPE: {current_mape}%. No previous MAPE available."

            # Save the current MAPE for future comparison in the correct location
            save_mape(
                current_mape,
                filepath='model_training/models/previous_mape.txt')

        else:
            message = "Model training completed, but no MAPE was calculated."

        return jsonify(
            {'message': message,
             'output': output}), 200

    except subprocess.CalledProcessError as e:
        return jsonify(
            {'error': f'Error occurred during model training: {str(e)}',
             'output': e.output}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
