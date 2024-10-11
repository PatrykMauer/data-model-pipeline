#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_filename.xlsx> [filter_date]"
    exit 1
fi

# Set variables based on the input filename and optional filter date
INPUT_FILENAME="$1"
BASENAME=$(basename "$INPUT_FILENAME" .xlsx)
INTERIM_FILENAME="data_pipeline/data/interim/${BASENAME}.xlsx"
FILTERED_FILENAME="data_pipeline/data/interim/filtered_${BASENAME}.xlsx"
ENCODED_FILENAME="data_pipeline/data/processed/encoded_${BASENAME}.xlsx"
FILTER_DATE="${2:-}"

# Step 1: Process Data
echo "Processing data..."
python data_pipeline/src/data/process_data.py "data_pipeline/data/raw/${INPUT_FILENAME}" "$INTERIM_FILENAME"

# Step 2: Filter Data
echo "Filtering data..."
python data_pipeline/src/data/filter_data.py "$INTERIM_FILENAME" "$FILTERED_FILENAME"

# Step 2.1: Optionally filter by date
if [ -n "$FILTER_DATE" ]; then
    echo "Filtering data by date: $FILTER_DATE..."
    python data_pipeline/src/data/filter_by_date.py "$FILTERED_FILENAME" "$FILTERED_FILENAME" "$FILTER_DATE"
fi

# Step 3: Encode Data
echo "Encoding data..."
python data_pipeline/src/data/encode_data_const.py "$FILTERED_FILENAME" "$ENCODED_FILENAME"

echo "Data processing completed successfully."

python model_training/train_model.py encoded_results_2024_05_11.xlsx
