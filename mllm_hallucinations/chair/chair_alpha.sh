#!/bin/bash

# Argument for the Python script (only input file) 
#NPUT_FILE="/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/our_qwen2vl_alpha_0.4_tokens_64.json"
#OUTPUT_DIR="/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/json_results"

#INPUT_FILE="/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/our_qwen2vl_alpha_0.4_tokens_64.json"
OUTPUT_DIR="/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/json_results"
INPUT_FILE="/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/vcd_results_alpha_0.4_tokens_128.json"

# Run the conversion script with the input argument
echo "Running the conversion script..."
python /scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/convert_json.py --input_file "$INPUT_FILE"

# Check if the conversion was successful
if [ $? -eq 0 ]; then
    echo "Conversion successful. Proceeding with the next step..."
    
    # Extract alpha from the input file for naming consistency
    alpha_str=$(echo "$INPUT_FILE" | sed -E 's/.*alpha_([0-9.]+).*/\1/')
    
    # Construct the paths to the generated greedy and sampling files
    GREEDY_FILE="${OUTPUT_DIR}/our_128_${alpha_str}_greedy.json"
    SAMPLING_FILE="${OUTPUT_DIR}/our_128_${alpha_str}_sampling.json"
    
    # Run the CHAIR-metric-standalone script for greedy
    echo "Running chair.py with file: $GREEDY_FILE"
    python /scratch/bcyh/samyakr99/Chair/CHAIR-metric-standalone/chair.py \
        --cap_file "$GREEDY_FILE" \
        --image_id_key image_id \
        --caption_key caption \
        --cache chair.pkl \
        --save_path "outputs_128_${alpha_str}_greedy.json"
    
    # Check if chair.py ran successfully for greedy
    if [ $? -eq 0 ]; then
        echo "CHAIR-metric-standalone script completed successfully for greedy."
    else
        echo "CHAIR-metric-standalone failed for greedy. Exiting."
        exit 1
    fi
    
    # Run the CHAIR-metric-standalone script for sampling
    echo "Running chair.py with file: $SAMPLING_FILE"
    python /scratch/bcyh/samyakr99/Chair/CHAIR-metric-standalone/chair.py \
        --cap_file "$SAMPLING_FILE" \
        --image_id_key image_id \
        --caption_key caption \
        --cache chair.pkl \
        --save_path "outputs_128_${alpha_str}_sampling.json"
    
    # Check if chair.py ran successfully for sampling
    if [ $? -eq 0 ]; then
        echo "CHAIR-metric-standalone script completed successfully for sampling."
    else
        echo "CHAIR-metric-standalone failed for sampling. Exiting."
        exit 1
    fi

else
    echo "Conversion failed. Exiting."
    exit 1
fi
