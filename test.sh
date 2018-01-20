#!/bin/bash

# You should change the output_dir after training.
output_dir="model_output_vangogh2photo"
phase="test"
testing_set="./datasets/test/vangogh2photo"
# which trained model will be used.
# checkpoint="model-15000"

python main.py --output_dir="$output_dir" \
               --phase="$phase" \
               --testing_set="$testing_set" \
               --checkpoint="$checkpoint"

