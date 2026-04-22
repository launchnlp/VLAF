#!/bin/bash

for input_file in results/sad_stages_re_best_layer/*/*.json; do
    echo "Evaluating: ${input_file}"
    python -m evaluation.sad_stages_evaluation \
        --input_file "${input_file}"
done
