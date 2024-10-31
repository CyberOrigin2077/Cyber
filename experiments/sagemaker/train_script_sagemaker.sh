#!/bin/bash

accelerate launch \
    --config_file accelerate_config.yaml \
    experiments/models/world/train_dynamic_distributed.py \
    --data_dir data/cyber_pipette/data \
    --config experiments/configs/models/world/genie.yaml \
    --checkpoint_dir "$SM_MODEL_DIR" 