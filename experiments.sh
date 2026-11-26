#!/usr/bin/env bash
set -euo pipefail

config="${1:-config.toml}"

python -m imgcnn train --config "$config" --name resnet
python -m imgcnn evaluate --config "$config" --name resnet
python -m imgcnn export --config "$config" --name resnet
python -m imgcnn score --config "$config" --name resnet
python -m imgcnn scratch --config "$config"

python -m imgcnn train --config "$config" --name small-three-planes --set train.fraction=0.125
python -m imgcnn train --config "$config" --name small-ink-only --set train.fraction=0.125 --set train.channels=1

python -m imgcnn report --config "$config" --name resnet
