#!/bin/bash
# Download sample videos for PitchView prototype
# Usage: ./download_videos.sh

set -e

echo "Downloading sample videos..."

# Create directories
mkdir -p raw

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown for Google Drive downloads..."
    pip install gdown
fi

# Google Drive file IDs and filenames
VIDEOS=(
    "20251116_103024_left.mp4|14h6jFOvYTAN32xRkn_2UWLdyQQQj_yxo"
    "20251116_103024_right.mp4|1jbyafqqyV3XF4EjrSaTZ7l2-sJyRo__m"
)

for entry in "${VIDEOS[@]}"; do
    IFS='|' read -r filename file_id <<< "$entry"
    
    if [ ! -f "raw/$filename" ]; then
        echo "Downloading $filename from Google Drive..."
        gdown "https://drive.google.com/uc?id=$file_id" -O "raw/$filename"
    else
        echo "$filename already exists, skipping..."
    fi
done

echo "Done! Videos are ready in raw/"
