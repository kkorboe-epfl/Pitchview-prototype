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
    "left.mp4|1kx-GF8W27AX_OEODd-eetmHcPDzkaqDe"
    "right.mp4|1tKbCwkAx3arpgFpYrzFOMXKsGVeQVLi3"
)

# Download videos in parallel
pids=()
for entry in "${VIDEOS[@]}"; do
    IFS='|' read -r filename file_id <<< "$entry"
    
    if [ ! -f "raw/$filename" ]; then
        echo "Downloading $filename from Google Drive..."
        gdown "https://drive.google.com/uc?id=$file_id" -O "raw/$filename" &
        pids+=($!)
    else
        echo "$filename already exists, skipping..."
    fi
done

# Wait for all downloads to complete
for pid in "${pids[@]}"; do
    wait $pid
done

echo "Done! Videos are ready in raw/"

# Flip videos in parallel
echo "Flipping videos..."
ffmpeg -i raw/left.mp4 -vf "hflip,vflip" -c:a copy -y raw/leftflip.mp4 &
ffmpeg -i raw/right.mp4 -vf "hflip,vflip" -c:a copy -y raw/rightflip.mp4 &

# Wait for both ffmpeg processes to complete
wait

echo "Videos flipped: raw/leftflip.mp4 and raw/rightflip.mp4"