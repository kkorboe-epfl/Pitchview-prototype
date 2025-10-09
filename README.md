# Video Processing Pipeline

This project provides a pipeline to split, crop, and stitch stereo videos from `.m2ts` sources, generating panoramic outputs. It is optimized for desktop and can be tuned for low-power devices like Raspberry Pi.  

The dataset used for testing comes from the [Stereo Egomotion Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/StereoEgomotion/).

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Project Structure](#project-structure)  
3. [Test Script](#test-script) 
4. [Environment Variables](#environment-variables)
5. [Usage](#usage) 

---

## Prerequisites

- Python ≥ 3.10  
- [OpenCV](https://opencv.org/) (`opencv-python`)  
- [NumPy](https://numpy.org/)  
- [python-dotenv](https://pypi.org/project/python-dotenv/)  
- FFmpeg installed and available in PATH  

**Install Python dependencies:**  

```bash
pip install opencv-python numpy python-dotenv
```

## Project Structure
- project-root/
  - data/
    - cars/
      - backup_m2ts/        # *Original .m2ts files moved here after conversion*
      - mp4/                # *Converted .mp4 videos*
    - test_videos              
    - videos_split/
      - cropped/            # *Cropped left/right videos for stitching*
  - output/
    - videos_stitched/      # *Stitched panoramic outputs*
  - scripts/
    - preprocessing/
      - convert_m2ts_to_mp4.py
      - video_splitting.py
      - video_splitting_with_crop.py
    - stitching/
      - video_stitching.py
    - test_stitching_pipeline.py
  - .env                   # *Configuration file*
  - README.md

## Test Script

A test script `test_stitching_pipeline.py` is provided in the scripts folder to quickly verify that the video processing pipeline is working. It uses a small sample video (`car001.mp4`) in `data/test_videos` and runs the full pipeline: splitting, cropping, and stitching.


## Environment Variables
The pipeline uses a `.env` file to define input, output, and stitching parameters. This allows you to change videos, paths, or settings without modifying the Python scripts.

Create a `.env` file in the project root. See example below:

```dotenv
VIDEO_NAME=car002

RAW_DIR=data/cars/mp4
SPLIT_DIR=data/videos_split
CROPPED_DIR=data/videos_split/cropped
STITCHED_DIR=output/videos_stitched

FEATURE_DETECTOR=auto
OVERLAP_PIXELS=50
FPS=20

LIVE_PREVIEW=true
```

#### Explanation of Environment Variables

| Variable           | Description |
|-------------------|-------------|
| `VIDEO_NAME`       | Base name of the video to process (without extension). Example: `car002`. |
| `RAW_DIR`          | Folder containing the original input videos (e.g., `.m2ts` or `.mp4` files). |
| `SPLIT_DIR`        | Folder where the left/right split videos are saved. |
| `CROPPED_DIR`      | Folder for cropped left/right videos before stitching. |
| `STITCHED_DIR`     | Folder where the stitched panoramic video will be saved. |
| `FEATURE_DETECTOR` | Feature detector to use: `auto` (default), `sift`, or `orb`. `auto` selects based on CPU performance. |
| `OVERLAP_PIXELS`   | Horizontal overlap width in pixels for linear blending during stitching. Higher values can smooth transitions but may introduce more distortion. |
| `FPS`              | Optional output frame rate override. If unspecified, the pipeline uses the minimum FPS of the left/right videos. |
| `LIVE_PREVIEW`     | If set to `true`, opens a live preview window showing the stitched video as it is being processed. Press `q` to close the preview early. |

**Notes:**

- `auto` mode chooses **SIFT** on high-performance CPUs for quality, and **ORB** on low-power devices like Raspberry Pi for speed.  
- `OVERLAP_PIXELS` and `FPS` can be adjusted to balance visual quality and performance.

## Usage

This section shows how to process your stereo videos from `.m2ts` to stitched panoramic outputs using the `.env` configuration.

---

#### 1. Convert `.m2ts` videos to `.mp4`
Note: The dataset is not included in this repository. Download the [Stereo Egomotion Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/StereoEgomotion/) and place the .m2ts video files in the data/cars folder before running the script.

Run the conversion script to process all `.m2ts` files in `data/cars`:

```bash
python scripts/preprocessing/convert_m2ts_to_mp4.py
```

#### 2. Split Stereo Videos into Left and Right Channels

Run the splitting script:

```bash
python scripts/preprocessing/video_splitting.py
```

This outputs two separate videos:
  - `left_<VIDEO_NAME>.mp4`  
  - `right_<VIDEO_NAME>.mp4`

**Notes:**
- If you want to crop the videos, run the crop script `video_splitting_with_crop.py`.  

#### 3. Stitch Left and Right Videos into a Panoramic Video

Once you have the left and right (optionally cropped) videos, you can stitch them into a single panoramic video.

Run the stitching script:

```bash
python scripts/stitching/video_stitching.py
```

