import subprocess
from pathlib import Path
import shutil

# Directories
input_dir = Path("data/cars")
output_dir = input_dir / "mp4"
backup_dir = input_dir / "backup_m2ts"

# Create output and backup directories if they don't exist
output_dir.mkdir(parents=True, exist_ok=True)
backup_dir.mkdir(parents=True, exist_ok=True)

for m2ts_file in input_dir.glob("*.m2ts"):
    output_file = output_dir / f"{m2ts_file.stem}.mp4"
    print(f"Converting {m2ts_file.name} -> {output_file.name}...")

    # ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(m2ts_file),        # input file
        "-c:v", "libx264",           # video codec
        "-crf", "18",                # quality (lower = better)
        "-preset", "fast",           # encoding speed/efficiency
        "-c:a", "aac",               # audio codec
        "-b:a", "192k",              # audio bitrate
        str(output_file)             # output file
    ]

    # Run ffmpeg
    subprocess.run(cmd, check=True)

    # Move original file to backup folder
    shutil.move(str(m2ts_file), backup_dir / m2ts_file.name)
    print(f"Moved original {m2ts_file.name} -> {backup_dir.name}/")

print("✅ All .m2ts files converted and originals backed up.")
