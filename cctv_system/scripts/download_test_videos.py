"""Download public sample traffic videos for testing the violation detection system."""

import urllib.request
from pathlib import Path

def download_videos():
    # Save directory: cctv_system/datasets/test_data
    dest_dir = Path(__file__).resolve().parent.parent / "datasets" / "test_data"
    dest_dir.mkdir(parents=True, exist_ok=True)

    videos = {
        "traffic.mp4": "https://github.com/DeGirum/PySDKExamples/raw/main/images/Traffic.mp4",
        "car_detection.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
    }

    for name, url in videos.items():
        target_path = dest_dir / name
        print(f"Downloading {name} from {url}...")
        try:
            # Simple stream download with basic progress indication
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, int(block_num * block_size * 100 / total_size))
                    print(f"\rProgress: {percent}%", end="", flush=True)

            urllib.request.urlretrieve(url, str(target_path), reporthook=report_progress)
            print(f"\nSaved to {target_path} ({target_path.stat().st_size} bytes)\n")
        except Exception as exc:
            print(f"\nError downloading {name}: {exc}\n")

if __name__ == "__main__":
    download_videos()
