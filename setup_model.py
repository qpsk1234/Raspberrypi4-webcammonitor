import urllib.request
import os

def download_model():
    # SSD MobileNet V2 (COCO 90 classes)
    # 人間以外（車、動物、日用品など）も検知可能な標準的なモデルです。
    model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/lite-model_ssd_mobilenet_v2_100_det_retinanet_1.tflite"
    model_filename = "model.tflite"

    print(f"Downloading COCO-compatible object detection model...")
    print(f"Source: {model_url}")
    try:
        req = urllib.request.Request(model_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(model_filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("\nDownload complete!")
        print(f"File saved as: {os.path.abspath(model_filename)}")
        print("This model can detect 90 types of objects including persons, cars, dogs, etc.")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_model()
