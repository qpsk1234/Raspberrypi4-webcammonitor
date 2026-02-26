import cv2
import os
import time

def test_codec(codec, extension):
    filename = f"test_{codec}{extension}"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    # 640x480, 20fps, 2 seconds
    writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
    
    if not writer.isOpened():
        return False, "Failed to open Writer"

    try:
        # Create a simple animation
        for i in range(40):
            frame = cv2.imread(r"m:\vive-local\records\test_image.jpg") # 既存の画像があれば流用
            if frame is None:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Testing {codec}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            writer.write(frame)
        
        writer.release()
        size = os.path.getsize(filename)
        if size < 1000:
            return False, f"File too small ({size} bytes)"
        return True, f"Success! ({size} bytes)"
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    candidates = [
        ('avc1', '.mp4'),
        ('mp4v', '.mp4'),
        ('H264', '.mp4'),
        ('X264', '.mp4'),
        ('MJPG', '.mp4'),
        ('MJPG', '.avi'),
        ('XVID', '.avi'),
    ]

    print("Checking video codecs compatibility...")
    results = []
    for codec, ext in candidates:
        print(f"Testing {codec} in {ext}...", end=" ", flush=True)
        ok, msg = test_codec(codec, ext)
        if ok:
            print(f"OK: {msg}")
            results.append((codec, ext, "PASS"))
        else:
            print(f"Failed: {msg}")
            results.append((codec, ext, "FAIL"))
    
    print("\n--- Summary ---")
    for c, e, r in results:
        print(f"{c}{e}: {r}")
