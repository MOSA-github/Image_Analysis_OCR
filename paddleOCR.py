import cv2
from paddleocr import PaddleOCR
import os
from utils.save_histogram import save_histogram
from utils.remove_lines import remove_lines
import numpy as np

# Uses PP-OCRv6 models by default
ocr = PaddleOCR(
    use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
    # use_doc_unwarping=False, # Disables text image rectification model via this parameter
    # use_textline_orientation=False, # Disables text line orientation classification model via this parameter
    det_db_thresh=0.1,
    det_db_box_thresh=0.2,
    lang="en",
    # rec_char_type="en" 
)
# ocr = PaddleOCR(lang="en") # Uses English model by specifying language parameter
# ocr = PaddleOCR(ocr_version="PP-OCRv5") # Switches to PP-OCRv5 version via ocr_version parameter
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # Switches to PP-OCRv4 version via ocr_version parameter
# ocr = PaddleOCR(device="gpu") # Enables GPU acceleration for model inference via device parameter
# ocr = PaddleOCR(
#     ocr_version="PP-OCRv3",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
#     text_det_thresh=0.1,
#     text_det_box_thresh=0.3,
# )

# open data
video_path = "sample_data/IMG_5301.mp4"
cap = cv2.VideoCapture(video_path)

output_dir = "results"

if not cap.isOpened():
    raise RuntimeError(f"動画を開けません: {video_path}")

# 動画情報
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"フレーム数: {frame_count}")
print(f"再生時間: {frame_count / fps:.2f} 秒")

# 1fps以下
interval = max(1, int(fps)//4)

frame_index = 0

# 領域
x1, x2 = 180, 230
y1, y2 = 480, 535

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[y1:y2, x1:x2]
    h, w = img.shape[:2]
    img = cv2.resize(img, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)

    # 鮮鋭化フィルタ
    kernel_sharp = np.array([
        [0, -1, 0],
        [-1,  5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # 局所ヒストグラム平たん化
    # img = cv2.resize(img, (w*3, h*3))
    # clahe = cv2.createCLAHE(
    #     clipLimit=3.0,
    #     tileGridSize=(40, 40)
    # )
    # img = clahe.apply(img)

    img = cv2.filter2D(img, -1, kernel_sharp)
    
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1秒ごとにOCR
    if frame_index % interval == 0:
        print(f"\n--- frame {frame_index} ---")
        frame = preprocess(frame)
        save_histogram(frame, f"histogram/frame_{frame_index:04d}.png")
        result = ocr.predict(frame)
        for res in result:
            # JSONとして保存
            json_path = os.path.join(
                output_dir,
                f"frame_{frame_index:04d}.json"
            )
            res.save_to_json(json_path)

            img_path = os.path.join(
                output_dir,
                f"frame_{frame_index:04d}.png"
            )
            res.save_to_img(img_path)
        cv2.imwrite(f"frame/frame_{frame_index:04d}.png", frame)
    frame_index += 1
cap.release()

print("\n終了")