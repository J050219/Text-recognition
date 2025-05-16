import cv2
import numpy as np
import easyocr
import pandas as pd

# 初始化 OCR
reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)

# 載入圖片
img = cv2.imread("img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 圖像處理：模糊、二值化、線條加強
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = 255 - thresh
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morphed = cv2.dilate(thresh, kernel, iterations=1)

# 偵測格子
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 50 and h > 20:  # 過濾雜訊區塊
        boxes.append((x, y, w, h))

# 根據 y 座標排序，再根據 x 排序（行內排序）
boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

# 分組為行
rows = []
current_row = []
last_y = -1
threshold = 20  # 行距容忍值
for box in boxes:
    x, y, w, h = box
    if last_y == -1 or abs(y - last_y) < threshold:
        current_row.append(box)
        last_y = y
    else:
        rows.append(sorted(current_row, key=lambda b: b[0]))
        current_row = [box]
        last_y = y
if current_row:
    rows.append(sorted(current_row, key=lambda b: b[0]))

# OCR 並建構表格陣列
table_data = []
for row in rows:
    row_data = []
    for (x, y, w, h) in row:
        roi = img[y:y+h, x:x+w]
        text = reader.readtext(roi, detail=0, paragraph=False)
        clean_text = ' '.join(text).strip()
        row_data.append(clean_text)
    table_data.append(row_data)

# 輸出為 CSV 檔
df = pd.DataFrame(table_data)
df.to_csv("detect.csv", index=False, encoding='utf-8-sig')
print("✅ CSV 輸出完成：detect.csv")
