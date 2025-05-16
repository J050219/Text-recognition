import cv2
import numpy as np
import pandas as pd
import easyocr
from collections import defaultdict

# 初始化 EasyOCR（支援繁中＋英文）
reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)

# 讀取圖像
img = cv2.imread("img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = 255 - thresh

# 強化格線（用膨脹來連接邊緣）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morphed = cv2.dilate(thresh, kernel, iterations=1)

# 偵測輪廓
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 過濾出儲存格區塊
cells = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 50 < w < img.shape[1] and 20 < h < img.shape[0] // 2:
        cells.append((x, y, w, h))

# 按照 y (行) + x (欄) 排序
cells = sorted(cells, key=lambda b: (b[1], b[0]))

# OCR 每個 cell 並記錄結果
cell_data = []
for (x, y, w, h) in cells:
    roi = img[y:y+h, x:x+w]
    text = reader.readtext(roi, detail=0, paragraph=False)
    clean_text = ' '.join(text).strip()
    cell_data.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': clean_text})

# ===== 分群：根據 y 座標吸附進 row 結構 =====
row_threshold = 15
rows_grouped = defaultdict(list)

for cell in cell_data:
    assigned = False
    for ry in rows_grouped:
        if abs(cell['y'] - ry) <= row_threshold:
            rows_grouped[ry].append(cell)
            assigned = True
            break
    if not assigned:
        rows_grouped[cell['y']].append(cell)

# 每行依照 x 排序後組成完整表格陣列
sorted_rows = []
for ry in sorted(rows_grouped.keys()):
    row = sorted(rows_grouped[ry], key=lambda c: c['x'])
    sorted_rows.append([c['text'] for c in row])

# 補齊每行缺少的欄位（不規則表格修正）
max_cols = max(len(r) for r in sorted_rows)
for row in sorted_rows:
    while len(row) < max_cols:
        row.append("")

# 匯出 CSV
df = pd.DataFrame(sorted_rows)
df.to_csv("text_REC.csv", index=False, encoding='utf-8-sig')

print("✅ 表格偵測與辨識完成，已輸出：text_REC.csv")
