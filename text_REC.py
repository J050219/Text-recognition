import cv2
import numpy as np
import easyocr
import pandas as pd

# 初始化 OCR
reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)

# 載入圖片並預處理
img = cv2.imread("img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = 255 - thresh

# 加強格線特徵
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morphed = cv2.dilate(thresh, kernel, iterations=1)

# 偵測所有格子輪廓
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 過濾並擷取每個格子的座標 (x, y, w, h)
cells = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 50 and h > 20:  # 過濾雜訊
        cells.append((x, y, w, h))

# 將 cell 區塊依照座標排序（上到下，再左到右）
cells = sorted(cells, key=lambda b: (b[1], b[0]))

# OCR 每個 cell 並儲存對應文字與座標
cell_data = []
for (x, y, w, h) in cells:
    roi = img[y:y+h, x:x+w]
    text = reader.readtext(roi, detail=0, paragraph=False)
    clean_text = ' '.join(text).strip()
    cell_data.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': clean_text})

# 利用座標分群 → 成為 row/column 結構
# 先用 Y 軸群組分成多行（允許小誤差）
rows = []
current_row = []
last_y = -1
tolerance = 20

for cell in sorted(cell_data, key=lambda c: c['y']):
    if last_y == -1 or abs(cell['y'] - last_y) < tolerance:
        current_row.append(cell)
    else:
        rows.append(sorted(current_row, key=lambda c: c['x']))
        current_row = [cell]
    last_y = cell['y']

if current_row:
    rows.append(sorted(current_row, key=lambda c: c['x']))

# 轉換成 2D 陣列並輸出為 CSV
table_data = []
for row in rows:
    row_text = [cell['text'] for cell in row]
    table_data.append(row_text)

df = pd.DataFrame(table_data)
df.to_csv("text_REC_1.csv", index=False, encoding='utf-8-sig')
print("✅ 偵測與位置歸位完成，已輸出 CSV：text_REC_1.csv")
