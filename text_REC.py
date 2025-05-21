import cv2
import pandas as pd
import easyocr
import numpy as np
from collections import defaultdict
# 讀取圖片
image = cv2.imread("img.jpg")
# 初始化 OCR（中英文）
reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)
# 執行辨識
results = reader.readtext(image)
# 擷取文字與其中心座標
text_items = []
for (bbox, text, conf) in results:
    if conf > 0.3 and text.strip():
        (tl, tr, br, bl) = bbox
        cx = int((tl[0] + tr[0] + br[0] + bl[0]) / 4)
        cy = int((tl[1] + tr[1] + br[1] + bl[1]) / 4)
        text_items.append({'text': text.strip(), 'x': cx, 'y': cy})
# === 將文字依 Y 分群（形成表格列）===
row_threshold = 25
rows_dict = defaultdict(list)
row_keys = []
for item in sorted(text_items, key=lambda i: i['y']):
    matched = False
    for key in row_keys:
        if abs(item['y'] - key) <= row_threshold:
            rows_dict[key].append(item)
            matched = True
            break
    if not matched:
        rows_dict[item['y']].append(item)
        row_keys.append(item['y'])
# 對每行文字依 X 軸排序
sorted_rows = []
for key in sorted(row_keys):
    row = sorted(rows_dict[key], key=lambda i: i['x'])
    sorted_rows.append([cell['text'] for cell in row])
# 對齊欄位
max_cols = max(len(r) for r in sorted_rows)
for r in sorted_rows:
    while len(r) < max_cols:
        r.append("")
# 輸出 CSV
df = pd.DataFrame(sorted_rows)
df.to_csv("text_REC_2.csv", index=False, encoding="utf-8-sig")
print("✅ 已輸出為：text_REC_2.csv")
