# 手動切割表格 + OVIS2文字辨識

這個工具能讓你從圖片中**手動框選表格區塊**，並使用 [OVIS2-4B](https://huggingface.co/AIDC-AI/Ovis2-4B) 進行**文字辨識(OCR)**。
辨識結果會即時顯示在終端機，並輸出CSV的檔案。

---

## 環境安裝

- python 3.8+
- NVIDIA GPU (建議 VRAM >= 8GB)
- CUDA 12.3+

安裝必要的套件:

```bash
pip install transformers==4.46.2 numpy==1.25.0 pillow==10.3.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 
pip install flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp311-cp311-win_amd64.whl
pip install transformers opencv-python pillow pandas
```
---

## 專案目錄架構
```
Text-recognition/
<tr> images/     #放需要辨識的圖片
<tr> tables/     #存放框選後的子圖片
<tr> output/     #匯出辨識結果 CSV
<tr> main.py     #主程式
<tr> README.md   #使用說明
```

## **使用方法**
1. 將要辨識的圖片放入images/資料夾中
2. 執行 python main.py
3. 框選表格區域(按ESC結束選取)
4. 將所有選取區塊進行OVIS辨識
5. 辨識結果即時顯示於終端機
6. 生成CSV檔存於output/資料夾中

## **模型資訊-OVIS2**
OVIS2-4B擅長:

    圖文理解

    表格辨識

    支援繁體中文+手寫字