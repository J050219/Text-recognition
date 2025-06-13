import os
import cv2
import pandas as pd
import torch
import gc
from PIL import Image
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2-4B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    multimodel_max_lenght=32768,
).cuda()

text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

output_dir = "output"
table_dir = "tables"
image_dir = "images"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

def split_tables(image_path, image_name):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 偵測輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_paths = []

    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 50:  # 排除太小的雜訊區塊
            roi = img[y:y+h, x:x+w]
            save_path = os.path.join(table_dir, f"{os.path.splitext(image_name)[0]}_table{idx+1}.jpg")
            cv2.imwrite(save_path, roi)
            table_paths.append(save_path)
    return table_paths

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
# 執行辨識
for image_name in image_files:
    image_path = os.path.join(image_dir, image_name)
    table_images = split_tables(image_path, image_name)

    for table_path in table_images:
        image = Image.open(table_path).convert("RGB")
        images = [image]

        # 設定提示詞與圖片數
        max_partition = 3
        text_prompt = "請辨識圖片中的所有文字，列出完整清單。"
        query = "<image>\n" + text_prompt

        # 前處理
        prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(model.device)
        attention_mask = attention_mask.unsqueeze(0).to(model.device)
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
        pixel_values = [pixel_values]

        # 生成 output_ids
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                do_sample=False,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
            )[0]

        # 解析 output_ids 成 output_text
        output_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"\n📄 {image_name} 辨識結果：\n{output_text}\n")

        # 將結果按行分割並記錄
        lines = [line.strip() for line in output_text.split('\n') if line.strip()]
        df = pd.DataFrame(lines, columns=["Text"])

        output_name = os.path.splitext(os.path.basename(table_path))[0] + ".csv"
        output_path = os.path.join(output_dir, output_name)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # 清除資源（這行必須在 decode 之後）
        del input_ids, attention_mask, pixel_values, output_ids
        torch.cuda.empty_cache()
        gc.collect()
        
print("✅ 已將各張圖片辨識結果分別輸出至 output 資料夾中。")
