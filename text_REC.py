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
    _, binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 偵測輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 30:  # 過濾小塊雜訊
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # 按 Y（top→bottom）排序
    table_paths = []

    for i, (x, y, w, h) in enumerate(boxes):
        roi = img[y:y+h, x:x+w]
        save_path = os.path.join(table_dir, f"{os.path.splitext(image_name)[0]}_table{i+1}.jpg")
        cv2.imwrite(save_path, roi)
        table_paths.append((save_path, y))  # 回傳圖片路徑與 Y 座標
    return table_paths

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
# 執行辨識
for image_name in image_files:
    image_path = os.path.join(image_dir, image_name)
    table_images = split_tables(image_path, image_name)

    all_text_blocks = []
    for table_path, y in table_images:
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
        all_text_blocks.append((y, output_text))
        print(f"\n📄 {image_name} 辨識結果：\n{output_text}\n")

        del input_ids, attention_mask, pixel_values, output_ids
        torch.cuda.empty_cache()
        gc.collect()

    all_text_blocks.sort(key=lambda x: x[0])  # 按 Y 座標排序
    final_lines = []
    for _, text in all_text_blocks:
        # 將結果按行分割並記錄
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        final_lines.extend(lines)

    df = pd.DataFrame(final_lines, columns=["Text"])
    df.to_csv(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.csv"), index=False, encoding="utf-8-sig")

print("✅ 已將各張圖片辨識結果分別輸出至 output 資料夾中。")
