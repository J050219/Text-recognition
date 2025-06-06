import os
import pandas as pd
import torch
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

# 讀取圖片
image_dir = "images"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

all_data = []

# 執行辨識
for image_name in image_files:
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path).convert("RGB")
    images = [image]

    # 設定提示詞與圖片數
    max_partition = 9
    text_prompt = "請辨識圖片中的所有文字，列出完整清單。"
    query = "<image>\n" + text_prompt

    # 前處理
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = attention_mask.unsqueeze(0).to(model.device)
    pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]

    # 推論
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

        output_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"\n📄 {image_name} 辨識結果：\n{output_text}\n")

        # 將結果按行分割並記錄
        lines = [line.strip() for line in output_text.split('\n') if line.strip()]
        for line in lines:
            all_data.append([image_name, line])

# 輸出 CSV
if all_data:
    df = pd.DataFrame(all_data, columns=["Image", "Text"])
    df.to_csv("text.csv", index=False, encoding="utf-8-sig")
print("✅ 已輸出為：text.csv")
