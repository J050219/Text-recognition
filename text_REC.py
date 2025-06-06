import os
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
import gc

model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2-4B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    multimodel_max_lenght=32768,
).cuda()

text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 讀取圖片
image_dir = "images"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

all_data = []

# 執行辨識
for image_name in image_files:
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path).convert("RGB").resize((960,960))
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

    # 推論
        # 推論
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
    
    # 清除資源（這行必須在 decode 之後）
    del input_ids, attention_mask, pixel_values, output_ids
    torch.cuda.empty_cache()
    gc.collect()
    

    # 將結果按行分割並記錄
    lines = [line.strip() for line in output_text.split('\n') if line.strip()]
    df = pd.DataFrame(lines, columns=["Text"])
        
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
print("✅ 已將各張圖片辨識結果分別輸出至 output 資料夾中。")
