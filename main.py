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

# 手動切割
def manual_crop(image_path, image_name):
    img = cv2.imread(image_path)
    boxes = []
    cropping = False
    ix , iy = -1, -1
    current_cursor = (0,0)

    screen_height = 500
    scale = screen_height / img.shape[0]
    display_img = cv2.resize(img, (int(img.shape[1] * scale), screen_height))
    display_clone = display_img.copy()

    def draw_rectangle(event, x, y, flags, param):
        nonlocal cropping, ix, iy, current_cursor, boxes

        if event == cv2.EVENT_LBUTTONDOWN:
            cropping = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            current_cursor = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            cropping = False
            fx , fy = min(x, ix), min(y, iy)
            tx , ty = max(x, ix), max(y, iy)

            real_fx = max(0, min(int(fx / scale), img.shape[1] - 1))
            real_fy = max(0, min(int(fy / scale), img.shape[0] - 1))
            real_tx = max(0, min(int(tx / scale), img.shape[1] - 1))
            real_ty = max(0, min(int(ty / scale), img.shape[0] - 1))

            boxes.append((real_fx, real_fy, real_tx, real_ty))
            #cv2.rectangle(display_clone, (fx, fy), (tx, ty), (0, 255, 0), 2)
            #cv2.imshow("crop", display_clone)

    cv2.namedWindow("crop")
    cv2.setMouseCallback("crop", draw_rectangle)
    print("請在圖片上選擇要切割的區域，按下 Esc 鍵完成。")

    while True:
        display = display_clone.copy()

        for (x1, y1, x2, y2) in boxes:
            fx = int(x1 * scale)
            fy = int(y1 * scale)
            tx = int(x2 * scale)
            ty = int(y2 * scale)
            cv2.rectangle(display, (fx, fy), (tx, ty), (0, 255, 0), 2)

        if cropping:
            x, y = current_cursor
            cv2.rectangle(display, (ix, iy), (x, y), (0, 255, 255), 2)
        
        cv2.imshow("crop", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cv2.destroyAllWindows()

    table_paths = []
    for i, (x1 ,y1 ,x2 ,y2) in enumerate(boxes):
        roi = img[y1:y2, x1:x2]
        save_path = os.path.join(table_dir, f"{os.path.splitext(image_name)[0]}_table_{i}.jpg")
        cv2.imwrite(save_path, roi)
        table_paths.append(save_path)
    return table_paths

def text_position(text_blocks, threshold=10):
    sorted_blocks = sorted(text_blocks, key=lambda b: (round(b['y'] / threshold), b['x']))  # 按照 y 座標排序
    return sorted_blocks

def recognize_text(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
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
    lines = [line.strip() for line in output_text.split('\n') if line.strip()]
    
    text_blocks = []
    for i, line in enumerate(lines):
        if not line:
            continue
        # 假設每行文字的 y 座標是行號 * 30，x 座標是隨機的
        text_blocks.append({
            'text': line,
            'x': 0,  # 假設 x 座標為 0
            'y': i * 30  # 假設 y 座標為行號 * 30
        })

    return text_position(text_blocks)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
# 執行辨識
for image_name in image_files:
    image_path = os.path.join(image_dir, image_name)
    table_path = manual_crop(image_path, image_name)

    all_text_blocks = []
    for idx, path in enumerate(table_path):
        lines = recognize_text(path)
        #text_content = "\n".join(lines)

        #print(f"\n📄 {image_name} - Table{idx+1} 辨識結果：\n{text_content}\n{'='*40}")

        for row_idx, text in enumerate(lines):
            all_text_blocks.append({
                "Text": text
            })

    #max_rows = max(len(col) for col in all_text_blocks)
    #aligned_cols = [col + [''] * (max_rows - len(col)) for col in all_text_blocks]
    df = pd.DataFrame(all_text_blocks, columns=["Text"])

    output_name = os.path.splitext(image_name)[0] + ".csv"
    output_path = os.path.join(output_dir, output_name)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出辨識結果：{output_name}")

print("✅ 已將各張圖片辨識結果分別輸出至 output 資料夾中。")
