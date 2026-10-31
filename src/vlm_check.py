"""経由地での写真判定 (VLM = Qwen3-VL-8B-Instruct をローカルで)。
使い方: from vlm_check import VLM; v = VLM(); v.ask('shot.png', '質問')"""
import time, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = 'Qwen/Qwen3-VL-8B-Instruct'


class VLM:
    def __init__(self, device='cuda'):
        t0 = time.time()
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device)
        self.load_s = time.time() - t0

    def ask(self, image_path, question, max_new_tokens=60):
        img = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': question}]}]
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        return text, time.time() - t0


if __name__ == '__main__':
    import sys
    v = VLM(); print(f'loaded in {v.load_s:.0f}s')
    for p in sys.argv[1:]:
        print(p, '->', v.ask(p, 'この画像に倒れている人はいますか? 「はい」か「いいえ」で答え、そのあと一言で状況を説明してください。'))
