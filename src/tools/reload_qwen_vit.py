import torch
from transformers import AutoModelForCausalLM
import os 

torch.manual_seed(1234)

qwen_model_path = 'pretrained/Qwen-VL-Chat'
save_folder = 'pretrained/visual_tokenizer/'
os.makedirs(save_folder, exist_ok = True)
save_path = os.path.join(save_folder,'qwen_vit_G.pt')

model = AutoModelForCausalLM.from_pretrained(qwen_model_path, device_map="cpu", trust_remote_code=True).eval()

visual_encoder = model.transformer.visual
print(visual_encoder)

torch.save(visual_encoder.state_dict(), save_path)
