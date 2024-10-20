import argparse
import os

import torch
import numpy as np
from gguf import *

from qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)

def get_tensor_name(name: str) -> str:

    return name.replace("visual", "v").replace("encoder.layers", "blk").replace("embeddings.", "").replace("proj.", "").replace("self_attn.", "attn_").replace("layer_norm", "ln").replace("layernorm", "ln").replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("embedding", "embd").replace("final", "post").replace("layrnorm", "ln").replace("lm_head","output")


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-path", default='/mnt/data/hfmc/models/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/8f9fc0b18bf5b2b2326f5866356dca94d59df96e/', help="Path to model directory cloned from HF Hub", required=False)
ap.add_argument("--use-f32", action="store_true", default=False, help="Use f32 instead of f16")
ap.add_argument("-o", "--output-dir", help="Directory to save GGUF files. Default is the original model directory", default=None)

# # with proper
args = ap.parse_args()

model_path = args.model_path
model_name = os.path.basename(model_path).replace(".pth", "")
dir_model = args.model_path

ftype_str = ["f32", "f16"]
ftype = 1
if args.use_f32:
    ftype = 0


model = Qwen2VLForConditionalGeneration.from_pretrained(dir_model)

output_dir = args.output_dir if args.output_dir is not None else dir_model
os.makedirs(output_dir, exist_ok=True)
output_prefix = os.path.basename(output_dir).replace("ggml_", "")

fname_out = os.path.join(output_dir, f"lm_head-{ftype_str[ftype]}.gguf")
fout = GGUFWriter(path=fname_out, arch="clip")

fout.add_file_type(ftype)
fout.add_name(model_name)

fout.add_description("lm_head part of qwen2vl")


state_dict = model.state_dict()
for name, data in state_dict.items():
    if name.find('model') != -1 or name.find('visual') != -1:
        continue

    name = get_tensor_name(name)
    data = data.squeeze().numpy()

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0

    if (data.ndim == 2 or data.ndim == 1) and ftype == 1:
        data = data.astype(np.float16)
        ftype_cur = 1


    print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")
    fout.add_tensor(name, data)


fout.write_header_to_file()
fout.write_kv_data_to_file()
fout.write_tensors_to_file()
fout.close()

print("Done. Output file: " + fname_out)
