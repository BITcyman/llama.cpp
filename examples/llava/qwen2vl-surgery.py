import argparse
import os 
import torch
from transformers import AutoModel, AutoTokenizer
from qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to Qwen2-VL model", default='/mnt/data/hfmc/models/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/8f9fc0b18bf5b2b2326f5866356dca94d59df96e/')
args = ap.parse_args()

# find the model part that includes the the multimodal projector weights
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model,
    trust_remote_code=True,
    local_files_only=True
)

checkpoint = model.state_dict()

# get a list of mm tensor names
mm_tensors = [k for k, v in checkpoint.items() if k.startswith("lm_head")]

# store these tensors in a new dictionary and torch.save them
lm_head = {name: checkpoint[name].float() for name in mm_tensors}
torch.save(lm_head, f"{args.model}/qwen2_vl.lm_head")


clip_tensors = [k for k, v in checkpoint.items() if k.startswith("visual")]
if len(clip_tensors) > 0:
    clip = {name.replace("visual.", ""): checkpoint[name].float() for name in clip_tensors}
    torch.save(clip, f"{args.model}/qwen2_vl.clip")

    # added tokens should be removed to be able to convert Mistral models
    if os.path.exists(f"{args.model}/added_tokens.json"):
        with open(f"{args.model}/added_tokens.json", "w") as f:
            f.write("{}\n")
    
config = model.model.config  # for qwen2vl, the llm is called 'model'
config.auto_map = {
    "AutoConfig": "configuration_qwen2_vl.Qwen2VLConfig",
    "AutoModel": "modeling_qwen2_vl.Qwen2VLModel",
    "AutoModelForCausalLM": "modeling_qwen2_vl.Qwen2VLForConditionalGeneration",
    "AutoModelForSeq2SeqLM": "modeling_qwen2_vl.Qwen2VLForConditionalGeneration"
}

model.model.save_pretrained(f"{args.model}/model")
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
tok.save_pretrained(f"{args.model}/model")

print("Done!")
print(f"Now you can convert {args.model} to a regular LLaMA GGUF file.")
print(f"Also, use {args.model}/minicpmv.projector to prepare a minicpmv-encoder.gguf file.")
