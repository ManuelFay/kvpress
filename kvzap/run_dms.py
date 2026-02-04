import os
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TextStreamer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)

# Load config and set pad_token_id (missing from nvidia's config.json)
config = AutoConfig.from_pretrained("nvidia/Qwen3-8B-DMS-8x", trust_remote_code=True, local_files_only=True)
config.pad_token_id = tokenizer.pad_token_id

model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Qwen3-8B-DMS-8x",
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # The `trust_remote_code` part is important
    # as otherwise Qwen3 code (without DMS) will be loaded
    trust_remote_code=True,
    local_files_only=True
)


conversation = [
    {"role": "user", "content": "Solve: x^2 -2x + 1 = 0"}
]
prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

streamer = TextStreamer(tokenizer, skip_prompt=False)
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    streamer=streamer,
    max_new_tokens=128
)
