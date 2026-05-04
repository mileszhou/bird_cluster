from vllm import LLM
from PIL import Image

# 1. Initialize the engine (it will pre-allocate your 4090/5090 VRAM)
# Try "Qwen/Qwen2-VL-7B-Instruct" or "mistralai/Pixtral-12B-2409"
llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct", limit_mm_per_prompt={"image": 1})

# Retrieve the internal tokenizer
tokenizer = llm.get_tokenizer()

# 2. Prepare your batch (can be hundreds of images)
image = Image.open("./data/jpg/_D5D8111.jpg")
prompt = "USER: <image>\nDescribe the primary objects in this photo.\nASSISTANT:"

# Example for a Vision-Language model
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image"}, # The template will replace this with the correct placeholder
        ],
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 3. Generate (Continuous Batching happens here automatically)
# outputs = llm.generate({
#     "prompt": prompt,
#     "multi_modal_data": {"image": image},
# })

# Then pass this prompt to llm.generate
outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}})

print(outputs[0].outputs[0].text)