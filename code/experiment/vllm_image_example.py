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

system_prompt = (
    "You are an expert bird and wild animal identification system. "
    "For the given image, output a JSON object with the following fields: "
    "`category` – a string that must be one of: 'bird', 'animal', 'people' (including a single person), or 'scenery'. "
    "`label` – the English name of the bird/animal, or a brief English description if the category is 'people' or 'scenery'."
    "`label_cn` – the Chinese name corresponding to `label`."
    "`confidence` – a float between 0.0 and 1.0 indicating the model's confidence. "
    "If the image contains no recognizable animal, set `category` to 'people' or 'scenery' as appropriate and provide an appropriate English description, leaving `label_cn` blank."
)

# Example for a Vision-Language model
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Identify this image."},
        ],
    },
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