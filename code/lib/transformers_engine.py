import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import json
import re
from pathlib import Path

# Global variable to cache the model and processor to avoid re-loading for every image
_TRANSFORMERS_CACHE = {
    "model": None,
    "processor": None,
    "device": None
}

def _get_transformers_engine(model_id: str):
    """Loads and returns the model and processor, using a cache to prevent redundant loads."""
    if _TRANSFORMERS_CACHE["model"] is not None:
        return _TRANSFORMERS_CACHE["model"], _TRANSFORMERS_CACHE["processor"], _TRANSFORMERS_CACHE["device"]

    print(f"🚀 Loading Transformers model: {model_id} (this may take a while)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use device_map="auto" to handle multi-GPU setups automatically
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2_seq.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    _TRANSFORMERS_CACHE["model"] = model
    _TRANSFORMERS_CACHE["processor"] = processor
    _TRANSFORMERS_CACHE["device"] = device
    
    print(f"✅ Model loaded on {device}")
    return model, processor, device

def predict_with_transformers(image_path: Path, model_id: str, conf_threshold: float, no_bird_conf: float) -> tuple:
    """
    Performs multimodal inference using the local Transformers library.
    
    Returns: (category, label, label_cn, confidence, raw_json)
    """
    category = "scenery"
    label = "unknown"
    label_cn = ""
    confidence = 0.0
    raw_json = "{}"

    try:
        model, processor, device = _get_transformers_engine(model_id)
        
        # 1. Prepare the input
        image = Image.open(image_path).convert("RGB")
        prompt = (
            "You are an expert bird and wild animal identification system. "
            "For the given image, output a JSON object with the following fields: "
            "`category` – a string that must be one of: 'bird', 'animal', 'people', or 'scenery'. "
            "`label` – the English name of the bird/animal, or a brief English description if the category is 'people' or 'scenery'."
            "`label_cn` – the Chinese name corresponding to `label`."
            "`confidence` – a float between 0.0 and 1.0 indicating the model's confidence. "
            "If the image contains no recognizable animal, set `category` to 'people' or 'scenery' as appropriate and provide an appropriate English description, leaving `label_cn` blank."
        )
        
        # Note: The prompt format depends on the specific model architecture (e.s. Qwen-VL vs Llava).
        # This implementation assumes a standard Chat template approach.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # 2. Process and Generate
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        
        # For multimodal models, we often need to pass the images separately to the processor
        # This part is highly model-dependent, but this is the common pattern for vision-language models
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        
        # We combine the text inputs and the pixel values
        # (Note: implementation details vary by model; this is a generalized pattern)
        output_ids = model.generate(**inputs, pixel_values=pixel_values, max_new_tokens=200)
        
        # 3. Decode the response
        response_text = processor.decode(output_ids[0], skip_special_tokens=True)
        
        # 4. Parse the JSON from the text response
        # Extract JSON block if the model wraps it in markdown
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            raw_json = json.dumps(data, ensure_ascii=False)
            label = data.get('label', 'unknown').lower()
            label_cn = data.get('label_cn', '未知')
            confidence = float(data.get('confidence', 0.0))
            category = data.get('category', 'scenery')
        else:
            # Fallback if no JSON found
            response_text = response_text.strip()
            label = response_text
            raw_json = json.dumps({"raw_text": response_text}, ensure_ascii=False)

    except Exception as e:
        print(f"⚠️  Transformers request failed for {image_path.name}: {e}")

    return category, label, label_cn, confidence, raw_json