import requests
from PIL import Image
import io
import random
import torch  # <-- Import torch here
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

# Initialize the BlipProcessor and BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize the StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cpu")

def get_image_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def generate_fashion_image(shirt_url, pant_url, model_type):
    shirt_img = download_image(shirt_url)
    pant_img = download_image(pant_url)
    
    # Combine the images (shirt and pant)
    combined_image = Image.new('RGB', (shirt_img.width, shirt_img.height + pant_img.height))
    combined_image.paste(shirt_img, (0, 0))
    combined_image.paste(pant_img, (0, shirt_img.height))
    
    # Save the combined image to the generated folder
    output_path = f"static/generated/generated_output_{random.randint(1000, 9999)}.jpg"
    combined_image.save(output_path)
    return output_path

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise ValueError("Image download failed.")
