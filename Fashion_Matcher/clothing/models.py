import requests
from PIL import Image
import io
import random
from diffusers import StableDiffusionPipeline
import torch

 # Pre-download BLIP Model
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Downloading BLIP Model...")
BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=False)
BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=False)

# Pre-download Stable Diffusion Model
from diffusers import StableDiffusionPipeline

print("Downloading Stable Diffusion Model...")
StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", local_files_only=False)

print("Models downloaded successfully!")



# Load the Stable Diffusion pipeline once during server initialization
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

def generate_image(shirt_desc, pant_desc):
    """
    Generate an AI model image wearing the specified clothes.

    Args:
        shirt_desc (str): Description of the shirt (e.g., "white formal shirt").
        pant_desc (str): Description of the pants (e.g., "black formal pants").

    Returns:
        str: Path to the generated output image.
    """
    prompt = f"A male model wearing a {shirt_desc} and {pant_desc}, studio lighting, photorealistic"
    image = pipe(prompt).images[0]
    output_path = f"static/generated_model_{random.randint(1000, 9999)}.jpg"
    image.save(output_path)
    return output_path



def download_image(url):
    """Download an image from a URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise ValueError(f"Failed to download image from {url}")

def generate_fashion_image(shirt_url, pant_url, model_type):
    """
    Generate an image of a virtual model wearing the clothes from the given URLs.

    Args:
        shirt_url (str): URL of the shirt image.
        pant_url (str): URL of the pants image.
        model_type (str): Selected AI model type.

    Returns:
        str: Path to the generated output image.
    """
    try:
        # Download images
        shirt_img = download_image(shirt_url)
        pant_img = download_image(pant_url)

        # AI Model Logic
        output_path = f"clothing/static/generated_output_{random.randint(1000, 9999)}.jpg"
        
        # For demo purposes, we merge the images (Replace this with actual AI logic)
        combined_image = Image.new('RGB', (shirt_img.width, shirt_img.height + pant_img.height))
        combined_image.paste(shirt_img, (0, 0))
        combined_image.paste(pant_img, (0, shirt_img.height))
        combined_image.save(output_path)

        return output_path
    except Exception as e:
        return f"Error: {e}"
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP Initialized Successfully")

def get_image_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    description = processor.decode(outputs[0], skip_special_tokens=True)
    return description
from .utils import get_image_description
from django.http import JsonResponse
import os

def upload_and_describe_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        # Save the image temporarily
        temp_image_path = f"temp/{image_file.name}"
        with open(temp_image_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        try:
            # Get the description of the image
            description = get_image_description(temp_image_path)
            os.remove(temp_image_path)  # Clean up the temporary file
            return JsonResponse({"status": "success", "description": description})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})
    return JsonResponse({"status": "error", "message": "No image provided"})
