from django.shortcuts import render
from django.http import JsonResponse
from .utils import get_image_description, generate_fashion_image
import os

def upload_and_describe_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        temp_image_path = f"temp/{image_file.name}"
        os.makedirs("temp", exist_ok=True)
        with open(temp_image_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        try:
            description = get_image_description(temp_image_path)
            os.remove(temp_image_path)
            return JsonResponse({"status": "success", "description": description})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})
    return JsonResponse({"status": "error", "message": "Invalid request."})

def male(request):
    if request.method == "POST":
        if "image" in request.FILES:
            return upload_and_describe_image(request)

        # Logic for URL-based matching
        shirt_url = request.POST.get("top_url")
        pant_url = request.POST.get("bottom_url")
        model_type = request.POST.get("model")
        
        if not shirt_url or not pant_url or not model_type:
            return JsonResponse({"status": "error", "message": "All fields are required."})
        
        try:
            output_image_path = generate_fashion_image(shirt_url, pant_url, model_type)
            return JsonResponse({"status": "success", "result": output_image_path})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return render(request, "male.html")
