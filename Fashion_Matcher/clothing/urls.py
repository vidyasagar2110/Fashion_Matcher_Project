from django.urls import path
from . import views

urlpatterns = [
    path("upload-and-describe/", views.upload_and_describe_image, name="upload_and_describe_image"),
    path("male/", views.male, name="male"),
]
