import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
from PIL import Image, ImageDraw, ImageFont

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = "Hair Diseases - Final"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
class_names = train_dataset.classes 
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  
model.load_state_dict(torch.load("hair_disease_classifier.pth"))
model = model.to(device)
model.eval()  

def predict_and_save(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue 
        image = Image.open(image_path).convert("RGB")
        original_image = image.copy()  
        image = transform(image).unsqueeze(0).to(device)  

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]

        draw = ImageDraw.Draw(original_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40) 
        except IOError:
            font = ImageFont.load_default()  

       
        text_size = draw.textbbox((0, 0), f"Label: {label}", font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

       
        draw.rectangle([(5, 5), (5 + text_width + 10, 5 + text_height + 10)], fill="yellow")  
        draw.text((10, 10), f"Label: {label}", fill="red", font=font)

        output_path = os.path.join(output_folder, image_name)
        original_image.save(output_path)
        print(f"Saved {output_path} with label: {label}")

test_image_folder = "test_images"  
output_folder = "labeled_images"  

predict_and_save(test_image_folder, output_folder)
