import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Load trained model
def load_model(model_path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load saved model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Gender Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load model and predict
    model = load_model("point_resnet18_best.pth")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    if predicted_class == 1:
        class_name = 'Male'
    else:
        class_name = 'Female'

    st.write(f"Predicted Class: {class_name}")
