import torch
import joblib
from PIL import Image
from torchvision import transforms

from model import Net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
label_encoder = joblib.load("models/label_encoder.pkl")
num_classes = len(label_encoder.classes_)

model = Net(num_classes)

model.load_state_dict(
    torch.load("models/best_model.pth", map_location=device)
)

model.to(device)
model.eval()
def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence = probabilities.max().item()

        prediction = torch.argmax(output, dim=1).item()

    label = label_encoder.inverse_transform([prediction])[0]

    return label, confidence
if __name__ == "__main__":

    image_path = "sample_images/test.jpg"   # Replace with an actual image path

    label, confidence = predict_image(image_path)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")