from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import torch
from torch import nn, optim
import os
#from Tomato_Training import img_size, class_names

app = Flask(__name__)

img_size=(256,256)
class_names = ['Tomato_Bacterial_spot',
             'Tomato_Early_blight',
             'Tomato_Late_blight',
             'Tomato_Leaf_Mold',
             'Tomato_Septoria_leaf_spot',
             'Tomato_Spider_mites_Two_spotted_spider_mite',
             'Tomato__Target_Spot',
             'Tomato__Tomato_YellowLeaf__Curl_Virus',
             'Tomato__Tomato_mosaic_virus',
             'Tomato_healthy']

def load_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * img_size[0] // 8 * img_size[1] // 8, 64),
        nn.ReLU(),
        nn.Linear(64, len(class_names)),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))  # Load the weights you saved earlier
    model.eval()
    return model

def process_image(image_path):
    image = Image.open(image_path)
    # Apply the same transformations you used when training
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join('static', secure_filename(image_file.filename))
            image_file.save(image_location)
            image = process_image(image_location)
            model = load_model()
            prediction = model(image)  # Perform the prediction
            prediction = prediction.argmax().item()  # Get the index of the highest value
            prediction = class_names[prediction]
            return render_template('index.html', prediction=prediction, image_location=image_location)
    return render_template('index.html', prediction=0, image_location=None)

if __name__ == '__main__':
    app.run(debug=True)
