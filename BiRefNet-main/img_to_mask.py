# Imports
from PIL import Image
import torch
import os
import cv2
from torchvision import transforms
from IPython.display import display
from glob import glob
from models.birefnet import BiRefNet
import matplotlib.pyplot as plt

# Load Model
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
model = BiRefNet(bb_pretrained=False)
state_dict = ['BiRefNet-massive-epoch_240.pth'][0]
if os.path.exists(state_dict):
    birefnet_dict = torch.load(state_dict, map_location="cpu")
    unwanted_prefix = '_orig_mod.'
    for k, v in list(birefnet_dict.items()):
        if k.startswith(unwanted_prefix):
            birefnet_dict[k[len(unwanted_prefix):]] = birefnet_dict.pop(k)
    model.load_state_dict(birefnet_dict)
model = model.to(device)
model.eval()
print('BiRefNet is ready to use.')

# Input Data
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
imagepath = '/data1/wjx/S003/input/image/example3.png'
image = Image.open(imagepath).convert('RGB')
# image = cv2.imread('/data1/wjx/S003/input/image/example3.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_images = transform_image(image).unsqueeze(0).to('cuda')

# Prediction
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid().cpu()
pred = preds[0].squeeze()

# Show Results
pred_pil = transforms.ToPILImage()(pred*255)
# Scale proportionally with max length to 1024 for faster showing
scale_ratio = 1024 / max(image.size)
scaled_size = (int(image.size[0] * scale_ratio), int(image.size[1] * scale_ratio))
image_masked = image.resize((1024, 1024))
image_masked.putalpha(pred_pil)
display(image_masked.resize(scaled_size))
display(image.resize(scaled_size))
display(pred_pil.resize(scaled_size))

# 显示
plt.imshow(image_masked.resize(scaled_size))
plt.axis('off')
plt.show()
plt.imshow(image.resize(scaled_size))
plt.axis('off')
plt.show()
plt.imshow(pred_pil.resize(scaled_size))
plt.axis('off')
plt.show()