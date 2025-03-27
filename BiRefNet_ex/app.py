import os
from glob import glob
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import gradio as gr
import spaces
from gradio_imageslider import ImageSlider

torch.set_float32_matmul_precision('high')
torch.jit.script = lambda f: f

from models.birefnet import BiRefNet
from config import Config


config = Config()
device = config.device


def array_to_pil_image(image, size=(1024, 1024)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(image).convert('RGB')
    return image


class ImagePreprocessor():
    def __init__(self, resolution=(1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            # transforms.Resize(resolution),    # 1. keep consistent with the cv2.resize used in training 2. redundant with that in path_to_image()
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image):
        image = self.transform_image(image)
        return image


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


# def predict(image_1, image_2):
#     images = [image_1, image_2]
@spaces.GPU
def predict(image, resolution):
    resolution = f"{image.shape[1]}x{image.shape[0]}" if resolution == '' else resolution
    # Image is a RGB numpy array.
    resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]
    images = [image]
    image_shapes = [image.shape[:2] for image in images]
    images = [array_to_pil_image(image, resolution) for image in images]

    image_preprocessor = ImagePreprocessor(resolution=resolution)
    images_proc = []
    for image in images:
        images_proc.append(image_preprocessor.proc(image))
    images_proc = torch.cat([image_proc.unsqueeze(0) for image_proc in images_proc])

    with torch.no_grad():
        scaled_preds_tensor = model(images_proc.to(device))[-1].sigmoid()   # BiRefNet needs an sigmoid activation outside the forward.
    preds = []
    for image_shape, pred_tensor in zip(image_shapes, scaled_preds_tensor):
        if device == 'cuda':
            pred_tensor = pred_tensor.cpu()
        preds.append(torch.nn.functional.interpolate(pred_tensor.unsqueeze(0), size=image_shape, mode='bilinear', align_corners=True).squeeze().numpy())
    image_preds = []
    for image, pred in zip(images, preds):
        image = image.resize(pred.shape[::-1])
        pred = np.repeat(np.expand_dims(pred, axis=-1), 3, axis=-1)
        image_preds.append((pred * image).astype(np.uint8))

    return image, image_preds[0]


examples = [[_] for _ in glob('materials/examples/*')][:]

# Add the option of resolution in a text box.
for idx_example, example in enumerate(examples):
    examples[idx_example].append('1024x1024')
examples.append(examples[-1].copy())
examples[-1][1] = '512x512'

demo = gr.Interface(
    fn=predict,
    inputs=['image', gr.Textbox(lines=1, placeholder="Type the resolution (`WxH`) you want, e.g., `512x512`. Higher resolutions can be much slower for inference.", label="Resolution")],
    outputs=ImageSlider(),
    examples=examples,
    title='Online demo for `Bilateral Reference for High-Resolution Dichotomous Image Segmentation`',
    description=('Upload a picture, our model will give you the binary maps of the highly accurate segmentation of the salient objects in it. :)'
                 '\nThe resolution used in our training was `1024x1024`, which is too much burden for the huggingface free spaces like this one (cost nearly 40s). Please set resolution as more than `768x768` for images with many texture details to obtain good results!\n Ours codes can be found at https://github.com/ZhengPeng7/BiRefNet.')
)
demo.launch(debug=True)
