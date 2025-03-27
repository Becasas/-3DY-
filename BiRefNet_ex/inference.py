# 对单张图像推理，by wjx
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torch
from matplotlib import pyplot as plt

from models.birefnet import BiRefNet

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

def load_model(path='BiRefNet-massive-epoch_240.pth'):
    model = BiRefNet(bb_pretrained=False)
    state_dict = [path][0]
    if os.path.exists(state_dict):
        birefnet_dict = torch.load(state_dict,map_location="cpu")
        unwanted_prefix = '_orig_mod.'
        for k, v in list(birefnet_dict.items()):
            if k.startswith(unwanted_prefix):
                birefnet_dict[k[len(unwanted_prefix):]] = birefnet_dict.pop(k)
        model.load_state_dict(birefnet_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print('Model loaded')
    return model

def inference_image(model,image_original,shape):
    # 输入模型和图像，需要RGB格式
    image = array_to_pil_image(image_original)
    # image_shape = image_original.shape[:2]
    proce = ImagePreprocessor()
    image = proce.proc(image).unsqueeze(0).cuda()
    with torch.no_grad():
        pred_tensor = model(image)[-1].sigmoid()
    pred_tensor = pred_tensor.cpu()
    pred = torch.nn.functional.interpolate(pred_tensor, size=(shape[1], shape[0]), mode='bilinear',align_corners=True).squeeze().numpy()
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    pred = np.repeat(np.expand_dims(pred, axis=-1), 3, axis=-1)
    image_original = cv2.resize(image_original,shape,interpolation=cv2.INTER_LINEAR)
    image_pred = (pred * image_original).astype(np.uint8)
    # mask = (pred*255).astype(np.uint8)
    return image_pred

if __name__ == '__main__':
    torch.set_num_threads(1)
    cuda = '5'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    model = load_model()
    img_path = '/data1/wjx/S003/input/image/example3.png'
    img_or = cv2.imread(img_path)
    img_or = cv2.cvtColor(img_or, cv2.COLOR_BGR2RGB)
    image_shape = img_or.shape[:2]
    img = array_to_pil_image(img_or)
    proce = ImagePreprocessor()
    img = proce.proc(img).unsqueeze(0).cuda()

    with torch.no_grad():
        pred_tensor = model(img)[-1].sigmoid()
    pred_tensor = pred_tensor.cpu()
    pred = torch.nn.functional.interpolate(pred_tensor, size=image_shape, mode='bilinear',align_corners=True).squeeze().numpy()
    print(pred.shape)
    pred = np.repeat(np.expand_dims(pred, axis=-1), 3, axis=-1)
    # image_pred.putalpha(pred_pil)
    # image = cv2.cvtColor(img_or, cv2.COLOR_BGR2RGB)
    image_pred = (pred * img_or).astype(np.uint8)
    mask = (pred*255).astype(np.uint8) # np.zeros_like(image_pred)
    image_pred = cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB)
    cv2.imwrite('/data1/wjx/S003/output/example3.png',image_pred)
    plt.imshow(img_or)
    plt.axis('off')
    plt.show()
    plt.imshow(image_pred)
    plt.axis('off')
    plt.show()
    plt.imshow(mask)
    plt.axis('off')
    plt.show()