import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

if __name__ == '__main__':
    sam = sam_model_registry["vit_h"](checkpoint="/data1/wjx/S003/model/sam_vit_h_4b8939.pth")
    device = 'cuda'
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    example = '/data1/wjx/S003/input/example1.png'
    example = np.array(Image.open(example))
    example = example[ :, :, :3] # 必须是RGB格式，长边是1024
    masks = mask_generator.generate(example)
    print(masks)