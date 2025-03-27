# 对视频逐帧推理,wjx 2024/07/17
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import datetime

from tqdm import tqdm

from models.birefnet import BiRefNet
import torchvision.transforms as transforms
import torch
from inference import array_to_pil_image,ImagePreprocessor,load_model,inference_image
class video():
    # 定义一个视频对象
    def __init__(self):
        pass
# defining function for creating a writer (for mp4 videos)
def create_video_writer(video_cap,out_shape,output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, out_shape)
    return writer
def stack(frame,black_mask,frame_pred,out_shape, source_shape):
    # 将原视频，黑色遮罩，分割结果 三者堆叠
    # if source_shape[0]/source_shape[1] != 16/9:
    #     # 帧的画幅不是16:9，不能直接resize
    #     black_mask_new = np.zeros_like(black_mask)
    #     frame_new = np.zeros_like(black_mask)
    #     # 中间对齐
    #     assert source_shape[0] == out_shape[0] # 源视频为1920:X，
    #     start_w = int(out_shape[0]/2-source_shape[0]/2)
    #     end_w = int(out_shape[0]/2+source_shape[0]/2)
    #     start_h = int(out_shape[1]/2-source_shape[1]/2)
    #     end_h = int(out_shape[1]/2+source_shape[1]/2)
    #     black_mask = cv2.resize(black_mask, source_shape)
    #     black_mask_new[start_h:end_h,:,:] = black_mask
    #     frame_new[start_h:end_h,:,:] = frame
    #     # beishu = out_shape[1]/source_shape[1] # 代表放大倍数，对预测结果放大
    #     # scalex = int(beishu * out_shape[0])
    #     # scaley = int(beishu * out_shape[1])
    #     # frame_pred = cv2.resize(frame_pred, (scalex,scaley))
    #     # # frame_pred_new = np.zeros(out_shape)
    #     # frame_pred_new = frame_pred[int(scaley/2-out_shape[1]/2):int(scaley/2+out_shape[1]/2),int(scalex/2-out_shape[0]/2):int(scalex/2+out_shape[0]/2),:]
    # else:
    #     # 帧的画幅与预设16:9（1920:1080）相同,直接resize不会形变
    frame_new = cv2.resize(frame, out_shape)
    frame_pred_new = frame_pred
    frame_new[black_mask <= 127] = 0 # 0是黑色，255是白色, 黑色遮罩将其置为0
    # 注意此为灰度而非0、1二值，因此需要设置为127更平滑 2024/7/26，后期可以考虑用透明度的概念优化
    # ,为了使用超级键，应当用0、255二值
    # new_frame = frame.copy()
    # 找到V1中（0,0,0）黑色像素的位置
    black_pixels = np.all(frame_new == [0, 0, 0], axis=2)

    # 用V3替换V1中的黑色部分
    frame_new[black_pixels] = frame_pred_new[black_pixels]
    # new_frame = np.concatenate((frame,black_mask,frame_pred),axis=1)
    # plt.imshow(frame_new)
    # plt.axis('off')
    # plt.show()
    frame_new = cv2.cvtColor(frame_new, cv2.COLOR_RGB2BGR)
    return frame_new

if "__name__ == __main__":
    torch.set_num_threads(1) # 设置线程数
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    # 模型加载
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    print(current_time + "   starting")
    model = load_model()
    video_path = "/data1/wjx/S003/input/entire.mp4" # **** # # '/data1/wjx/S004/Segment-and-Track-Anything-main/assets/cars.mp4' #'"/data1/wjx/S003/input/advs.mp4" # 视频路径
    name = video_path.split('/')[-1]
    road = name.split('.')[0]
    # out_path = video_path.replace("input","output")  # "/data1/wjx/S003/output/divided1.mp4" # 输出路径
    out_path = video_path.replace(name,road+'_out.mp4') # 在原路径生成 + _out.mp4
    black_mask = cv2.imread('/data1/wjx/S003/input/black_mask/black_mask.png',cv2.IMREAD_COLOR) # , cv2.IMREAD_GRAYSCALE)
    # black_mask = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2RGB)
    # out_shape = [1920,872] # 注意birefnet的输入最好是1024*1024，其输出也是1024,1024
    cap = cv2.VideoCapture(video_path)
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))] # (W,H)
    out_shape = [1920,1080] #source_shape  # 可以指定具体的像素或者与源视频保持一致，注意源视频应当尽量与1024x1024更接近
    black_mask = cv2.resize(black_mask, out_shape) # 将黑色遮罩调整为输出相同像素
    # 创建进度条
    progress_bar = tqdm(total=total_frames, desc='Processing frames')
    writer = create_video_writer(cap, out_shape, out_path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_frame = inference_image(model, frame, out_shape)
        # 模型的分割结果是（H，W，C）的RGB np.array
        # 1、有抖动问题,想办法防抖动，2、一些场景难以分割
        # 将原视频，黑色遮罩，分割结果 三者堆叠合并
        new_frame = stack(frame, black_mask, pred_frame, out_shape, source_shape)
        # new_frame = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB)
        writer.write(new_frame)
        # 更新进度条
        progress_bar.update(1)
        cv2.waitKey(1)
writer.release()
# 关闭进度条
progress_bar.close()
current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
print("Out file:" + out_path)
print(current_time + "    Finished!")
