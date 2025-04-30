from simCLR import SimCLR
import os
import numpy as np
from cowDataset import CustomDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
from umap import UMAP
import hdbscan
from sklearn.neighbors import KNeighborsClassifier
import cv2
from ultralytics import YOLO
from PIL import Image
import sys
import pickle


FPS = 25
W = 2560
H = 1440


def transform_video(video, embed_model, track_model, umap, knn, out, cam_settings):
    cap = cv2.VideoCapture(video)

    transform = T.Compose([T.Resize((128, 128)),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])])

    success, frame = cap.read()

    while success:
        left_cutoff = int(cam_settings['left'] * frame.shape[1])
        right_cutoff = int(cam_settings['right'] * frame.shape[1])
        top_cutoff = int(cam_settings['top'] * frame.shape[0])

        results = track_model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                area = (x_max - x_min) * (y_max - y_min)
                if (y_max + y_min)/2 > top_cutoff and x_max < right_cutoff and x_min > left_cutoff:
                    if ((area > 150000 and area < 300000) and (x_max - x_min) <= (y_max-y_min)):
                        crop = frame[int(y_min)+10:int(y_max)-20,
                                     int(x_min): int(x_max)]
                        if cam_settings['rotation']:
                            crop = cv2.rotate(crop, cv2.ROTATE_180)
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_pil = Image.fromarray(crop_rgb)
                        transformed_crop = transform(crop_pil)
                        transformed_crop = transformed_crop.unsqueeze(0)
                        transformed_crop = transformed_crop.to('cuda')
                        with torch.no_grad():
                            feature = embed_model.base(transformed_crop)
                            output = feature.view(feature.shape[0], -1)
                            output = output.cpu().detach().numpy()
                        embedding = umap.transform(output)
                        label = knn.predict(embedding)
                        label = str(label[0])

                        cv2.rectangle(frame, (int(x_min), int(y_min)),
                                      (int(x_max), int(y_max)), (255, 0, 0), 2)
                        text_x, text_y = int(x_min), int(y_min) - 10
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (text_x, text_y - text_height - 5),
                                      (text_x + text_width + 5, text_y), (255, 0, 0), -1)
                        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        success, frame = cap.read()


if __name__ == "__main__":
    video_dir = sys.argv[1]
    save_dir = sys.argv[2]
    embed_model = SimCLR.load_from_checkpoint(
        f"{save_dir}/models/best-model-checkpoint", epochs=10)
    embed_model.eval()
    track_model = YOLO("yolov8m.pt")
    umap = UMAP(n_components=10, random_state=42)
    days_available = os.listdir(video_dir)

    knn = pickle.load(open(f"{save_dir}/models/knn_pickle", 'rb'))

    paths = [[f'{video_dir}/{day}/camera-1/', f'{video_dir}/{day}/camera-2/']
             for day in os.listdir(video_dir)]

    videos = [[[p+f for f in os.listdir(p)]
               for p in day_path] for day_path in paths]

    camera_1_args = {'left': 0.15, 'right': 1.0,
                     'top': 0.15, 'rotation': False}
    camera_2_args = {'left': 0.17, 'right': 0.63,
                     'top': 0.10, 'rotation': True}

    for day_id, day in enumerate(videos):
        for idx, set in enumerate(day):
            outfile = f"{save_dir}/out_videos/{days_available[day_id]}/camera_{idx+1}.avi"

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(outfile, fourcc, FPS, (W, H))
            for vid in set:
                if idx == 0:
                    transform_video(vid, embed_model, track_model,
                                    umap, knn, out, camera_1_args)
                else:
                    transform_video(vid, embed_model, track_model,
                                    umap, knn, out, camera_2_args)
                print(f"finished video: {vid}")
            out.release()
