from ultralytics import YOLO
import os
import cv2
import sys

COWS = {}
MAX_ID = 1


def track_video(video, idx, cam_settings, save_day, save_dir):
    global MAX_ID
    cap = cv2.VideoCapture(video)

    success, frame = cap.read()

    count = 0
    out_counter = 0

    while success:
        left_cutoff = int(cam_settings['left'] * frame.shape[1])
        right_cutoff = int(cam_settings['right'] * frame.shape[1])
        top_cutoff = int(cam_settings['top'] * frame.shape[0])
        if count % 5 == 0:
            results = model.track(frame, persist=True,
                                  tracker="botsort.yaml", verbose=False)

            for result in results:
                for box in result.boxes:
                    try:
                        id = box.id.int().detach().numpy()[0]
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                        area = (x_max - x_min) * (y_max - y_min)
                        if (y_max + y_min)/2 > top_cutoff and x_max < right_cutoff and x_min > left_cutoff:
                            if ((area > 150000 and area < 300000) and (x_max - x_min) <= (y_max-y_min)):
                                crop = frame[int(y_min)+10:int(y_max)-20,
                                             int(x_min): int(x_max)]
                                if cam_settings['rotation']:
                                    crop = cv2.rotate(crop, cv2.ROTATE_180)
                                if id in COWS:
                                    out_id, pic_count = COWS[id]
                                    pic_count += 1
                                    cv2.imwrite(
                                        f"{save_dir}/{save_day}/camera-{idx}/id_{str(out_id)}/img_{str(pic_count)}.jpg", crop)
                                    COWS[id] = (out_id, pic_count)
                                else:
                                    os.mkdir(
                                        f"{save_dir}/{save_day}/camera-{idx}/id_{str(MAX_ID)}")
                                    COWS[id] = (MAX_ID, 1)
                                    cv2.imwrite(
                                        f"{save_dir}/{save_day}/camera-{idx}/id_{str(MAX_ID)}/img_1.jpg", crop)
                                    MAX_ID += 1
                                out_counter += 1
                    except Exception as e:
                        pass
        count += 1
        success, frame = cap.read()
    return out_counter


if __name__ == "__main__":
    video_dir = sys.argv[1]
    save_dir = sys.argv[2]
    days_available = os.listdir(video_dir)
    for day in days_available:
        os.makedirs(f'{save_dir}/images/{day}/camera-1', exist_ok=True)
        os.makedirs(f'{save_dir}/images/{day}/camera-2', exist_ok=True)

    model = YOLO('yolov8m.pt')

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
            COWS = {}
            MAX_ID = 1
            for video in set:
                if idx == 0:
                    out_imgs = track_video(
                        video, str(idx+1), camera_1_args, days_available[day_id], save_dir)
                elif idx == 1:
                    out_imgs = track_video(
                        video, str(idx+1), camera_2_args, days_available[day_id], save_dir)
                print(
                    f"Finished video {video}..... Number of images: {out_imgs}")
                print(f"Video: {video}, camera: {idx}")
