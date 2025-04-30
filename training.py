import sys
import os

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode.lower() == "train":
        data_dir = sys.argv[2]
        save_dir = sys.argv[3]
        # tracking
        sys.argv = ['track.py', data_dir, save_dir]
        with open("track.py") as file:
            exec(file.read())
        # training
        sys.argv = ['train-simclr.py', save_dir]
        with open("train-simclr.py") as file:
            exec(file.read())

        # clustering
        sys.argv = ['cluster.py', save_dir]
        with open("cluster.py") as file:
            exec(file.read())

    # example downstream task
    elif mode.lower() == "infer":
        data_dir = sys.argv[2]
        save_dir = sys.argv[3]
        if os.path.exists(f"{save_dir}/models/best-model-checkpoint.ckpt") and os.path.exists(f"{save_dir}/models/knn_pickle"):
            sys.argv = ['tracking_inference.py', data_dir, save_dir]
            with open("tracking_inference.py") as file:
                exec(file.read())
        else:
            print(
                "Model and/or Classifier do not exist, Try training or altering save directories")
