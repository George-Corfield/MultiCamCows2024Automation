import torch
import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import HDBSCAN
from simCLR import SimCLR
from umap.umap_ import UMAP
from cowDataset import CustomDataset
from torch.utils.data import DataLoader
import random
import torchvision.transforms as T
import numpy as np
import pickle


transform = T.Compose([T.Resize((128, 128)),
                       T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])


def cluster_results(embeddings, num_classes=90):
    clf = HDBSCAN(min_cluster_size=(
        (len(embeddings)/num_classes)*0.48), min_samples=15)
    labels = clf.fit_predict(embeddings)
    hdbscan_label_mask = labels != -1
    train_embed = embeddings[hdbscan_label_mask]
    train_labels = labels[hdbscan_label_mask]
    return train_embed, train_labels


def get_dataset(label_map, dir, size=None):
    inputs = np.loadtxt(dir, dtype="U")
    inputs = sorted(inputs)
    if size != None:
        inputs = random.sample(inputs, size)
    inputs = np.array(inputs)
    print(len(inputs))
    images = CustomDataset("./images/images", inputs, label_map, transform)
    loader = DataLoader(images, batch_size=32, num_workers=0, shuffle=True)
    return loader


def get_embeddings(model, dataloader):
    features = []
    labels = []

    with torch.no_grad():
        for images, targets, _, _ in dataloader:
            images = images.to("cuda")
            out = model.base(images)
            output = out.view(out.shape[0], -1)
            features.append(output.cpu().detach().numpy())

            labels.append(targets.numpy())

    return np.vstack(features), np.hstack(labels)


if __name__ == "__main__":
    save_dir = f"{sys.argv[1]}/models"
    data_dir = f"{sys.argv[1]}/images"
    model = SimCLR.load_from_checkpoint(
        f'{save_dir}/best-model-checkpoint.ckpt', epochs=20)
    model.eval()
    label_map = {}
    index = 0
    for date in sorted(os.listdir(data_dir)):
        date_path = os.path.join(data_dir, date)

        for camera in sorted(os.listdir(date_path)):
            camera_path = os.path.join(date_path, camera)

            for idx in sorted(os.listdir(camera_path)):
                label_map[(date, camera, idx)] = index
                index += 1

    dataloader = get_dataset(label_map, f"{save_dir}/train_data_paths.txt")
    features, labels = get_embeddings(model, dataloader)
    umap = UMAP(n_components=10, random_state=42)
    embeddings = umap.fit_transform(features)
    embeddings, unified_labels = cluster_results(embeddings)
    knn = KNeighborsClassifier().fit(embeddings, labels)

    pickle.dump(knn, open(f"{save_dir}/knn_pickle", 'wb'))
