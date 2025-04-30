from torch.utils.data.sampler import BatchSampler
import random


class CowSampler(BatchSampler):

    def __init__(self, data, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle

        self.camera_splits = {
            'camera_1': [],
            'camera_2': [],
        }
        for idx, (_, label, cam) in enumerate(data):
            self.camera_splits[cam].append((idx, label))

        if self.shuffle:
            for cam_id in self.camera_splits.keys():
                random.shuffle(self.camera_splits[cam_id])
        else:
            for cam_id in self.camera_splits.keys():
                self.camera_splits[cam_id].sort(key=lambda x: x[1])

        self.batches = self.generate_batches()

    def generate_batches(self):
        batches = []
        print(self.camera_splits.keys())
        for cam_id in self.camera_splits.keys():
            num_samples = len(self.camera_splits[cam_id])//self.batch_size
            for i in range(num_samples):
                batch = [x[0]
                         for x in self.camera_splits[cam_id][:self.batch_size]]
                self.camera_splits[cam_id] = self.camera_splits[cam_id][self.batch_size:]
                batches.append(batch)
            if len(self.camera_splits[cam_id]) > 0:
                batches.append([x[0] for x in self.camera_splits[cam_id]])
                self.camera_splits[cam_id] = []

        if (self.shuffle):
            random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
