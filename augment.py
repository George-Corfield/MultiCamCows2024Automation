import torchvision.transforms as T
import torch.nn as nn


class Augmentation:
    """Transforms any data into two correlated views of the same example creating a positive pair"""

    def __init__(self, img_size=128, s=1):

        self.transform = T.Compose(
            [T.ToTensor(),
             T.Resize(size=(img_size, img_size)),
             nn.Sequential(
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.ElasticTransform(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            )
            ]

        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)
