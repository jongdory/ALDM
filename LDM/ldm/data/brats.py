import os
import numpy as np
import pandas

import torch
from torch.utils.data import Dataset

from monai import transforms
from monai.data import Dataset as MonaiDataset


def get_transforms(phase="train"):
    modalities = ["t1", "t1ce", "t2", "flair"]

    if phase == "train":
        train_transforms = transforms.Compose(
            [
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=0, allow_missing_keys=True),
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=1, allow_missing_keys=True),
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=2, allow_missing_keys=True),
            ]
        )
    
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=modalities, allow_missing_keys=True),
            transforms.AddChanneld(keys=modalities, allow_missing_keys=True),
            transforms.Orientationd(keys=modalities, axcodes="RAS", allow_missing_keys=True),
            transforms.EnsureTyped(keys=modalities, allow_missing_keys=True),
            transforms.CropForegroundd(keys=modalities, source_key="t1", margin=0, allow_missing_keys=True),
            transforms.SpatialPadd(keys=modalities, spatial_size=(144, 192, 144), allow_missing_keys=True),
            transforms.CenterSpatialCropd(keys=modalities, roi_size=(144, 192, 144), allow_missing_keys=True),
            transforms.ScaleIntensityRangePercentilesd(keys=modalities, lower=0.5, upper=99.5, b_min=-1, b_max=1, allow_missing_keys=True),
            train_transforms if phase == "train" else transforms.Compose([])
        ]
    )


def get_brats_dataset(data_path, csv_path=None, phase="train"):
    transform = get_transforms(phase=phase)
    
    datalist = []
    if csv_path is not None:
        df = pandas.read_csv(csv_path)
        for sub_id in df["id"].tolist():
            split_list = df[df["id"] == sub_id]["split"].tolist()
            if split_list and split_list[0] == phase:
                datalist.append(sub_id)
    else:
        if phase == "train": 
            datalist = os.listdir(data_path)
        else:
            datalist = os.listdir(data_path)[-10:]

    data = []

    for subject in datalist:
        sub_path = os.path.join(data_path, subject)
        
        if os.path.exists(sub_path) == False: 
            continue

        t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz")
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz")
        t2 = os.path.join(sub_path, f"{subject}_t2.nii.gz")
        flair = os.path.join(sub_path, f"{subject}_flair.nii.gz")

        data.append({"t1":t1, "t1ce":t1ce, "t2":t2, "flair":flair, "subject_id": subject, "path": t1})
                    
    print(phase, " num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)


class Brain3DBase(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    
class BraTSbase(Brain3DBase):
    def __init__(self, source=None, target=None):
        self.modalities = ["t1", "t1ce", "t2", "flair"]
        self.source = source
        self.target = target
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = dict(self.data[i])

        if self.source is None:
            source, target = np.random.choice(self.modalities, size=2, replace=False)
        else:
            source, target = self.source, self.target

        item["source"] = item[source]
        item["target"] = item[target]
        item["target_class"] = torch.tensor(self.modalities.index(target))

        return item


class BraTS2021Train(BraTSbase):
    def __init__(self, data_path, csv_path=None, phase="train"):
        super().__init__()
        self.data = get_brats_dataset(data_path, csv_path, phase)

class BraTS2021Test(BraTSbase):
    def __init__(self, data_path, csv_path=None, phase="test", source=None, target=None):
        super().__init__()
        self.data = get_brats_dataset(data_path, csv_path, phase)
