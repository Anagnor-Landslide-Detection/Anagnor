import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        df = pd.read_csv(csv_file)
        print(df["event_time"])

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample2

csv_file = "nasa_global_landslide_catalog_point.csv"
df = pd.read_csv(csv_file)
print(df["event_date"])
print(df["latitude"])
print(df["longitude"]) 