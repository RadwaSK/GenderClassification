from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms


class DAiSEE_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)

        len0 = len(self.df[self.df.labels == 0])
        len1 = len(self.df[self.df.labels == 1])
        print('Male count =', len0, '\nFemale count =', len1)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index, 1]
        assert label.dtype == int
        assert label == 1 or label == 0

        path = self.df.iloc[index, 0]
        data = self.transform(path)
        return data, label
