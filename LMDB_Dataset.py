from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms


class IMDB_Dataset(Dataset):
    def __init__(self, csv_file, ratio=None, transform=None):
        df = pd.read_csv(csv_file)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        if ratio:
            male_df = df[df.labels == 0]
            female_df = df[df.labels == 1]
            count = ratio * len(df)
            female_count = min(count // 2, len(female_df))
            male_count = min(female_count, count // 2)
            female_ratio = female_count / len(female_df)
            male_ratio = male_count / len(male_df)
            female_df = female_df.sample(frac=female_ratio)
            male_df = male_df.sample(frac=male_ratio)
            self.df = pd.concat((male_df, female_df))
        else:
            self.df = df.sample(frac=1)

        len0 = len(self.df[self.df.labels == 'male'])
        len1 = len(self.df[self.df.labels == 'female'])
        print('Male count =', len0, '\nFemale count =', len1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        gender = self.df.iloc[index, 1]
        #         assert label.dtype == int
        assert gender == 'male' or gender == 'female'

        label = 0 if gender == 'male' else 1
        #         print(label)

        path = self.df.iloc[index, 0]
        #         print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data = self.transform(img)
        return data, label