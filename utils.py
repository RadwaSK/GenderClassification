from torch.utils.data import DataLoader
from DAiSEE_Dataset import DAiSEE_Dataset
import torchvision
from LandmarksTransform import LandmarksTransform


def get_t_v_dataloaders(batch_size, train_csv, val_csv):
    transform = torchvision.transforms.Compose([LandmarksTransform()])
    train_dataset = DAiSEE_Dataset(train_csv, transform)
    val_dataset = DAiSEE_Dataset(val_csv, transform)
    return {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True),
            'validation': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)}


def get_test_dataloader(batch_size, test_csv, transform):
    transform = torchvision.transforms.Compose([LandmarksTransform()])
    test_dataset = DAiSEE_Dataset(test_csv, transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.1):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter > self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0
