import pandas as pd
import os
from os.path import join


labels_path = '../emotiondetection/dataset/DAiSEE/GenderClips'
data_path = '../emotiondetection/dataset/DAiSEE/Data'

Females_txt = open(join(labels_path, 'Females'), 'r')
females_videos = Females_txt.readlines()
females_videos = [m[:m.find('.')] for m in females_videos]

Males_txt = open(join(labels_path, 'Males'), 'r')
males_videos = Males_txt.readlines()
males_videos = [m[:m.find('.')] for m in males_videos]

female_subjects = [s[:6] for s in females_videos]
male_subjects = [s[:6] for s in males_videos]

train_subjects = os.listdir(join(data_path, 'Train_frames'))
val_subjects = os.listdir(join(data_path, 'Validation_frames'))
test_subjects = os.listdir(join(data_path, 'Test_frames'))

train_males_csv = pd.DataFrame([], columns=['path', 'labels'])
val_males_csv = pd.DataFrame([], columns=['path', 'labels'])
test_males_csv = pd.DataFrame([], columns=['path', 'labels'])

train = []
val = []
test = []
for videos, label in zip([females_videos, males_videos], [1, 0]):
    for v in videos:
        s = v[:6]
        if s in train_subjects:
            path = join(data_path, 'Train_frames', s, v)
            for i in range(1, 6):
                v_path = join(path, str(i))
                images = [[join(v_path, im), label] for im in os.listdir(v_path)]
                train += images
        elif s in val_subjects:
            path = join(data_path, 'Validation_frames', s, v)
            for i in range(1, 6):
                v_path = join(path, str(i))
                images = [[join(v_path, im), label] for im in os.listdir(v_path)]
                val += images
        elif s in test_subjects:
            path = join(data_path, 'Test_frames', s, v)
            for i in range(1, 6):
                v_path = join(path, str(i))
                images = [[join(v_path, im), label] for im in os.listdir(v_path)]
                test += images

train_csv = pd.DataFrame(train, columns=['path', 'labels']).sample(frac=1)
val_csv = pd.DataFrame(val, columns=['path', 'labels']).sample(frac=1)
test_csv = pd.DataFrame(test, columns=['path', 'labels']).sample(frac=1)

train_csv.to_csv(join(labels_path, 'train.csv'), index=False)
val_csv.to_csv(join(labels_path, 'val.csv'), index=False)
test_csv.to_csv(join(labels_path, 'test.csv'), index=False)
