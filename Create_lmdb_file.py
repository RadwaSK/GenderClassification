from scipy.io import loadmat
import numpy as np
import pandas as pd


def create_lmdb_file(file_path_imgs):
    imdb_data = loadmat(file_path_imgs)['imdb']

    imdb_actor = imdb_data[0][0][1][0]
    imdb_gender = imdb_data[0][0][3][0]

    imdb_img_path = imdb_data[0][0][2][0]
    imdb_face_score1 = imdb_data[0][0][6][0]
    imdb_face_score2 = imdb_data[0][0][7][0]

    imdb_labels = []

    for n in range(len(imdb_gender)):
        if imdb_gender[n] == 1:
            imdb_labels.append('male')
        else:
            imdb_labels.append('female')

    imdb_ALL_paths = []

    for path in imdb_img_path:
        imdb_ALL_paths.append('/kaggle/input/imdb-wiki-faces-dataset/imdb_crop/' + path[0])


    df_imdb = np.vstack((imdb_ALL_paths, imdb_labels, imdb_face_score1, imdb_face_score2)).T

    df_imdb = pd.DataFrame(df_imdb)
    df_imdb.columns = ['path', 'labels', 'face_score1', 'face_score2']

    face_gender = df_imdb[df_imdb['face_score1'] != '-inf']
    face_gender = face_gender[face_gender['face_score2'] == 'nan']
    face_gender = face_gender.drop(['face_score1', 'face_score2'], axis=1)

    face_gender.to_csv('face_gender.csv', index=False)