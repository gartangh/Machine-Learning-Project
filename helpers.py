import os
import numpy as np
import glob
import time
import csv
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import fisher_vector as fv
from sklearn.mixture import GaussianMixture

def getImgPaths(data_folder):
    """Returns filepaths of all files contained in the given folder as strings."""
    return np.sort(glob.glob(os.path.join(data_folder,'*')))

def toOneHot(indices, min_int, max_int):
    """Converts an enumerable of indices to a one-hot representations."""
    one_hot_length = max_int - min_int + 1
    eye = np.eye(one_hot_length)
    return eye[np.array(indices) - min_int]

def writePredictionsToCsv(predictions, out_path, label_strings):
    """Writes the predictions to a csv file.
    Assumes the predictions are ordered by test interval id."""
    with open(out_path, 'w') as outfile:
    # Initialise the writer
        csvwriter = csv.writer(
            outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Write the header
        row_to_write=['Id'] + [label for label in label_strings]
        csvwriter.writerow(row_to_write)
        # Write the rows using 18 digit precision
        for idx, prediction in enumerate(predictions):
          assert len(prediction) == len(label_strings)
          csvwriter.writerow(
              [str(idx+1)] +
              ["%.18f" % p for p in prediction])

def generateUniqueFilename(basename, file_ext):
    """Adds a timestamp to filenames for easier tracking of submissions, models, etc."""
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return basename + '_' + timestamp + '.' + file_ext

def createPath(path):
    """Creates a directory if it does not exist"""
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def createCodebook(features, codebook_size = 100):
    """Creates a visual BOW codebook"""
    train_features_to_encode=[]
    for image_features in features:
        train_features_to_encode.append(image_features.data)
    train_features_to_encode = np.concatenate(train_features_to_encode, axis=0)
    codebook = MiniBatchKMeans(n_clusters=codebook_size, batch_size=codebook_size*10)
    start = time.time()
    codebook.fit(train_features_to_encode)
    end = time.time()
    print('training took {} seconds'.format(end-start))
    return codebook

def encodeImage(features, codebook):
    """Encodes one image given a visual BOW codebook"""
    # find the minimal feature distance for all patches of the image
    visual_words = codebook.predict(features)
    word_occurrence = pd.DataFrame(visual_words,columns=['cnt'])['cnt'].value_counts()
    bovw_vector = np.zeros(codebook.n_clusters)
    for key in word_occurrence.keys():
        bovw_vector[key] = word_occurrence[key]
    bovw_feature = bovw_vector/np.linalg.norm(bovw_vector)
    return bovw_feature

################################ FV ################################
def createGaussianMixture(features, K_value = 5):
    """Creates a visual BOW codebook"""
    train_features_to_encode=[]
    for image_features in features:
       train_features_to_encode.append(image_features.data)
    train_features_to_encode = np.concatenate(train_features_to_encode, axis=0)
    mixture = GaussianMixture(n_components=K_value, covariance_type='diag')
    start = time.time()
    mixture.fit(train_features_to_encode)
    end = time.time()
    print('GMM training took {} seconds'.format(end-start))
    return mixture

def powerNormalization(mid_level_features, gamma):
    power_normed = np.sign(mid_level_features) * np.abs(mid_level_features) ** gamma
    return power_normed

def encodeImage_FV(features,mixture,gamma):
    visual_words = fv.fisher_vector(features,mixture)
    bovw_vector = powerNormalization(visual_words, gamma)
    bovw_feature = bovw_vector/np.linalg.norm(bovw_vector)
    return bovw_feature
