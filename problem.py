import os
import copy
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, top_k_accuracy_score

import numpy as np
import pandas as pd
import numpy as np
import rampwf as rw

from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from rampwf.utils.importing import import_module_from_source
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


problem_title = 'Classification of word-level american sign language videos'

# A type (class) which will be used to create wrapper objects for y_pred
_prediction_label_names = list(range(2000))

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.Classifier()


class Accuracy(BaseScoreType):

    def __init__(self, name='accuracy', precision=5):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


class KTopAccuracy(BaseScoreType):

    def __init__(self, name='k_top_accuracy', precision=5, k=5):
        self.name = name
        self.precision = precision
        self.k = k

    # predictions are the probs for each class
    def __call__(self, y_true, y_pred):
        #sorted_indices = np.argsort(predictions, axis=1)[:, -self.k:]
        #correct = np.array([y_true[i] in sorted_indices[i] for i in range(len(y_true))])
        return top_k_accuracy_score(y_true, y_pred, k=self.k, normalize=True)


score_types = [
    Accuracy(name='accuracy', precision=5),
    KTopAccuracy(name='5_top_accuracy', precision=5, k=5),
    KTopAccuracy(name='10_top_accuracy', precision=5, k=10)
]


def get_cv(X, y):
    cv = StratifiedKFold(n_splits=2, random_state=42, shuffle=True) 
    #ShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    return cv.split(X, y)




def read_video(path):
    # Open the video file
    cap = cv2.VideoCapture(path)
    
    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create an empty NumPy array to hold the frames
    video = np.empty((frame_count, height, width, 3), np.dtype('uint8'))
    
    # Loop through the frames and add them to the array
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video[i] = frame
        i += 1
    
    # Release the video capture object
    cap.release()
    
    # Return the video as a NumPy array
    return video

def get_video_path(word, data):
    for gloss in data:
        if gloss['gloss'] == word:
            # check if the path exists and return it
            for i in range(len(gloss['instances'])):
                path = 'data/raw_videos/'+gloss['instances'][i]['video_id']+'.mp4'
                if os.path.exists(path):
                    return path
                else:
                    continue
    return None

def plot_video_frames(video):
    num_frames = video.shape[0]
    num_cols = min(10, num_frames)
    fig, axes = plt.subplots(1, num_cols, figsize=(200, 100))
    
    for i in range(num_cols):
        frame_idx = i*int(num_frames/num_cols)
        frame = video[frame_idx]
        axes[i].imshow(frame)
        axes[i].axis('off')
            
    plt.show()

def get_data(split, path):
    with open(path+'/data/WLASL_v0.3.json') as f:
        data = json.load(f)
    paths = []
    labels = []
    labels_dict = pd.read_pickle('./data/labels_dict.pkl')
    for d in data:
        for instance in d['instances']:
            if instance['split'] == split:
                path = 'data/raw_videos/'+instance['video_id']+'.mp4'
                if os.path.exists(path):
                    paths.append(path)
                    labels.append(labels_dict[d['gloss']])
                    
    return np.array(paths), np.array(labels)

def get_train_data(path='.'):
    return get_data('train', path)

def get_test_data(path='.'):
    paths, labels = get_data('test', path)
    paths_reduced = paths[:10]
    labels_reduced = labels[:10]
    return paths_reduced, labels_reduced


class WLSLDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, max_frames=34):
        self.paths = paths
        self.labels = np.array(labels)
        self.max_frames = max_frames
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        #self.le = LabelEncoder()
        #self.labels = self.le.fit_transform(self.labels)
        self.nb_classes = self.labels.max()

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        #
        
        video_tensor = torch.zeros((self.max_frames,3, 224, 224))
        
        cap = cv2.VideoCapture(path)
        
        # Loop through the frames
        i = 0
        starting_frame = 10
        while(cap.isOpened()):
            ret, frame = cap.read()
            i += 1
            if ret == False or i>=self.max_frames:
                break  
            # convert to RGB
            if i < starting_frame:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_transformed = self.transform(Image.fromarray(frame))
            video_tensor[i] = frame_transformed
            
        # Release the video capture object
        cap.release()
            
        return video_tensor, label

    def __len__(self):
        return len(self.paths)



