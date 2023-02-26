# Dependencies

import os
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
import rampwf as rw
from rampwf.utils.importing import import_module_from_source
import matplotlib.pyplot as plt
import cv2
import numpy as np

# --------------------------------------------------
#
# Challenge title

problem_title = "Bovine embryos survival prediction"


# --------------------------------------------------
#
# Select Prediction type

_prediction_label_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

pred_times = [27, 32, 37, 40, 44, 48, 53, 58, 63, 94]
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names * len(pred_times)
)

# --------------------------------------------------
#
# Select Workflow

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