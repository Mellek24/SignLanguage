from torch import nn
from torch import optim
import sys
import os
from pathlib import Path
from problem import WLSLDataset
import cv2
from PIL import Image
sys.path.append(str(Path(os.path.dirname(__file__)).parent)) # Dirty but it works
import torch


from sklearn.base import BaseEstimator

problem_title = 'Sign Language Classification'

device = "cuda:0" if torch.cuda.is_available() else "cpu"


import torch
import torch.nn as nn

class VideoClassifier(nn.Module):
    def __init__(self):
        super(VideoClassifier, self).__init__()
        
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(32*112*112, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2000),
        )
        
    def forward(self, x):
        # Reshape the input tensor to (batch_size*num_frames, 3, 224, 224)
        batch_size, num_frames, C, H, W = x.size()
        x = x.reshape(-1, C, H, W)
        
        x = self.conv_layers(x)
        x = x.view(batch_size, num_frames, -1)
        x = x.mean(dim=1)  # average over frames
        x = self.fc_layers(x)
        return x


#
class Net(nn.Module):
    def __init__(self, nb_classes, hidden_size = 128):
        super(Net, self).__init__()
        self.net = nn.Linear(100, hidden_size)
        self.act = nn.ReLU()
        self.output = nn.Linear(hidden_size, nb_classes)
       
    def forward(self, x):
        x = x.view(-1, 3, 100, 224*224)
        x = torch.mean(x,[1,3])
        y = self.act(self.net(x))
        return(self.output(y))


class Classifier(BaseEstimator):

    def fit(self, X, y, nb_epochs = 1):
        self.dataset = WLSLDataset(X, y, max_frames=34)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        #self.model = Net(nb_classes = self.dataset.nb_classes)
        # use the video classifier
        self.model = VideoClassifier()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
  
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)

            loss = criterion(outputs, labels)#.to(torch.float32))
            loss.backward()
            optimizer.step()
            break

        return self

    def predict_proba(self, X):
        videos_tensor = torch.zeros((len(X), self.dataset.max_frames,3, 224, 224))
        for j, path in enumerate(X) :
            cap = cv2.VideoCapture(path)
            
            # Loop through the frames
            i = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False or i>=self.dataset.max_frames:
                    break  
                # convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_transformed = self.dataset.transform(Image.fromarray(frame))
                videos_tensor[j, i] = frame_transformed
                i += 1
            # Release the video capture object
            cap.release()
        return self.model(videos_tensor)
        


    def predict(self, X):
        probas = self.predict_proba(X)
        most_likely_outputs = torch.argmax(probas, axis = 1)
        #predictions = torch.zeros_like(probas)
        #for i, most_likely_output in enumerate(most_likely_outputs) :
        #    predictions[i,most_likely_output] = 1
        return most_likely_outputs