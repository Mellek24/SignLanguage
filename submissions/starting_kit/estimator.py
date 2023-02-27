from torch import nn
from torch import optim
import sys
import os
from pathlib import Path
#from torchvision.models import resnet101
import torch.nn.functional as F
from problem import WLSLDataset
sys.path.append(str(Path(os.path.dirname(__file__)).parent)) # Dirty but it works

#from bop_scripts.preprocessing import remove_outliers
#from bop_scripts.nn_models import torchMLPClassifier_sklearn, torchMLP
#from bop_scripts.models import generate_model, fit_all_classifiers
import torch
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

problem_title = 'Sign Language Classification'

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# def torch_classifier_fn ():


#     return torch_sklearn_classifier
class Net(nn.Module):
    def __init__(self, nb_classes):
        super(Net, self).__init__()
        self.net = nn.Linear(100, 128)
        self.act = nn.ReLU()
        self.output = nn.Linear(128, nb_classes)
       
    def forward(self, x):
        x = x.view(-1, 3, 100, 224*224)
        x = torch.mean(x,[1,3])
        y = self.act(self.net(x))
        return(self.output(y))
 

class Classifier(BaseEstimator):

    def fit(self, X, y):
        dataset = WLSLDataset(X, y, max_frames=100)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = nn.MSELoss()
        self.model = Net(nb_classes = dataset.nb_classes)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

  
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)

            loss = criterion(outputs, labels.to(torch.float32))
            loss.backward()
            optimizer.step()

            if i > 5 :
                break

        return self

    def predict_proba(self, X):
        dataset =  WLSLDataset(X, []*len(X))
        return self.model(dataset)

    def predict(self, X):
        return self.predict_proba(X)