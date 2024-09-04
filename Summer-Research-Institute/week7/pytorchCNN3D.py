import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

NUMCLASSES = 5
TRAINPATH = 'UCF101Dataset/train'
TESTPATH = 'UCF101Dataset/test'
TRAINLABELSPATH = 'UCF101Dataset/train.csv'
TESTLABELSPATH = 'UCF101Dataset/test.csv'

train_df = pd.read_csv(TRAINLABELSPATH)
test_df = pd.read_csv(TESTLABELSPATH)


class VideoDataset(Dataset):
    def __init__(self, paths,labels):
        self.paths = paths
        self.labels = labels
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self,x):
        path = self.paths[x]
        label = self.labels[x]
        
        frames = torch.tensor(load_frames(path))
        frames = frames.float()

        return frames, label
    
"""
    Loads and frames from the provided file path
    
    returns: numpy array of generated frames
"""
def load_frames(path, numFrames=16): 

    cap = cv2.VideoCapture(path) # opening video
    frames = []

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameInterval = max(totalFrames // numFrames, 1)
    for i in range(numFrames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*frameInterval) # set frame position
        ret, frame = cap.read() # read frame at position

        if not ret: # exit loop if at end of video
            break

        frame = cv2.resize(frame, (112,112))
        frames.append(frame)

    while len(frames) < numFrames:
        frames.append(np.zeros((112,112,3), np.uint8)) # fill in blank frames with zeroes 

    return np.array(frames)

class VideoClassifier(nn.Module):
    def __init__(self,numClasses):
        super(VideoClassifier, self).__init__()
        self.conv3D1 = nn.Conv3d(3,64, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv3D2 = nn.Conv3d(64,128, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv3D3 = nn.Conv3d(128,256, kernel_size=(3,3,3), padding=(1,1,1))

        self.FC1 = nn.Linear(256*4*4*4, 128)
        self.FC2 = nn.Linear(128, numClasses)
    
    def forward(self, x):
        x = F.relu(F.max_pool3d(self.conv3D1(x), kernel_size=(2,2,2)))
        x = F.relu(F.max_pool3d(self.conv3D2(x), kernel_size=(2,2,2)))
        x = F.relu(F.max_pool3d(self.conv3D3(x), kernel_size=(2,2,2)))
        
        x = x.view(-1, 256*4*4*4)

        x = F.relu(self.FC1(x))
        x = self.FC2(x)
        return x
    
train_video_paths = train_df['video_name'].values
train_labels = train_df['tag'].values

test_video_paths = test_df['video_name'].values
test_labels = test_df['tag'].values

traindataset = VideoDataset(train_video_paths, train_labels)
traindataloader = DataLoader(traindataset, batch_size=32, shuffle=True)

testdataset = VideoDataset(test_video_paths, test_labels)
testdataloader = DataLoader(testdataset, batch_size=32, shuffle=False)

model = VideoClassifier(NUMCLASSES)
optimizer = optim.Adam(model.parameters(), lr=0.001)
lossFunction = nn.CrossEntropyLoss()

# MODEL TRAINING
for epoch in range(10):
    for batch in traindataloader:
        frames, labels = batch
        inputs = frames.permute(0,4,1,2,3) # convert dimensions
        outputs = model(inputs)
        loss = lossFunction(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# MODEL TESTING
model.eval()
testLoss,correct = 0
with torch.no_grad():
    for batch in testdataloader:
        frames, labels = batch
        inputs = frames.permute(0,4,1,2,3)
        outputs = model(inputs)
        loss = lossFunction(outputs, labels)
        testLoss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()


accuracy = correct / len(testdataset)
print(f'Test Loss: {testLoss/len(testdataloader)}')
print(f'Accuracy: {accuracy:.2f}')
