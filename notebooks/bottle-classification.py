# %%
!pip install opencv-python
!pip install matplotlib
!pip install seaborn
!pip install torch
!pip install torchvision
!pip install glob2
!pip install pandas
!pip install scikit-learn
!pip install fastapi
!pip install "uvicorn[standard]"
!pip install python-multipart
!pip install pipreqs

# %%
# Importing the necessary libraries
import os
import sys

import torch
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
# %%
# Defining the path to the folders where the images are stored.
PATH_FOLDER_IMAGES = '../data/prueba'
PATH_SEGMENTS = '../data/segments'
DATASET_PATH = '../data/dataset'

# %%
sys.path.append(os.path.join(os.getcwd(), '../src'))
# %%
import bottle

#%%
# Loading the dataframe from the dataset folder.
df = bottle.dataframe.load_data_frame(DATASET_PATH, filter='**/*.jpg')
df.info()

x_train, x_test, y_train, y_test = train_test_split(df['file'], df['state'], train_size=.7, test_size=.3)

print('x_train:', len(x_train))
print('x_test:', len(x_test))
print('y_train:', len(y_train))
print('y_test:', len(y_test))

# %%
num_classes = df['state'].unique().shape[0]
learning_rate = 1e-5
weight_decay = 1e-3


transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_file_list = df['file']
train_labels = df['state']


model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features


# model = ConvNet()
# num_ftrs = 1600


model.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
