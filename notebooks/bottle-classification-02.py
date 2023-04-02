# %%
!pip install opencv-python
!pip install matplotlib
!pip install seaborn
!pip install torch
!pip install torchvision
!pip install glob2
!pip install pandas
!pip install scikit-learn

# %%
from torchvision import models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as T
import torchvision
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob2
import pandas as pd

# %%
PATH_FOLDER_IMAGES = '../data/prueba'
PATH_SEGMENTS = '../data/segments'
PATH_DATASET = '../data/dataset'

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
file_filter = '**/*.jpg'
list_images_files = np.array(glob2.glob(os.path.join(PATH_DATASET, file_filter), recursive=True))

print(f'cantidad de images: {len(list_images_files)}')
# %%  Build dataset in DF
rows = []
for img in list_images_files:
    img = img.replace('\\', '/')
    url_parts = img.split('/')
    rows += [[img, url_parts[-3], 1 if url_parts[-2] == 'open' else 0]]

# %%
df = pd.DataFrame(data=np.array(rows), columns=['file', 'category', 'state'])
df['state'] = df['state'].astype(np.int8)
df['file'] = df['file'].astype(np.str_)
df['category'] = df['category'].astype(np.str_)
df.head()

# %%
df.describe(include='all')

# %%
df.query('state != 1 and state != 0').describe()
# %%
df.info()

# %%
sns.set(style="whitegrid")
sns.catplot(legend=True, x="category", kind="count",
            palette="ch:.25", data=df, height=4, aspect=1)

# %%
sns.set(style="whitegrid")
sns.catplot(x="state", kind="count", palette="ch:.25",
            data=df[df['category'] == 'water'], height=4, aspect=1)
sns.set(style="whitegrid")
sns.catplot(x="state", kind="count", palette="ch:.25",
            data=df[df['category'] == 'beer'], height=4, aspect=1)

# %%
imageList = []
for row in df.iterrows():
    # imageList += [mpimg.imread(row[1]['file'])]
    img = np.array(Image.open(row[1]['file']))
    tensor = torch.from_numpy(img)
    imageList += [tensor]
    
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# %%
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net() 
#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%%
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')