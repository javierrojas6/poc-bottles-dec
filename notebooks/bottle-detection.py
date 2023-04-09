# %%
import seaborn as sns
sns.set_style('darkgrid')
import os
import sys
import matplotlib.pyplot as plt
import torch

from datetime import datetime
# %%
name = 'tailored'
date_number = datetime.now().strftime('%Y%m%d%H%M%S')
device = 'cpu'
ext = 'pth'

print('name', '.'.join([f'{name}-{date_number}',device,ext]))
#%% 
COCOLabels(4).name
# %%
# Defining the path to the folders where the images are stored.
DATASET_PATH = '../data/dataset'

# %%
sys.path.append(os.path.join(os.getcwd(), '../src'))
# %%
import bottle

#%%
# Loading the dataframe from the dataset folder.
df = bottle.dataframe.load_data_frame(DATASET_PATH, filter='**/*.jpg')
df.info()

# %%# %%
device = bottle.utils.get_device()
# %%
model = bottle.model.build_bottle_detection_model(device)
model.eval()
# %%
pred = bottle.image.get_prediction(model, img_path=df['file'][100],class_names=bottle.model.COCO_CLASS_NAMES)

# %%
filtered_data = bottle.image.filter_prediction(pred,class_filter=['bottle'])

# %% 
img = bottle.image.plot_segments(img_path=df['file'][100], pred_data=filtered_data)

plt.figure(figsize=(20, 30))
plt.imshow(img)
plt.grid(False)
plt.axis(True)
plt.title(f'detected {img.shape}')

# %%
model2 = bottle.model.build_bottle_cap_state_detection_model(device)
#%%
model2.load_state_dict(torch.load('../pretrained/bottle-net.pth'))