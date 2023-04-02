
# %%
import os
import seaborn as sns
import torchvision.transforms as T
import sys

# %%
DATASET_PATH = '../data/dataset'

# %%
sys.path.append(os.path.join(os.getcwd(), '../src'))
# %%
import bottle
# %%
%matplotlib inline
sns.set_style('darkgrid')

# %% load files list
df = bottle.dataset.load_data_frame(DATASET_PATH, filter='**/*.jpg')
df.info()
# %%
sns.set(style="whitegrid")
sns.catplot(legend=True, x="category", kind="count", palette="ch:.25", data=df, height=4, aspect=1)

sns.set(style="whitegrid")
sns.catplot(x="state", kind="count", palette="ch:.25", data=df[df['category'] == 'water'], height=4, aspect=1)

sns.set(style="whitegrid")
sns.catplot(x="state", kind="count", palette="ch:.25", data=df[df['category'] == 'beer'], height=4, aspect=1)

# %%
transform = T.Compose([
    T.Resize((100, 200)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset_loader = bottle.data.get_data_loader(df['file'], df['state'], transform)

# %%
dataiter = iter(dataset_loader)
images, labels = next(dataiter)

# %% print images
bottle.dataset.plot_image_gallery(images, title='Gallery', transform_color=False, figsize=(20, 20))