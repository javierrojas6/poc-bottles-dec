import glob2
import os
import numpy as np
import pandas as pd

def load_data_frame(url, filter='**/*.jpg'):
    """
    It takes a URL and a filter, and returns a dataframe with the file, category, and state of each
    image
    
    :param url: the path to the folder containing the images
    :param filter: The filter to use when searching for images, defaults to **/*.jpg (optional)
    :return: A dataframe with the columns file, category, and state.
    """
    files = np.array(glob2.glob(os.path.join(url, filter), recursive=True))
    
    rows = []
    for img in files:
        img = img.replace('\\', '/')
        url_parts = img.split('/')
        rows += [[img, url_parts[-3], 1 if url_parts[-2] == 'open' else 0]]
    
    df = pd.DataFrame(data=np.array(rows), columns=['file', 'category', 'state'])
    
    df['state'] = df['state'].astype(np.int64) # this is for training purposes, GPU
    df['file'] = df['file'].astype(np.str_)
    df['category'] = df['category'].astype(np.str_)

    return df
