import os
import sys
import argparse
import numpy as np
import torchvision.transforms as T
import warnings

from sklearn.model_selection import train_test_split
import sklearn

sys.path.append(os.path.join(os.getcwd(), 'src'))
import bottle


parser = argparse.ArgumentParser()

parser.add_argument('dataset_path', type = str, help = 'data/dataset' )
parser.add_argument('-d', '--device', dest='device', type = str, default = 'device to compute the process' )
parser.add_argument('-f', '--file_filter', dest='file_filter', type = str, default = '**/*.jpg', help='filter of files to search by default will use **/*.jpg all JPG files will be included' )
parser.add_argument('-m', '--model', dest='model', type = str, choices=['bottle-detection', 'bottle-cap'], help='the model to train' )
parser.add_argument('-b', '--batch', dest='batch', type = int, default=32, help='use powers of 2. examples: 2, 4, 8, 16, etc...' )
parser.add_argument('-n', '--name', dest='file_name', type = str, default='bottle-net.pth', help='the file name to store trained model' )

parser.add_argument( '-debug', type = bool, default = True, help='if we display console process info' )

args = parser.parse_args()

warnings.filterwarnings('ignore')

print()
df = bottle.dataframe.load_data_frame(args.dataset_path, filter=args.file_filter)
df.info()
print()

device = bottle.utils.get_device()

if args.model == 'bottle-detection':
    model = bottle.model.build_bottle_detection_model(device)
elif args.model == 'bottle-cap':
    model = bottle.model.build_bottle_cap_state_detection_model(device, False)
    
# split dataset
x_train, x_test, y_train, y_test = train_test_split(df['file'], df['state'], train_size=.7, test_size=.3)

# restart implicit indices
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

transform = T.Compose([
    T.Resize((100, 200)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# creates the loaders train and test
train_loader = bottle.data.get_data_loader(x_train, y_train, transform, args.batch)
test_loader = bottle.data.get_data_loader(x_test, y_test, transform, args.batch)
    
model_wrapper = bottle.model.ModelWrapper(model)

model_wrapper.load(args.file_name)

y_true, y_pred = model_wrapper.evaluate(test_dataset=test_loader,
                       device=device,
                       debug_mode=True)

metrics = bottle.model.evaluate_classification(y_true, y_pred, labels=[0 ,1])

print('metrics', metrics)