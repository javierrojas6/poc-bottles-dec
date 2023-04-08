import numpy as np
import random
import cv2
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

COLORS = [[255,0,0],[0,255,0], [0,0,255]]

def colored_mask(mask):
  r = np.zeros_like(mask).astype(np.uint8)
  g = np.zeros_like(mask).astype(np.uint8)
  b = np.zeros_like(mask).astype(np.uint8)

  r[mask == 1], g[mask == 1], b[mask == 1] = COLORS[random.randrange(0, len(COLORS))]
  return np.stack([r, g, b], axis = 2)

def get_prediction(model, img_path, class_names, transforms = []):
  img = Image.open(img_path)

  if len(transforms) == 0:
    transforms = T.Compose([T.ToTensor()])

  img = transforms(img)
  pred = model([img])

  scores = np.array(list(pred[0]['scores'].detach().numpy()))
  masks = np.array((pred[0]['masks'] > 0).squeeze().detach().cpu().numpy())
  classes = np.array([class_names[i] for i in list(pred[0]['labels'].detach().numpy())])
  boxes = np.array([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())])

  return masks, boxes, classes, scores

def filter_prediction(prediction_data, threshold = .5, class_filter = []):
  masks, boxes, classes, scores = prediction_data

  condition = scores >= threshold

  if len(class_filter) > 0:
    condition &= np.in1d(classes, class_filter)

  masks = masks[condition]
  boxes = boxes[condition]
  classes = classes[condition]            
  scores = scores[condition] 

  return masks, boxes, classes, scores

def plot_segments(img_path,
                  pred_data,
                  draw_bounding_boxes = True,
                  draw_labels = True,
                  draw_score = True,
                  score_fix = 2,
                  rect_thickness = 1,
                  text_size = 1,
                  text_thickness = 2):
  
  masks, boxes, pred_cls, scores = pred_data
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  if draw_score:
    format_string = "{:.Xf}".replace('X', str(score_fix))

  for i in range(len(masks)):
    rgb_mask = colored_mask(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, .5, 0)

    if draw_bounding_boxes:
      cv2.rectangle(
          img,
          (int(boxes[i][0][0]), int(boxes[i][0][1])),
          (int(boxes[i][1][0]), int(boxes[i][1][1])),
          color=(0,255,0),
          thickness=rect_thickness
      )

    if draw_labels:
      label_params = [pred_cls[i]]

      if draw_score:
        label_params += [format_string.format(scores[i])]

      cv2.putText(
          img,
          " ".join(label_params),
          (int(boxes[i][0][0]) + 5, int(boxes[i][0][1]) + 20),
          cv2.FONT_HERSHEY_SIMPLEX,
          text_size,
          (0, 255, 0),
          thickness=text_thickness
      )

  return img