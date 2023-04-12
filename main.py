
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import re
import os
import sys
import warnings
import shutil
import uuid

from PIL import Image
from pyparsing import List

sys.path.append(os.path.join(os.getcwd(), 'src'))
import bottle

app = FastAPI()

# sets parameters
bottle_cap_model_file = './pretrained/bottle-cap-20230411101512.cpu.pth'
threshold = 0.7
device = 'cpu'
public_folder = 'public'

warnings.filterwarnings('ignore')

# return types
class BottleItem(BaseModel):
    label: str
    score: float
    box: List
    state: str

class AnalysisRespose(BaseModel):
    bottles: list[BottleItem]
    image: str

# loads AI models
bottle_detection_model = bottle.model.build_bottle_detection_model(device)
##  switch to evaluation mode
bottle_detection_model.eval()
bottle_cap_state_model = bottle.model.build_bottle_cap_state_detection_model(device, False)

bottle_cap_state_model_wrapper = bottle.model.ModelWrapper(bottle_cap_state_model)
bottle_cap_state_model_wrapper.load(bottle_cap_model_file)

# app settings
## public folder
app.mount(f"/{public_folder}", StaticFiles(directory=public_folder), name=public_folder)

@app.post("/analyze/picture")
async def analyze_picture(request: Request, file: UploadFile):
    # File validation
    if file.content_type == None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid image file")

    is_image = re.search(r"image/(jpg|jpeg|png)", file.content_type, re.RegexFlag.IGNORECASE)
    if is_image == None :
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid image file")

    # save locally the file
    _, file_extension = os.path.splitext(file.filename)
    new_file_name = f'{uuid.uuid4()}{file_extension}'
    store_filepath = '/'.join([public_folder, new_file_name])

    with open(store_filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # let's analyze the file
    pred = bottle.image.get_prediction(bottle_detection_model, img_path=store_filepath, class_names=bottle.model.COCO_CLASS_NAMES)
    masks, boxes, classes, scores = bottle.image.filter_prediction(pred, threshold=threshold, class_filter=['bottle'])

    bottles_detected = len(masks)
    bottles_image = []

    # examine if there are one or many bottles in the same footage
    if bottles_detected == 0: raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Nothing detected")
    elif bottles_detected == 1: bottles_image = [Image.open(store_filepath)]
    else: bottles_image = bottle.image.split_images_by_mask(store_filepath, masks, boxes)

    predicted_cap_state = []
    for i in range(len(bottles_image)):
        predicted_cap_state += [bottle_cap_state_model_wrapper.predict(bottles_image[i], labels = ('opened', 'closed'))]

    new_classes = []
    for i in range(len(classes)):
        new_classes += [f'{predicted_cap_state[i]} {classes[i]}']

    # let's store a new image with debug info plotted
    img = bottle.image.plot_segments(img_path=store_filepath,
                                    pred_data=[masks, boxes, new_classes, scores],
                                    text_thickness=2,
                                    draw_score=False,
                                    draw_mask=False,
                                    mask_alpha=.3,
                                    text_size=1)
    filename, file_extension = os.path.splitext(new_file_name)
    debug_filename = f'{filename}-debug{file_extension}'
    bottle.image.save_image(img, '/'.join([public_folder, debug_filename]))

    bottles = []
    for i in range(len(classes)):
        bottles += [BottleItem(label=classes[i],
                        score=scores[i],
                        box=boxes[i].tolist(),
                        state=predicted_cap_state[i])]

    # build public url for debug image
    protocol = request.scope['type']
    host, port = request.scope['server']
    host_url = f'{protocol}://' + (':'.join([host, str(port)]) if port != 80 else host)
    image_public_url = '/'.join([host_url, public_folder, debug_filename])
    response = AnalysisRespose(bottles=bottles, image=image_public_url)
    
    return JSONResponse(content=jsonable_encoder(response))
