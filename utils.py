
# from http.client import _DataType
from attr import fields_dict
from cv2 import LINE_AA
from detectron2 import data
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import boxes
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import VisImage, Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1],
                       metadata=dataset_custom_metadata,
                       scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15, 20))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, val_dataset_name, num_classes, device,
                  output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = train_dataset_name
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg


def on_image(image_path, predictor, output_dir):
    im = cv2.imread(image_path)
    print('Processing the input image... ' )
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    image = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    dataset_dicts = []
    dataset_dict = {}

   
    # v1 = outputs
    # for _, anno in v1.items():
    #     dataset_dicts.append(anno)
    # result = ''.join(map(str, dataset_dicts))
    # result1 = result.split()
    
    v2 = outputs["instances"].pred_boxes.tensor.numpy()
    # v2 = outputs['instances'][outputs['instances'].pred_classes==1].pred_boxes.tensor.cpu().numpy()
    x1 = v2[0][0]
    y1 = v2[0][1]
    w1 = v2[0][2]
    h1 = v2[0][3]
    x1 = x1 // 2  # 458
    y1 = y1 // 2  # 310
    w1 = w1 // 2  # 668
    h1 = h1 // 2  # 666
    w = w1 // 2
    h = h1 // 2
    w2 = (x1 + w1) // 2
    h2 = (y1 + h1) // 2
    w4 = (h2 + y1) // 2
    h4 = (h1 + h2) // 2



    y2 = y1 * 2  # 620
    h3 = h // 2  # 330
    w3 = w2 // 2  # 550
    # w4 = (w1 - x1) // 2
   

    # h4 = h4 + 40
    
    print(x1, y1, w1, h1, w, h, w2, h2)
    print(w2, w4, w2, h4)
    
    font_face = cv2.FONT_HERSHEY_DUPLEX
    
    if len(outputs['instances']) >= 1:
        text = ("Number of WELDED cases detected: " + str(len(outputs['instances'])))
    else:
        text = " WELDED cases have not been detected! "
    org = (30, 40)
    fontScale = 1
    color = (250, 250, 250)
    thickness = 1

    # s = input('x값 입력:')
    # intx = int(s)
    # print(intx)

    
    image = image.get_image()
    # image = np.array(image.get_image())
    image = cv2.line(image, (int(w2), int(w4)), (int(w2), int(h4)), (0, 0, 255), thickness=3, lineType=LINE_AA)




    # imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY_INV)
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contr = contours[0]
    # # 감싸는 사각형 표시(검정색)
    # x,y,w,h = cv2.boundingRect(contr)
    # image = cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,0), 3)
    # # 최소한의 사각형 표시(초록색)
    # rect = cv2.minAreaRect(contr)
    # box = cv2.boxPoints(rect)   # 중심점과 각도를 4개의 꼭지점 좌표로 변환
    # box = np.int0(box)          # 정수로 변환
    # img = cv2.drawContours(image, [box], -1, (0,0,255), 3)

        
    # ret, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # print(np.where(image[1][0][0]))
    
    # img = cv2.putText(img, text, org, font_face, fontScale, color, thickness, cv2.LINE_AA)

    # img = cv2.line(img, (150,50), (200, 100), (0,255,0), thickness=1, lineType=LINE_AA)

    # poly1 = np.array([[50, 50], [100, 100]])
    # img = cv2.polylines(image, [poly1], True, (0, 255, 0))
    
    
    filename = './output/outputImage.jpg'
    # cv2.imwrite(filename, img)
    # print('Successfully saved the Proceeded output image')

    plt.figure(figsize=(15, 20))
    plt.imshow(image)
    plt.show()

def draw_line(predictor):
    pass



def on_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(
            'Error in opening the file...'
        )
        return
    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions['instances'].to('cpu'))

        cv2.imshow('Result', output.get_image()[:, :, ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        (success, image) = cap.read()
