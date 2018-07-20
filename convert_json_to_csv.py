import json
import os
import pdb
from operator import itemgetter
from matplotlib import pylab as pl
import cv2
import numpy as np
import glob
import PIL.Image as pil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

ROOT_DIR = '/home/lei/data/traffic_sign/'
OUTPUT_DIR='/home/lei/data/traffic_sign/output/'
INPUT_JASON_FILE = 'auto_gen_all_labels.json'
OUTPUT_CSV_FILE = 'traffic_sign_labels.csv'
OUTPUT_IMAGE_DIR = '/home/lei/data/traffic_sign/train_filter/'

#change the name for each traffic sign
CLASS_TO_USE = ["SL_10", "SL_20", "SL_25", "SL_30", "SL_35", "SL_40", "SL_45", "YD", "Stop", "NPS", "PCS", "SA"]

one_decimal = "{0:0.1f}"
final_train_json = []
final_test_json = []
final_val_json = []

sorted_annos = {}
sorted_annos["imgs"] = {}

all_image_data = {}

def get_file_name(full_name):
    head, tail = os.path.split(full_name)
    return tail

def verify_boundingbox_on_image(image_name, objects):
    # image_name
    # objects = (object_label,object_xmin,object_xmax,object_ymin,object_ymax)
    # pdb.set_trace()
    im = np.array(pil.open(image_name), dtype=np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(im)
    for bbox in objects:
        label = bbox[0]
        xmin = bbox[1]
        ymin = bbox[3]
        width = bbox[2] - bbox[1]
        height = bbox[4] - bbox[3]

        rect = patches.Rectangle((xmin,ymin),width, height,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin,ymin, label,bbox=dict(facecolor='red', alpha=0.5))
    #plt.show()
    output_image_name = ROOT_DIR + 'verify_csv/' + get_file_name(image_name) 
    fig.savefig(output_image_name)
    plt.close(fig)

def convert():
    global sorted_annos

    input_file = ROOT_DIR + INPUT_JASON_FILE
    csv_list = []
    #pdb.set_trace()

    srcData = json.loads(open(input_file).read())
    srcImageTotalCount = len(srcData['imgs'])
    #print srcImageTotalCount

    imagecount = 0
    objectcount = 0
    imageErrorCnt = 0
    object_counter_per_class = []

    for eachImage in srcData["imgs"]:
        #print eachImage
        #print srcData["imgs"][eachImage]

        #pdb.set_trace()
        if len(srcData["imgs"][eachImage]['objects']) > 0:
            #pdb.set_trace()
            #print srcData["imgs"][eachImage]['path']
            #print srcData["imgs"][eachImage]['objects']
            im_full_name_original = ROOT_DIR + (srcData["imgs"][eachImage]['path']).strip('./')
            image_name = get_file_name(im_full_name_original)
            if (os.path.isfile(im_full_name_original))==False:
                imageErrorCnt = imageErrorCnt + 1
                print '------------------------------ERROR-----------------------------'
                print srcData["imgs"][eachImage]['path']
                print srcData["imgs"][eachImage]['objects']
                continue
            imagecount = imagecount + 1
            im_full_name = OUTPUT_IMAGE_DIR + image_name
            shutil.copyfile(im_full_name_original, im_full_name)
            image = pil.open(im_full_name)
            width = int(image.size[0])
            height = int(image.size[1])

            #verification
            ver_objects = []

            for eachObject in srcData["imgs"][eachImage]['objects']:
                for target_label in CLASS_TO_USE:
                    if eachObject['category'] == target_label:
                        #pdb.set_trace()
                        #print '-----------Found target sign: ' + target_label
                        #print eachObject['bbox']
                        objectcount = objectcount + 1

                        object_im_path = image_name
                        object_im_width = width
                        object_im_height = height
                        object_label = target_label
                        object_xmin = eachObject['bbox']['xmin']
                        object_xmax = eachObject['bbox']['xmax']
                        object_ymin = eachObject['bbox']['ymin']
                        object_ymax = eachObject['bbox']['ymax']
                        item_value = (object_im_path, object_im_width, object_im_height, object_label, object_xmin, object_xmax, object_ymin, object_ymax)
                        csv_list.append(item_value)
                        # verfication lables
                        ver_object = (object_label,object_xmin,object_xmax,object_ymin,object_ymax)
                        ver_objects.append(ver_object)
                        
                    else:
                        continue
            # verfication lables
            #pdb.set_trace()
            # verify_boundingbox_on_image(im_full_name, ver_objects)
    print srcImageTotalCount
    print imagecount
    print objectcount
    print imageErrorCnt

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    csv_df = pd.DataFrame(csv_list, columns=column_name)
    #pdb.set_trace()
    print csv_df.count()
    print csv_df['class'].value_counts()
    return csv_df

def main():
    csv_df = convert()
    csv_df.to_csv(OUTPUT_CSV_FILE, index=None)


main()