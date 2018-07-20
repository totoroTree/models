"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python convert_csv_to_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python convert_csv_to_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record

  # Reference
  https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from collections import namedtuple, OrderedDict

VALIDATION_IMAGE_DADASET_SIZE = 300

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_path', '', 'Path to input training images')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '', 'Path to all the labels list')
FLAGS = flags.FLAGS

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

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map_dict):
    #pdb.set_trace()
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        if row['class'] in label_map_dict:
            classes.append(label_map_dict[row['class']])
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
        else:
            continue
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    #pdb.set_trace()
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    val_counter = VALIDATION_IMAGE_DADASET_SIZE
    is_validation_img =  False
    val_count = 0
    train_count = 0
    train_writer = tf.python_io.TFRecordWriter('%s_train.tfrecord'%
                                                FLAGS.output_path)
    val_writer = tf.python_io.TFRecordWriter('%s_val.tfrecord'%
                                            FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    for group in grouped:
        #pdb.set_trace()
        print('filename:{}'.format(group.filename))
        val_counter -= 1
        if val_counter == 0:
            is_validation_img = True
          
        tf_example = create_tf_example(group, FLAGS.image_path, label_map_dict)
        if is_validation_img:
            val_writer.write(tf_example.SerializeToString())
            val_count += 1
            is_validation_img = False
            val_counter = VALIDATION_IMAGE_DADASET_SIZE
        else:
            train_writer.write(tf_example.SerializeToString())
            train_count += 1

    train_writer.close()
    val_writer.close()

    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
    print('val_count: {}'.format(val_count))
    print('train_count: {}'.format(train_count))

if __name__ == '__main__':
    tf.app.run()
