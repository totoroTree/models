python object_detection/dataset_tools/create_kitti_tf_record.py \
--data_dir=/home/lei/data/kitti \
--output_path=/home/lei/data/kitti/kitti.record \
--label_map_path=/home/lei/projects/01_dl/tensorflow/models/research/object_detection/data/kitti_label_map.pbtxt \
--classes_to_use='car,van,truck,pedestrian,person_sitting,cyclist,tram,misc,dontcare'
