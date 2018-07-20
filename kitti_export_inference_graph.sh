PIPELINE_CONFIG_PATH=/home/lei/projects/01_dl/tensorflow/models/research/object_detection/samples/configs/ssd_mobilenet_v1_kitti.config
MODEL_DIR=/home/lei/data/kitti/ssd_train/graph_server
python object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path ${PIPELINE_CONFIG_PATH} \
--trained_checkpoint_prefix /home/lei/data/kitti/ssd_train/graph_server/model.ckpt \
--output_directory ${MODEL_DIR}
