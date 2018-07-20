PIPELINE_CONFIG_PATH=/home/lei/projects/01_dl/tensorflow/models/research/object_detection/samples/configs/ssd_mobilenet_v1_kitti.config
MODEL_DIR=/home/lei/data/kitti/ssd_train
export TF_CPP_MIN_LOG_LEVEL=2.
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --train_dir=${MODEL_DIR}
