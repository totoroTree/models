export TF_CPP_MIN_LOG_LEVEL=2.
PIPELINE_CONFIG_PATH=/home/lei/projects/01_dl/tensorflow/models/research/object_detection/samples/configs/ssd_mobilenet_v1_kitti.config
CKPT_DIR=/home/lei/data/kitti/ssd_train/graph
EVAL_DIR=/home/lei/data/kitti/ssd_train/eval/

python object_detection/eval.py \
    --logtostderr \
    --checkpoint_dir=${CKPT_DIR} \
    --eval_dir=${EVAL_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH}