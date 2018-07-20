PIPELINE_CONFIG_PATH=/home/lei/projects/01_dl/tensorflow/models/research/object_detection/samples/configs/ssd_mobilenet_v1_kitti.config
MODEL_DIR=/home/lei/data/kitti/outputslim.learning.train
NUM_TRAIN_STEPS=10000
NUM_EVAL_STEPS=200
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr