"""
train_tfrecord_path = '/mnt/Find_Wally/Wally_ver3/tfrecords/train.tfrecord'
test_tfrecord_path = '/mnt/Find_Wally/Wally_ver3/tfrecords/test.tfrecord'
val_tfrecord_path = '/mnt/Find_Wally/Wally_ver3/tfrecords/val.tfrecord'
"""

train_tfrecord_path = '/mnt/Find_Wally/wally_raspCam_np/train_80x80.tfrecord'
test_tfrecord_path = '/mnt/Find_Wally/wally_raspCam_np/test_80x80.tfrecord'
val_tfrecord_path = '/mnt/Find_Wally/wally_raspCam_np/val_80x80.tfrecord'


"""
Usage :
python main.py --batch_size=60 --datatype='wally' --model_name='vgg_13' --BN --l2_weight_decay=0.001 --logit_type='gap' --num_epoch=100 --cropped_size=48 --opt='adam' --init_lr=0.0001 --lr_decay_step=10

"""




