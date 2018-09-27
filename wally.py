train_tfrecord_path = '/mnt/Find_Wally/Wally_ver3/train.tfrecord'
test_tfrecord_path = '/mnt/Find_Wally/Wally_ver3/test.tfrecord'
val_tfrecord_path = '/mnt/Find_Wally/Wally_ver3//val.tfrecord'


"""
Usage :
python main.py --batch_size=60 --datatype='wally' --model_name='vgg_13' --BN --l2_weight_decay=0.001 --logit_type='gap' --num_epoch=100 --cropped_size=64 --opt='adam' --init_lr=0.0001 --lr_decay_step=10

"""




