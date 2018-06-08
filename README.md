### 2018 . 5. 24
+ resnet v1 fixed
+ Learning rate Scheduler
+ L2 Loss
+ model save

function to add
###
+ Augmentation Scheduler
+ Model Restore
+ Transfer Model
+ Fine Tuning
+ L1 Loss
+ 왜 utils.plot_images 에서 255 로 나누지 않으면 보이지 않는거지? -->uint8로 바꾸어야 한다.
+ imgaug 을 사용할때도 uint8로 바꾸어야 한다.


###
Opinion




### should to fix
+ augmentation 을 input pipe line 에 넣어서 (py_func 사용 ) 한번에 처리하는 코드
+ validation 의 generation 확인하는 법



######
def __init__(self , recorder ):
        print '####################################################'
        print '#                   Tester                         #'
        print '####################################################'
        if recorder == None:
            self.recorder = recorder
        self.val_acc=0
        self.val_loss=0
        self.max_acc=0
        self.min_loss=10000000



####
    ## 구 버전과 통합하기 위해 w=None 을 추가했다. ##
    def get_class_map(self,name, x, cam_ind, im_width , w=None):
        out_ch = int(x.get_shape()[-1])
        conv_resize = tf.image.resize_bilinear(x, [im_width, im_width])
        if w is None:
            with tf.variable_scope(name, reuse=True) as scope:
                label_w = tf.gather(tf.transpose(tf.get_variable('w')), cam_ind)
                label_w = tf.reshape(label_w, [-1, out_ch, 1])
        else:
            label_w = tf.gather(tf.transpose(w), cam_ind)
            label_w = tf.reshape(label_w, [-1, out_ch, 1])

        conv_resize = tf.reshape(conv_resize, [-1, im_width * im_width, out_ch])
        classmap = tf.matmul(conv_resize, label_w, name='classmap')
        classmap = tf.reshape(classmap, [-1, im_width, im_width], name='classmap_reshape')
        return classmap

모델을 복원시키고 get_class_map을 통해서 weight 을 복원시킬려 하면 에러가 난다.
label_w = tf.gather(tf.transpose(tf.get_variable('w')), cam_ind)
이 부분에서 에러가 난다. scope 가 final 이고 weight 이름이 w 인데 못찾는 것이다

그 이유가 짐작컨데 get_class_map을 실행시킨것도 함수영역이라 그런게 아닐까 싶다.



#### CAM 을 얻으려면 bias 을 빼야 하나 마지막 부분에서 ?
#### CAM 을 얻으려면 한장씩 해야 한다  , 복수장씩 할려면
#### CAM 에서 바로 큰 activation map 을 얻을수 있다
#### cam_ind 는 필요없다

#### Tester에서 eval 을 추가하고 block box 을 추가했다 , Tester 하수에 None 을 넣은게 잘한걸까 , 어떻게 짜야 더 잘 짰다고 할수 있을까


####
추가해야 할점
sensitivity 와 specifiy 을 보여줘야 한다 . 평가 모델에서


####
모든 데이터셋은 Train , Test ,Val 이렇게 나누는 걸로 한다.


### 비정상 accuracy 와 정상 accuracy 가 어떻게 ?
라벨별 accuracy 을 보여주어야 한다 .

#### 지금 255 로 나누고 있는데 그게 아니라
이미지 전체의 평균과 분산을 빼는 방향으로 Normalization 을 해야 한다 .


2018 6.4 + Tester, validate 에 label 별 accuracy 추가 기능
2018 6.4 + show_acc_by_label : 라벨별로 accuracy 넣을수 있는거 추가..



2018.6.4 np.where 때문에 에러가 났었다.
간단하게 cls=[cls == cls_ind]로 바꿨다. np.nonzeros , np.where 등의 활용에 대해 연구하자


2018.6.5 np.where 때문에 에러가 났었다.
Regression 추가를 위해 loss function 을 추가한다.
loss function 을 선택할수 있는 option 을 각 모델마다 넣는다.

2018.6.7
Regeression 모델에서는 정답이 int 가 float 이 필요히다.
앞으로 모든 라벨은 float 로 하는걸로 한다 .

수정 사항 :
Dataprovider.get_sample
Dataprovider.reconstruct_tfrecord_rawdata

label = int(example.features.feature['label'].float_list.value[0])
sample 데이터 셋에서 ...