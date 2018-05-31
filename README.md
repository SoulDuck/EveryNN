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
