当你打开说明时是不能操作的！！！
当你打开说明时是不能操作的！！！
当你打开说明时是不能操作的！！！


默认参数：
source_train = 'corpora/train.vi'
target_train = 'corpora/train.en'
source_test = 'corpora/tst2013.vi'
target_test = 'corpora/tst2013.en'

batch_size = 32 # alias = N
lr = 0.0001 # learning rate
logdir = 'logdir' # log directory
    
maxlen = 10 # 每句话长度
min_cnt = 20 # 出现次数过少会显示 <UNK>.
num_blocks = 6 # number of encoder/decoder blocks
num_epochs = 20 #遍历次数

在这里maxlen最小要保障在10以上，同时epoch最好也在10-20之间
更改参数后先运行预处理
然后训练
等待即可得到结果
然后点击评分即可看到结果与翻译

新添加1.0：
增加说明
句子长度修改
可修改遍历次数
可修改min_cnt

2.0：
增加预处理，训练功能
增加默认值提示增加

3.0:
增加结果评分按钮

4.0：
结果评分会在进程中显示
可在评价中看到翻译过程了

5.0：
可以选择结果查看了
结果文件名称：（1）model_epoch_20_gs_17617
			  （2）model_epoch_20_gs_8240

6.0：
可以查看预处理结果了

7.0：
可以查看参数了

8.0：
可以自己添加测试数据了
这个数据需要足够数据支撑，所以建立在数据基础上，需要自己分词。
