# -*- coding: utf-8 -*-
#/usr/bin/python2
from tkinter import *
import hashlib
import time
from hyperparams import Hyperparams as hp
from prepro import *
from train import *
from eval import *
from eval1 import *
import os

LOG_LINE_NUM = 0

class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name
        #默认值
        

    #设置窗口
    def set_init_window(self):
        self.init_window_name.title("可视化界面_括号中为默认值")           #窗口名
        self.init_window_name.geometry('640x320+10+10')                         #290 160为窗口大小，+10 +10 定义窗口弹出时的默认展示位置
        #self.init_window_name.geometry('1068x681+10+10')
        #self.init_window_name["bg"] = "pink"                                    #窗口背景色，其他背景色见：blog.csdn.net/chl0000/article/details/7657887
        #self.init_window_name.attributes("-alpha",0.9)                          #虚化，值越小虚化程度越高
        #标签
        self.source_train_label = Label(self.init_window_name, text="source_train('corpora/train.vi')")
        self.source_train_label.grid(row=0, column=0)
        self.target_train_label = Label(self.init_window_name, text="target_train('corpora/train.en')")
        self.target_train_label.grid(row=0, column=12)
        self.batch_size_label = Label(self.init_window_name, text="batch_size(32)")
        self.batch_size_label.grid(row=0, column=22)
        self.learning_rate_label = Label(self.init_window_name, text="learning_rate(0.0001)")
        self.learning_rate_label.grid(row=0, column=32)
        self.log_directory_label = Label(self.init_window_name, text="log_directory('logdir')")
        self.log_directory_label.grid(row=20, column=0)
        self.maxlen_label = Label(self.init_window_name, text="max length")
        self.maxlen_label.grid(row=20, column=12)
        self.epoch_label = Label(self.init_window_name, text="epoch(10-20)")
        self.epoch_label.grid(row=20, column=22)
        self.min_cnt_label = Label(self.init_window_name, text="min_cnt(20)")
        self.min_cnt_label.grid(row=20, column=32)
        self.score_label = Label(self.init_window_name, text="结果路径（readme）：")
        self.score_label.grid(row=62, column=12)
        #文本框
        self.source_train = Entry(self.init_window_name, width=20)  #source_train
        self.source_train.grid(row=1, column=0, rowspan=10, columnspan=10)
        self.target_train = Entry(self.init_window_name, width=20)  #target_train
        self.target_train.grid(row=1, column=12, rowspan=10, columnspan=10)
        self.batch_size = Entry(self.init_window_name, width=20)  #batch_size
        self.batch_size.grid(row=1, column=22, rowspan=10, columnspan=10)
        self.learning_rate = Entry(self.init_window_name, width=20)  # learning rate
        self.learning_rate.grid(row=1, column=32, rowspan=10, columnspan=10)
        self.log_directory = Entry(self.init_window_name, width=20)  # log directory
        self.log_directory.grid(row=21, column=0, rowspan=10, columnspan=10)
        self.maxlen = Entry(self.init_window_name, width=20)  # max length
        self.maxlen.grid(row=21, column=12, rowspan=10, columnspan=10)
        self.epoch = Entry(self.init_window_name, width=20)  # 遍历次数
        self.epoch.grid(row=21, column=22, rowspan=10, columnspan=10)
        self.min_cnt = Entry(self.init_window_name, width=20)  # 最少出现次数
        self.min_cnt.grid(row=21, column=32, rowspan=10, columnspan=10)
        self.score = Entry(self.init_window_name, width=20)  # 结果路径
        self.score.grid(row=62, column=22, rowspan=10, columnspan=10)
        #按钮
        self.str_trans_to_md5_button = Button(self.init_window_name, text="参数修改", bg="lightblue", width=10,command=self.str_trans_to_md5)  # 调用内部方法  加()为直接调用
        self.str_trans_to_md5_button.grid(row=32, column=0)
        self.str_trans_to_md5_button1 = Button(self.init_window_name, text="预处理！", bg="lightblue", width=15,command=self.prep)  
        self.str_trans_to_md5_button1.grid(row=32, column=12)
        self.str_trans_to_md5_button2 = Button(self.init_window_name, text="训练！", bg="lightblue", width=15,command=self.train)  
        self.str_trans_to_md5_button2.grid(row=32, column=22)
        self.str_trans_to_md5_button3 = Button(self.init_window_name, text="说明（Readme）", bg="lightblue", width=15,command=self.readme)  
        self.str_trans_to_md5_button3.grid(row=42, column=12)
        self.str_trans_to_md5_button3 = Button(self.init_window_name, text="数据添加", bg="lightblue", width=15,command=self.addtst)  
        self.str_trans_to_md5_button3.grid(row=42, column=22)
        self.str_trans_to_md5_button3 = Button(self.init_window_name, text="参数查看", bg="lightblue", width=15,command=self.readhyper)  
        self.str_trans_to_md5_button3.grid(row=42, column=0)
        self.str_trans_to_md5_button4 = Button(self.init_window_name, text="结果评分v1.0", bg="lightblue", width=15,command=self.eval)  
        self.str_trans_to_md5_button4.grid(row=52, column=12)
        self.str_trans_to_md5_button4 = Button(self.init_window_name, text="结果评分v2.0", bg="lightblue", width=15,command=self.eval2)  
        self.str_trans_to_md5_button4.grid(row=52, column=22)
        self.str_trans_to_md5_button5 = Button(self.init_window_name, text="查看结果", bg="lightblue", width=15,command=self.checkscore)  
        self.str_trans_to_md5_button5.grid(row=62, column=0)
        self.str_trans_to_md5_button5 = Button(self.init_window_name, text="查看预处理A", bg="lightblue", width=15,command=self.prepA)  
        self.str_trans_to_md5_button5.grid(row=82, column=0)
        self.str_trans_to_md5_button5 = Button(self.init_window_name, text="查看预处理B", bg="lightblue", width=15,command=self.prepB)  
        self.str_trans_to_md5_button5.grid(row=82, column=22)

        
    #功能函数
    def str_trans_to_md5(self):
        
        data = ''
        with open('hyperparams.py', 'r+', encoding='UTF-8') as f:
            for line in f.readlines():
             print(line)
             if(line.find('    batch') == 0):
                 line = '    batch_size = %s'  %self.batch_size.get() + ' # alias = N\n'
                 #print(line)
             if(line.find('    lr') == 0):
                 line = '    lr = %s'  %self.learning_rate.get() + ' # learning rate.\n'
             if(line.find('    logdir') == 0):
                 line = '    logdir = %s'  %self.log_directory.get() + ' # log directory\n'
             if(line.find('    maxlen') == 0):
                 line = '    maxlen = %s'  %self.maxlen.get() + ' # 每句话长度\n'
             if(line.find('    num_epochs') == 0):
                 line = '    num_epochs = %s'  %self.epoch.get() + ' # 遍历次数\n'
             data += line
        with open('hyperparams.py', 'r+', encoding='UTF-8') as f:
            f.writelines(data)
      

    def prep(self):
        if __name__ == "__main__":
            make_vocab(hp.source_train, "de.vocab.tsv")
            make_vocab(hp.target_train, "en.vocab.tsv")
            print("Done")


    def train(self):
        if __name__ == "__main__":
            # Load vocabulary    
            de2idx, idx2de = load_de_vocab()
            en2idx, idx2en = load_en_vocab()
    
            # Construct graph
            g = Graph("train"); print("Graph loaded")
    
            # Start session
            sv = tf.train.Supervisor(graph=g.graph, 
                                     logdir=hp.logdir,
                                     save_model_secs=0)
            with sv.managed_session() as sess:
                for epoch in range(1, hp.num_epochs+1): 
                    print('%d' % (epoch))
                    if sv.should_stop(): break
                    for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                        sess.run(g.train_op)
                
                    gs = sess.run(g.global_step)   
                    sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    
            print("Done")    
          

    def readme(self):
        #os.system('notepad')
        os.system('notepad readme.txt')

    def addtst(self):
        #os.system('notepad')
        os.system('notepad D:\\VScoder\\de-en\\corpora\\tst1.vi')


    def readhyper(self):
        #os.system('notepad')
        os.system('notepad hyperparams.py')


    def eval(self):
        if __name__ == "__main__":
            eval()
            print("评分结束")

    def eval2(self):
        if __name__ == "__main__":
            eval2()
            print("评分结束")

    def checkscore(self):
        #os.system('notepad')
        os.system('notepad D:\\VScoder\\de-en\\results\\%s'  %self.score.get())


    def prepA(self):
        #os.system('notepad')
        os.system('notepad D:\\VScoder\\de-en\\preprocessed\\de.vocab.tsv')


    def prepB(self):
        #os.system('notepad')
        os.system('notepad D:\\VScoder\\de-en\\preprocessed\\en.vocab.tsv')

def gui_start():
    init_window = Tk()              #实例化出一个父窗口
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()          #父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示


gui_start()