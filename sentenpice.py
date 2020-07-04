import jieba

# 创建停用词列表


# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    print("正在分词")
    sentence_depart = jieba.cut(sentence.strip())
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word != '\t':
            outstr += word
            outstr += " "
    return outstr

# 给出文档路径
filename = "test.zh"
outfilename = "out.zh"
inputs = open(filename, 'r', encoding='UTF-8')
outputs = open(outfilename, 'w', encoding='UTF-8')

# 将输出结果写入ou.txt中
for line in inputs:
    line_seg = seg_depart(line)
    outputs.write(line_seg + '\n')
    print("-------------------正在分词-----------")
outputs.close()
inputs.close()
print("分词成功！！！")