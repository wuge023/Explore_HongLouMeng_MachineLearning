# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:54:32 2018

@author: lingyun
"""

import codecs
import os
import matplotlib.pyplot as plt
#mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
#mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

word = ['与','且','为','乃','么','之','乎','也','于','亦','仍',
        '以','但','何','偏','其','再','则','即','向','吗',
        '呀','呢','咧','因','尚','就','往','很','或','所',
        '故','方','更','焉','然','皆','矣','罢','者','而','若','让','越']
counter = {}
counter_sum = {}
input_folder = "chapters"
number_of_chapters = 120
output_file = codecs.open("words.csv", "w")
label = [0, 1]
def words_count():
    for chapter_no in range(1, number_of_chapters + 1):
        path = os.path.join(input_folder, "%d.txt" % chapter_no)        
        inpath = codecs.open(path, "r", encoding="utf-8")  
        for w in word:
            counter[w] = 0
            counter_sum[chapter_no] = 0
        with inpath as fr:
            for line in fr:
                line = line.strip()
                if len(line) == 0:
                    continue
                for w in line:
                    counter_sum[chapter_no]+=1
                    if w in word:
                        counter[w] += 1
                        if not w in counter:
                            counter[w] = 0
                        else:
                            counter[w] += 1                                  
        counter_list = sorted(counter.items(), key=lambda x: x[0], reverse=False)
        #print(counter_list)
        #laba = list(map(lambda x: x[0], counter_list))
        #print(laba)
        #output_file.write("第%d章," % chapter_no)        
        for item in counter_list:   
            output_file.write("%d," % item[1])
        if chapter_no<=80:
            output_file.write("%d,%.4f,%d\n" % (sum(counter.values()),sum(counter.values())/counter_sum[chapter_no], label[1]))
        else:
            output_file.write("%d,%.4f,%d\n" % (sum(counter.values()),sum(counter.values())/counter_sum[chapter_no], label[0]))
        
words_count()
output_file.close()        

''' 44个常用文言虚词和白话文虚词：word.csv
        属性名：['与', 且','为', '乃', '么', '之', '乎', '也', '于', '亦', '仍', '以', '但', '何', '偏', '其', '再', 
                 '则', '即', '向', '吗', '呀', '呢', '咧', '因', '尚', '就', '往', '很', '或', '所',
                 '故', '方', '更', '焉', '然', '皆', '矣', '罢', '者', '而', '若', '让', '越']
        最后一列：sum, rate, label
[
 '之', '其', '或', '亦', '方', '于', '即', '皆', '因', '仍', 
 '故', '尚', '呢', '了', '的', '着', '一', '不', '乃', '呀', 
 '吗', '咧', '啊', '把', '让', '向', '往', '是', '在', '越', 
 '再', '更', '比', '很', '偏', '别', '好', '可', '便', '就',
 '但', '儿',                  # 42 个文言虚词
 '又', '也', '都', '要',       # 高频副词 
 '这', '那', '你', '我', '他'  # 高频代词
 '来', '去', '道', '笑', '说'  #高频动词
]        
''''''
18个文言文常用虚词：
"而" 何 乃 之 乎 其 且 若 所 为 焉 也 以 因 于 与 则 者 
import pandas as pd

words = []
for i in word:
    words.append(i)
words.append("sum")
words.append("rate")  
words.append("label") 
data = pd.read_csv("words.csv", names=words, encoding="utf-8")

X = pd.DataFrame()
old_word = ["而","何", "乃", "之", "乎", "其", "且", "若", "所", "为", "焉", "也", "以", "因", "于", "与", "则", "者"]
for num, iword in enumerate(old_word):
    X[num] = data[iword]
X[19] = X.apply(lambda x: x.sum(), axis=1)#sum
#rate？？？？
for chapter_no in range(1,121):
    X.iloc[chapter_no-1, 18] = X.iloc[chapter_no-1, 18]/counter_sum[chapter_no]#每一章虚词总数 / 本章总字数??????
#plt.show(X.T.plot(kind='bar')) 
'''

import pandas as pd
summa = pd.read_csv("words.csv",header=None, encoding="utf-8")
summa = summa.iloc[:,0:44]
sum_perword = []
for i in range(0,44):
    sum_perword.append(sum(summa[i]))

cloud = open("cloud.csv", "w")  
for i,w in enumerate(word):
    cloud.write(w)
    cloud.write(",%d\n" % sum_perword[i])
cloud.close() 
'''   
clouds = open('cloud.txt','w')
for num,w in enumerate(word):
    for i in range(1,sum_perword[num]+1):
        clouds.write(w);
clouds.close()
clouds = open('cloud.txt','r').read()
cy = WordCloud(font_path="C:\\Users\\lingyun\\Downloads\\wryh.ttf").generate(clouds)
plt.imshow(cy)
plt.axis("off")
plt.show()
'''
dataset = pd.read_csv("words.csv",header=None)

fig = plt.figure(figsize=(50,10))
N = 44
label = []
for i in range(120):
    label.append(i+1)
for i in range(44):
    plt.plot(label, dataset[i],label=label[i])
plt.legend(word,loc='upper left')
    
plt.xlabel('chapters')
plt.ylabel('numbers')
plt.title('120 chapters')
plt.show()   