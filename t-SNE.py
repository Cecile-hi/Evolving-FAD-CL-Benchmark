from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
import pandas as pd
import torch
import os
from random import shuffle
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 6})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


ALL_NUMBER = 100

DATA_DIR = "/data3/zhangxiaohui/FAD-CL-Benchmark/features/imagenet_resnet50_clear_10_feature/"

ALL_CLASS = ["BACKGROUND", "baseball", "bus", "camera", "cosplay", "dress", "hockey", "laptop", "racing", "soccer", "sweater"]
CLASS_TO_LABLE = {"BACKGROUND":0, "baseball":1, "bus":2, "camera":3, "cosplay":4, "dress":5, "hockey":6, "laptop":7, "racing":8, "soccer":9, "sweater":10}

if __name__ == "__main__":

    # features = torch.zeros((ALL_NUMBER, 2048))
    features = np.zeros([ALL_NUMBER, 400, 60])
    labels = []
    count = 0
    for i in range(1, 18):
        for current_class in ALL_CLASS:
            features_dir = os.path.join(DATA_DIR, str(i), current_class)
            all_features = os.listdir(features_dir)
            for path in all_features:
                feature_path = os.path.join(features_dir, path)
                # current_x = torch.load(feature_path)
                current_x = np.load(feature_path)
                if len(current_x) < 400:
                    
                features[count] = current_x
                labels.append(CLASS_TO_LABLE[current_class])
                count += 1

    '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=20) #, n_iter=1500, n_iter_without_progress=50
    # X=np.loadtxt('val_x-vectors_clean-2.csv', dtype=float)
    # y=np.loadtxt('val_clean_labels-2.csv', dtype=float)
    X = np.array(features)
    label=np.array(labels)
    #print(X.values)
    # print(label)
    mylabel = []
    myname = []
    X_tsne = tsne.fit_transform(X)
    background, baseball, bus, camera, cosplay, dress, hockey, laptop, racing, soccer, sweater = [], [], [], [], [], [], [], [], [], [], [] 
    # import pdb
    # pdb.set_trace()
    for i in range(len(label)):
        if(label[i]==0.0):
            # mylabel.append('r')
            # myname.append(ALL_CLASS[0])
            background.append(X_tsne[i])
        elif(label[i]==1.0):
            # mylabel.append('y')
            # myname.append(ALL_CLASS[1])
            baseball.append(X_tsne[i])
        elif(label[i]==2.0):
            # mylabel.append('b')
            # myname.append(ALL_CLASS[2])
            bus.append(X_tsne[i])
        elif(label[i]==3.0):
            # mylabel.append('g')
            # myname.append(ALL_CLASS[3])
            camera.append(X_tsne[i])
        elif(label[i]==4.0):
            # mylabel.append('#00BFFF')
            # myname.append(ALL_CLASS[4])
            cosplay.append(X_tsne[i])
        elif(label[i]==5.0):
            # mylabel.append('#00CED1')
            # myname.append(ALL_CLASS[5])
            dress.append(X_tsne[i])
        elif(label[i]==6.0):
            # mylabel.append('#1E90FF')
            # myname.append(ALL_CLASS[6])
            hockey.append(X_tsne[i])
        elif(label[i]==7.0):
            # mylabel.append('#DC143C')
            # myname.append(ALL_CLASS[7])
            laptop.append(X_tsne[i])
        elif(label[i]==8.0):
            # mylabel.append("#3CB371")
            # myname.append(ALL_CLASS[8])
            racing.append(X_tsne[i])
        elif(label[i]==9.0):
            # mylabel.append("#98FB98")
            # myname.append(ALL_CLASS[9])
            soccer.append(X_tsne[i])
        elif(label[i]==10.0):
            # mylabel.append("#A0522D")
            # myname.append(ALL_CLASS[10])
            sweater.append(X_tsne[i])
        # elif(label[i]==11.0):
        #     mylabel.append("#B0E0E6")
        #     myname.append(ALL_CLASS[11])
        else:
            import pdb
            pdb.set_trace()
    #print(mylabel)
    background = np.array(background)
    baseball = np.array(baseball)
    bus = np.array(bus)
    camera = np.array(camera)
    cosplay = np.array(cosplay)
    dress = np.array(dress)
    hockey = np.array(hockey)
    laptop = np.array(laptop)
    racing = np.array(racing)
    soccer = np.array(soccer)
    sweater = np.array(sweater)

    plt.figure()

    # plt.scatter(X_tsne[:,0],X_tsne[:,1],c=mylabel, label= myname, s=0.5, alpha = 0.5)
    type1 = plt.scatter(background[:,0], background[:,1], c = "r", s=0.5, alpha = 0.5)#, label = "background")
    type2 = plt.scatter(baseball[:,0], baseball[:,1], c = "y", s=0.5, alpha = 0.5)#, label = "baseball")
    type3 = plt.scatter(bus[:,0], bus[:,1], c = "b", s=0.5, alpha = 0.5)#, label = "bus")
    type4 = plt.scatter(camera[:,0], camera[:,1], c = "g", s=0.5, alpha = 0.5)#, label = "camera")
    type5 = plt.scatter(cosplay[:,0], cosplay[:,1], c = "#FFFF00", s=0.5, alpha = 0.5)#, label = "cosplay")
    type6 = plt.scatter(dress[:,0], dress[:,1], c = "#00CED1", s=0.5, alpha = 0.5)#, label = "dress")
    type7 = plt.scatter(hockey[:,0], hockey[:,1], c = "#9400D3", s=0.5, alpha = 0.5)#, label = "hockey")
    type8 = plt.scatter(laptop[:,0], laptop[:,1], c = "#000000", s=0.5, alpha = 0.5)#, label = "laptop")
    type9 = plt.scatter(racing[:,0], racing[:,1], c = "#3CB371", s=0.5, alpha = 0.5)#, label = "racing")
    type10 = plt.scatter(soccer[:,0], soccer[:,1], c = "#FF8C00", s=0.5, alpha = 0.5)#, label = "soccer")
    type11 = plt.scatter(sweater[:,0], sweater[:,1], c = "#A0522D", s=0.5, alpha = 0.5)#, label = "sweater")
    # plt.legend()
    font = {'weight': 'normal', 'size': 5}
    plt.legend((type1, type2, type3, type4, type5, type6, type7, type8, type9, type10, type11), (u"background", u"baseball", u"bus", u"camera", u"cosplay", u"dress", u"hockey", u"laptop", u"racing", u"soccer", u"sweater"), prop = font)
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.show()
    plt.savefig(r'aaaa.pdf')



# fig = plot_embedding(X_tsne, label, 't-SNE embedding of the LSWF')
# plt.show(fig)


# print("Org data dimension is {}. 
#       Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
      
#   '''嵌入空间可视化'''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.show()
# if __name__ == '__main__':
#     print(1)