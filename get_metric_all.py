from dataclasses import replace
import os
from posixpath import split
import numpy as np
import argparse
import glob
from parse_log_to_result import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
def plot_2d_matrix(matrix, label_ticks, title, save_name,save_path='../plot_avalanche',min_acc=None, max_acc=None, normalize_by_column=False, plotting_trick=False):
    if plotting_trick:
        matrix_score = matrix.copy()
        matrix = matrix / matrix.mean(axis=0).reshape(1,-1)
        cbar_str = "Test Accuracy"
        format_score = lambda s : f"{s:.2f}"
    elif normalize_by_column:
        matrix = matrix / matrix.mean(axis=0).reshape(1,-1)
        cbar_str = "Test Accuracy / Average Test accuracy per column"
        format_score = lambda s : f"{s:.2f}"
        matrix_score = matrix
    else:
        cbar_str = "Test Accuracy"
        format_score = lambda s : f"{s:.2%}"
        matrix_score = matrix

    plt.figure(figsize=(10,10))
    x = ["Test " + n for n in label_ticks]
    y = ['Train ' + n for n in label_ticks]
    p = plt.imshow(matrix, interpolation='none', cmap=f'Blues', vmin=min_acc, vmax=max_acc)
    cbar = plt.colorbar()


    cbar.ax.set_ylabel(f"{cbar_str}", rotation=-90, va="bottom")
    cbar.ax.set_ylim(bottom=min_acc, top=max_acc)
    cbar.ax.invert_yaxis()

    plt.xticks(range(len(x)), x, fontsize=11, rotation = -90)
    plt.yticks(range(len(y)), y, fontsize=11)
    plt.title(title, fontsize=15)
    for i in range(len(x)):
        for j in range(len(y)):
            text = plt.text(j, i, format_score(matrix_score[i, j]),
                       ha="center", va="center", color="black")
    os.makedirs(save_path,exist_ok=True)
    plt.savefig(os.path.join(save_path,'{}.pdf'.format(save_name)))
    print("Image has been saved on {}".format(os.path.join(save_path,'{}.pdf'.format(save_name))))

argparser = argparse.ArgumentParser()
argparser.add_argument("--timestamp",type=int,default=10)
argparser.add_argument("--plot",type=int,default=0) # 1 for generating plot
argparser.add_argument("--verbose",type=int,default=0) # 1 for print out detailed metric
argparser.add_argument("--train_eval",type=int,default=0) # whether the code also include the evaluation of training set as well 
args = argparser.parse_args()


logpath='/data3/zhangxiaohui/clear10_moco_res50_public_private/metric'
log_list=sorted(os.listdir(logpath))

# unique_name=sorted(list(set(list(map(lambda log_name: log_name[:log_name.index('line')+4], log_list)))))
unique_name=sorted(list(set(list(map(lambda log_name: log_name.split(".")[0], log_list)))))

save_dict = {}


# each unique prefix
for name in unique_name:
    all_name=glob.glob(os.path.join(logpath,name+"*"))
    stat_list=[]
    plot_list=[]
    # for all element in same prefix 
    for elemt in all_name:
        result_list=[]
        log_file_name=os.path.join(logpath,elemt)
        file=open(log_file_name, 'r')
        while(True):
            try:
                line=file.readline()
            except:
                break
            if('Top1_Acc_Stream/eval_phase/test_stream/Task0' in line):
                try:
                    tmp_float_num = line.split()[-1]
                    if ',' in tmp_float_num:
                        tmp_float_num = tmp_float_num.replace(",", "")
                    result_list.append(float(tmp_float_num))
                except:
                    import pdb
                    pdb.set_trace()
            if not line:
                break
        file.close()
        if(args.train_eval==1):
            try:
                lowerIndex=[i for i in range(10,200,20)]
                upperIndex=[i for i in range(21,210,20)]
                eval_index=[k for i in range(len(lowerIndex)) for k in range(lowerIndex[i],upperIndex[i]-1) ]
                result_list=np.array(result_list)[eval_index]
            except:
                print('#################################################')
                print('Skipping {} for removing train result'.format(name))
                print('#################################################')
        if(len(result_list)!=int(args.timestamp*args.timestamp)):
            if('online' in name):
                # assert np.max(result_list[:args.timestamp])<0.3
                result_list=result_list[args.timestamp:]
            stat_list.append(np.mean(result_list))
            # print("{} count of {}, with mean of {}".format(name,len(result_list), np.mean(result_list)))
        else:
            result_list=np.array(result_list)
            if('online' in name):
                # assert np.max(result_list[:args.timestamp])<0.3
                index_list=get_online_protocol_index(class_=args.timestamp)
            else:
                index_list=get_offline_protocol_index(class_=args.timestamp)
            plot_list.append(result_list.reshape((args.timestamp,args.timestamp)))
            result_list=[str(np.mean(result_list[np.array(item[1])])) for item in index_list.items()]
            key_list=[item[0] for item in index_list.items()]
            stat_list.append(result_list)
    try:
        stat_list=np.array(stat_list).astype(float)
        stat_list=np.unique(stat_list,axis=0)
        if(args.verbose==1):
            print(stat_list)
        print("{} with {} of mean of {} std of {} with {} elem".format(name,", ".join(key_list),np.mean(stat_list,axis=0).tolist(),np.std(stat_list,axis=0),len(all_name)))
    except:
        print('---------------------------------------')
        print('skip {}'.format(name))
        print('---------------------------------------')
    if(args.plot==1 and 'online' not in name and 'Joint' not in name):
        try:
            plot_array=np.mean(np.array(plot_list),axis=0)
            plot_2d_matrix(plot_array, [str(i) for i in range(1,11)], '',name, normalize_by_column=False, plotting_trick=False)
            save_dict[name] = plot_array
        except:
            print('---------------------------------------')
            print('Plot skipped {}'.format(name))
            print('---------------------------------------')

totall_column = 10
for column in range(totall_column):
    X1 = [1,2,3,4,5,6,7,8,9,10]
    RWM = save_dict["test_metric_RWM"][:,column]
    EWC = save_dict["test_metric_EWC"][:,column]
    LwF = save_dict["test_metric_LwF"][:,column]
    Naive = save_dict["test_metric_Naive"][:,column]
    GDF = save_dict["test_metric_GDumbFinetune"][:,column]
    CWRStar = save_dict["test_metric_CWRStar"][:,column]
    GEM = save_dict["test_metric_GEM"][:,column]
    AGEM = save_dict["test_metric_AGEMFixed"][:,column]
    SI = save_dict["test_metric_SynapticIntelligence"][:,column]
    Res = save_dict["test_metric_Reservoir"][:,column]
    BRes = save_dict["test_metric_BiasReservoir_Dynamic_1"][:,column]
    OWM = save_dict["test_metric_OWM"][:,column]
    Replay = save_dict["test_metric_Replay"][:,column]

    plt.figure()

    # plt.plot(X1, RWM, label="RWM", color="#0000FF", marker='*', linestyle="-")
    # plt.plot(X1, EWC, label="EWC", color="#3CB371", marker='v', linestyle="-")
    # plt.plot(X1, LwF, label="LwF", color="#B22222", marker='1', linestyle="-")
    # plt.plot(X1, Naive, label="Finetune", color="#808000", marker='2', linestyle="-")
    # plt.plot(X1, GDF, label="GDF", color="#FFA500", marker='3', linestyle="-")
    # plt.plot(X1, CWRStar, label="CWR", color="#98FB98", marker='4', linestyle="-")
    # plt.plot(X1, GEM, label="GEM", color="#A0522D", marker='+', linestyle="-")
    # plt.plot(X1, AGEM, label="AGEM", color="#EE82EE", marker='x', linestyle="-")
    # plt.plot(X1, SI, label="SI", color="#87CEEB", marker='>', linestyle="-")
    # plt.plot(X1, Res, label="RF", color="#BC8F8F", marker='<', linestyle="-")
    # plt.plot(X1, BRes, label="BRF", color="#B0E0E6", marker='s', linestyle="-")
    # plt.plot(X1, OWM, label="OWM", color="#FF6347", marker='p', linestyle="-")
    # plt.plot(X1, Replay, label="Replay", color="#FF69B4", marker='8', linestyle="-")

    sns.lineplot(x=X1, y=RWM, label="RWM", marker='*', linestyle="-")
    sns.lineplot(x=X1, y=EWC, label="EWC", marker='v', linestyle="-")
    sns.lineplot(x=X1, y=LwF, label="LwF", marker='1', linestyle="-")
    sns.lineplot(x=X1, y=Naive, label="Finetune", marker='2', linestyle="-")
    sns.lineplot(x=X1, y=GDF, label="GDF", marker='3', linestyle="-")
    sns.lineplot(x=X1, y=CWRStar, label="CWR", marker='4', linestyle="-")
    sns.lineplot(x=X1, y=GEM, label="GEM", marker='+', linestyle="-")
    sns.lineplot(x=X1, y=AGEM, label="AGEM", marker='x', linestyle="-")
    sns.lineplot(x=X1, y=SI, label="SI", marker='>', linestyle="-")
    sns.lineplot(x=X1, y=Res, label="RF", marker='<', linestyle="-")
    sns.lineplot(x=X1, y=BRes, label="BRF", marker='s', linestyle="-")
    sns.lineplot(x=X1, y=OWM, label="OWM", marker='p', linestyle="-")
    sns.lineplot(x=X1, y=Replay, label="Replay", marker='8', linestyle="-")
    X_labels = [1,2,3,4,5,6,7,8,9,10]
    plt.xticks(X1, X_labels, rotation=0)
    plt.legend()
    plt.title("Acc(%) on Experience {}".format(column + 1))
    plt.xlabel("Experiences")
    plt.ylabel("Acc(%)")

    # Use sns.despine() to remove spines around the plot
    sns.despine()

    # Save the figure
    plt.savefig("./rwm_clear_exp_{}.pdf".format(column + 1))
    
    # Display the plot (if needed)
    plt.show()
                
    # X_labels = [1,2,3,4,5,6,7,8,9,10]
    # plt.xticks(X1,X_labels,rotation=0)
    # plt.legend()
    # plt.title("Acc(%) on Experience {}".format(column + 1))
    # plt.xlabel("Experiences")
    # plt.ylabel("Acc(%)")
    # # plt.xlim([5,50])
    # plt.show()
    # plt.savefig("./rwm_clear_exp_{}.pdf".format(column + 1)) 
    # g = sns.FacetGrid(data=pd.DataFrame(data={'X1': X1, 'RWM': RWM, 'EWC': EWC, "LwF": LwF, "Finetune": Naive, "GDF": GDF, "CWR": CWRStar, "GEM": GEM, "AGEM": AGEM, "SI": SI, "RF": Res, "BRF": BRes, "OWM": OWM, "Replay": Replay}), col='X1', col_wrap=1)
    # g.map_dataframe(sns.lineplot, x='X1', y='value', hue='variable', markers=True, linestyle='-')
    # g.set_axis_labels("Experiences", "Acc(%)")
    # g.set_titles("Acc(%) on Experience {col_name}")
    # g.add_legend()
    # g.savefig("./rwm_clear_exp_{}.pdf".format(column + 1))

    # data = pd.DataFrame({'X1': X1, 'RWM': RWM, 'EWC': EWC, "LwF": LwF, "Finetune": Naive, "GDF": GDF, "CWR": CWRStar, "GEM": GEM, "AGEM": AGEM, "SI": SI, "RF": Res, "BRF": BRes, "OWM": OWM, "Replay": Replay})
    # data_long = pd.melt(data, id_vars='X1', var_name='Method', value_name='Accuracy')

    # # Create a FacetGrid and plot lineplot
    # g = sns.FacetGrid(data_long, col='X1', col_wrap=1, hue='Method', sharey=False)
    # g.map_dataframe(sns.lineplot, x='X1', y='Accuracy', markers=True, linestyle='-')
    # g.set_axis_labels("Experiences", "Acc(%)")
    # g.set_titles("Acc(%) on Experience {col_name}")
    # g.add_legend()
    # g.savefig("./rwm_clear_exp_{}.pdf".format(column + 1))



