from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
import torch.optim as optim
from torch.optim import SGD
from torchvision.models import resnet18,resnet50,resnet101
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torchvision
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.gdumb import GDumbPlugin
from avalanche.training.strategies import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC,JointTraining,\
    SynapticIntelligence,CoPE, OWM, RAWM, RWM, ELMA
from avalanche.training.strategies.icarl import ICaRL
from avalanche.training.strategies.ar1 import AR1
from avalanche.training.strategies.deep_slda import StreamingLDA
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.training.plugins.load_best import LoadBestPlugin
from load_dataset import *
from parse_data_path import *
import argparse
from get_config import *
from extract_feature import *
from parse_log_to_result import *
import glob
import json
import random
import matplotlib.pyplot as plt
import glob
import sys
import torch.nn as nn

from avalanche.models.SCNN import CNNSelfAttention as scnn
from avalanche.models.simple_mlp import SimpleMLP, LoRAMLP, SimpleMLP_withbninitweight, SimpleMLP_withinitweight, LoRAMLP_withinitweight

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
                   
def build_logger(name):
    # log to text file
    text_logger = TextLogger(open('{}/log/log_{}.txt'.format(args.split,name), 'w'))
    
    # print to stdout
    interactive_logger = InteractiveLogger()
    tb_logger = TensorboardLogger('{}/tb_data'.format(args.split))
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, epoch_running=True,experience=True, stream=True,trained_experience=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        # timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        # cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=args.num_classes, save_image=False, stream=True),
        # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )
    return text_logger ,interactive_logger,eval_plugin

def make_scheduler(optimizer, step_size, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return scheduler

def extract(args):
    print('Extract/Check feature for {}'.format(args.data_folder_path))
    args.temp_split=args.split
    args.split='temp_folder' # dummy folder for extracting feature
    args = extract_feature(args)
    print('Finished extract/check feature {}'.format(args.pretrain_feature))
    os.system('rm -rf temp_folder')
    args.split = args.temp_split
    args.data_folder_path = os.path.join(args.feature_path, args.pretrain_feature)
    return args

def plot_eer_dict_onlysingle(data_dict, output_dir):
    
    list_length = len(list(data_dict.values())[0])

    os.makedirs(output_dir, exist_ok=True)

    for i in range(list_length):
        plt.figure()  
        x = list(data_dict.keys())  
        y = [values[i] * 100 for values in data_dict.values()] 
        plt.plot(x, y, label=f"EER")
        plt.xlabel('After Training On Task {}'.format(i + 1))
        plt.ylabel('Evaluate on Task {}'.format(i + 1))
        plt.title('Evaluation EER (%)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'Evaluation_on_task_{i+1}.png'))

def plot_eer_dict(data_dict, output_dir):
    
    list_length = len(list(data_dict.values())[0])

    os.makedirs(output_dir, exist_ok=True)

    # Plot individual graphs
    for i in range(list_length):
        plt.figure()  
        x = list(data_dict.keys())  
        y = [values[i] * 100 for values in data_dict.values()] 
        plt.plot(x, y, label=f"EER")
        plt.xlabel('After Training On Task {}'.format(i + 1))
        plt.ylabel('Evaluate on Task {}'.format(i + 1))
        plt.title('Evaluation EER (%)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'Evaluation_on_task_{i+1}.png'))
        plt.close()

    # Plot all curves on a single graph
    plt.figure()
    for i in range(list_length):
        y = [values[i] * 100 for values in data_dict.values()]
        plt.plot(x, y, label=f'Eval on Task {i+1}')
    
    plt.xlabel('After Training on Tasks')
    plt.ylabel('EER (%)')
    plt.title('Evaluation EER on All Tasks')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Evaluation_on_all_tasks.png'))
    plt.close()

def move_data_trinity(input_path,remove=False):
    '''
    Move data from /data to /scratch (for trinity server)
    It only move the data in args.data_folder_path, not the current script
    '''
    target_path=os.path.join('scratch/jiashi/',"/".join(input_path[1:].split('/')[:-1]))
    print('Moving data {} to local server'.format(input_path))
    # '/scratch/jiashi/data/jiashi/moco_resnet50_clear_10_feature'
    path_on_scratch=os.path.join(target_path,input_path.split('/')[-1])
    print(path_on_scratch)
    if(os.path.isdir(path_on_scratch)==False):
        if(remove==True):
            print('rm previous data path')
            assert False
        os.makedirs(target_path,exist_ok=True)
        os.system('cp -rf {} {}'.format(input_path,target_path))
    return path_on_scratch
    # return input_path

def main():
    global args
    args = get_config()
    set_seed(args.seed)
    try:
        restart = int(args.restart)
    except:
        print('restart flag must be 0/1')
        assert False
    if(restart == 1):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!You sure to remove the old checkpoint ???!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('enter Y/y to continue')
        value=input()
        if(value=="y" or value=='Y'):
            os.system("rm -rf {}".format(args.split))
            print('Removed old split folder')

    os.makedirs("{}".format(args.split), exist_ok=True)
    os.makedirs("{}/log/".format(args.split), exist_ok=True)
    os.makedirs("{}/model/".format(args.split), exist_ok=True)
    os.makedirs("{}/metric/".format(args.split), exist_ok=True)
    method_query = args.method.split() # list of CL method to run

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    Remember to delete the old feature path before generating new feature 
    '''
    if(args.pretrain_feature!='None'):
        if(args.data_test_path != '' and args.data_train_path != ''):
            data_test_path=args.data_test_path
            data_train_path=args.data_train_path

            train_num_instance_each_class=args.num_instance_each_class
            for stage in ['train','test']:
                if(stage=='train'):
                    args.data_folder_path=data_train_path
                    args.data_train_path=''
                    args.pretrain_feature='train_'+args.pretrain_feature
                    args=extract(args)
                    args.data_train_path=args.data_folder_path
                else:
                    args.num_instance_each_class=args.num_instance_each_class_test
                    args.data_folder_path=data_test_path
                    args.data_test_path=''
                    args.pretrain_feature=args.pretrain_feature.replace('train','test')
                    args=extract(args)
                    args.data_test_path=args.data_folder_path
                    args.num_instance_each_class=train_num_instance_each_class
        else:    
            args=extract(args)

    with open('{}/args.txt'.format(args.split), 'w') as f:
        print('args', args, file = f) # keep a copy of the args
    # os.system('cp -rf avalanche {}/'.format(args.split)) # keep a copy of the scripts
    for strate in method_query:
        for current_mode in ['offline']:
            print('========================================================')
            print('======FAD============******============FAD==============')
            print('====> Current Continual Learning Method is {} {} <===='.format(strate, current_mode))
            print('======FAD============******============FAD==============')
            print('========================================================')
            # skip previous train model if necessary
            model_path = sorted(glob.glob('{}/model/model_{}_{}*'.format(args.split,strate,current_mode)))
            if(len(model_path) == 0 and args.eval == True):
                checkpoint_path = '{}/model/model_{}_{}*'.format(args.split, strate, current_mode)
                print('Checkpoint for model {} is not found at path {}'.format(strate, checkpoint_path))
                continue
            
            # Get CL scenario
            scenario = get_data_set_offline(args) if(current_mode == 'offline') else get_data_set_online(args)
            # Define model
            if args.pretrain_feature=='None':
                pretrain=args.image_train_pretrain
                model=torchvision.models.__dict__[args.image_train_model_arch](pretrained=pretrain)
                if(args.image_train_model_arch=='resnet50' and args.image_train_attribute=='moco'):
                    model=moco_v2_yfcc_feb18_bucket_0_gpu_8(model)
            else:
                if strate=='ELMA':
                    model = LoRAMLP_withinitweight(num_classes = args.num_classes, input_size = args.pretrain_feature_shape, hidden_size = args.hidden_dim, hidden_layers = args.layer_num, drop_rate=0.0, lora_rank = args.lora_rank)    
                else:
                    # model = SimpleMLP(num_classes = args.num_classes, input_size = args.pretrain_feature_shape, hidden_size = 512, hidden_layers = 12, drop_rate=0.0)
                    # model = SimpleMLP_withbninitweight(num_classes = args.num_classes, input_size = args.pretrain_feature_shape, hidden_size = 512, hidden_layers = 12, drop_rate=0.0)
                    model = SimpleMLP_withinitweight(num_classes = args.num_classes, input_size = args.pretrain_feature_shape, hidden_size = args.hidden_dim, hidden_layers = args.layer_num, drop_rate=0.0)
                    # model = SimpleMLP_withinitweight(num_classes = args.num_classes, input_size = args.pretrain_feature_shape, hidden_size = 512, hidden_layers = 7, drop_rate=0.0)
                    # model=nn.Linear(args.pretrain_feature_shape,args.num_classes)        
            # Generate a Buffer for Replay/GDumb*/*Reservoir/AGEM*/CoPE 
            data_count = int(args.num_classes * args.num_instance_each_class) if current_mode == 'online' else int(args.num_classes * args.num_instance_each_class * args.buffer_ratio)
            data_count = min(args.max_memory_size, data_count) # buffer_size cannot be greater than args.max_memory_sizes
            if strate.split("_")[-1].isnumeric():
                buffer_size = int(data_count * strate.split("_")[-1])
            else:
                buffer_size = data_count
            print('=====> Buffer Size is {} <======'.format(buffer_size))
            
            # Check CPU/GPU
            if torch.cuda.is_available():
                model = model.cuda()
                if torch.cuda.device_count() > 1:
                    print("Let's use {} GPUs!".format(torch.cuda.device_count()))
                    model = nn.DataParallel(model)
                else:
                    print("Let's use 1 GPU!")
            else:
                print("Only use CPU!")
            # Check if load previous ckpt from previous method 
            if args.load_prev:
                if (len(model_path) != 0):
                    state_dict = torch.load(model_path[-1])
                    model.load_state_dict(state_dict)
                    print('Loaded previous model from {}'.format(model_path[-1]))
                else:
                    print("You want load previous methods model, but we cannot find it !!")
                    raise ValueError
            
            # optimizer=SGD(list(filter(lambda x: x.requires_grad, model.parameters())), lr=args.start_lr, weight_decay=float(args.weight_decay),momentum=args.momentum)
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.start_lr, weight_decay=float(args.weight_decay))
            scheduler = make_scheduler(optimizer,args.step_schedular_decay,args.schedular_step)
            # patience=5 # Number of epochs to wait without generalization improvements before stopping the training .
            # EarlyStoppingPlugin(patience, 'train_stream')
            plugin_list = [LRSchedulerPlugin(scheduler), LoadBestPlugin('train_stream')]
            text_logger ,interactive_logger,eval_plugin = build_logger("{}_{}".format(strate,current_mode))

            if strate=='CWRStar':
                cl_strategy = CWRStar(
                    model, optimizer,
                    CrossEntropyLoss(),cwr_layer_name=None, train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif strate=='ELMA':
                cl_strategy = ELMA(
                    model, optimizer,
                    CrossEntropyLoss(reduction = "none"),
                    train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin, device=device, plugins=plugin_list, split = args.split)
            elif 'Replay' in strate: 
                cl_strategy = Replay(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,mem_size=buffer_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif (strate=='JointTraining' and current_mode=='offline'):
                cl_strategy = JointTraining(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch*args.timestamp//3, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif 'GDumbFinetune' in strate:
                cl_strategy = GDumb(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=False,buffer='class_balance')
            # stanard gdumb= reset model+ class_balance buffer'
            elif 'GDumb' in strate:
                cl_strategy = GDumb(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=True,buffer='class_balance')
            elif 'BiasReservoir' in strate:
                if('reset' in strate):
                    resett=True
                else:
                    resett=False
                alpha_mode ='Dynamic' if 'Dynamic' in strate else 'Fixed'
                alpha_value=float(strate.split("_")[-1])
                cl_strategy = GDumb(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=resett,buffer='bias_reservoir_sampling',
                    alpha_mode=alpha_mode,alpha_value=alpha_value)
            # this is basically the 'reservoir sampling in the paper(no reset+ reservoir sampling'
            elif 'Reservoir' in strate:
                cl_strategy = GDumb(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=False,buffer='reservoir_sampling')
            
            elif 'Cumulative' in strate:
                if('reset' in strate):
                    resett=True
                else:
                    resett=False
                cl_strategy = Cumulative(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list,reset=resett)
            
            elif strate=='LwF':
                cl_strategy = LwF(
                    model, optimizer,
                    CrossEntropyLoss(),
                    alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
                    train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            
            elif strate=='OWM':
                cl_strategy = OWM(
                    model, optimizer,
                    CrossEntropyLoss(),
                    alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
                    train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif strate=='RAWM':
                cl_strategy = RAWM(
                    model, optimizer,
                    CrossEntropyLoss(),
                    alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
                    train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif strate=='RWM':
                cl_strategy = RWM(
                    model, optimizer,
                    CrossEntropyLoss(),
                    alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
                    train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif strate=='GEM':
                cl_strategy = GEM(
                    model, optimizer,
                    CrossEntropyLoss(), patterns_per_exp=data_count,memory_strength=0.5, train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif 'AGEMFixed' in strate:
                cl_strategy = AGEM(
                    model, optimizer,
                    CrossEntropyLoss(),patterns_per_exp=buffer_size,sample_size=buffer_size, train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list,reservoir=True)
            elif 'AGEM' in strate:
                cl_strategy = AGEM(
                    model, optimizer,
                    CrossEntropyLoss(),patterns_per_exp=buffer_size,sample_size=buffer_size, train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list,reservoir=False)
            elif strate=='EWC':
                cl_strategy = EWC(
                    model, optimizer,
                    CrossEntropyLoss(), ewc_lambda=0.4, mode='online',decay_factor=0.1,
                    train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif strate=='Naive':
                cl_strategy = Naive(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif strate=='ICaRL':
                cl_strategy = ICaRL(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif strate=='SynapticIntelligence':
                cl_strategy = SynapticIntelligence(
                    model, optimizer,
                    CrossEntropyLoss(), si_lambda=0.0001,train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif 'CoPE' in strate:
                cl_strategy = CoPE(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,mem_size=buffer_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif strate=='AR1':
                cl_strategy = AR1(
                    model, optimizer,
                    CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)
            elif strate=='StreamingLDA':
                cl_strategy = StreamingLDA(
                    slda_model=model, 
                    criterion=CrossEntropyLoss(), input_size= 224,num_classes=args.num_classes,train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                    evaluator=eval_plugin,device=device,plugins=plugin_list)       
            else:
                continue
            print('Starting experiment...')
            train_metric={}
            test_metric = {}
            if(strate=='JointTraining' and current_mode=='offline'):
                model_save_path='{}/model/model_{}_{}_time{}.pth'.format(args.split,strate,current_mode,0)
                if(args.eval==False):
                    train_metric[0]=cl_strategy.train(scenario.train_stream)
                test_metric[0]=cl_strategy.eval(scenario.test_stream)
                print('current strate is {} {}'.format(strate,current_mode))
                torch.save(model.state_dict(), model_save_path)
                with open("../{}/metric/train_metric_{}.json".format(args.split,strate), "w") as out_file:
                        json.dump(train_metric, out_file, indent = 6)
                with open("../{}/metric/test_metric_{}.json".format(args.split,strate), "w") as out_file:
                    json.dump(test_metric, out_file, indent = 6)
                
            else:
                train_list = scenario.train_stream
                cur_timestep=0
                if(len(model_path)!=0 and args.load_prev==True):
                    try:
                        with open("{}/metric/train_metric_{}.json".format(args.split,strate), "r") as file:
                            prev_train_metric=json.load(file)
                        with open("{}/metric/test_metric_{}.json".format(args.split,strate), "r") as file:
                            prev_test_metric=json.load(file)
                        #extract ../clear100_imgnet_res50/model/model_BiasReservoir_Dynamic_1.0_offline_time05.pth as 5
                        load_prev_time_index = int(model_path.split('_')[-1].split('.')[0][4:])
                        train_list = train_list[load_prev_time_index+1:]
                        cur_timestep = load_prev_time_index + 1
                        test_metric = prev_test_metric
                        train_metric = prev_train_metric
                        print('start runing from bucket {}'.format(cur_timestep))
                    except:
                        pass

                for experience in train_list:
                    model_save_path='{}/model/model_{}_{}_time{}.pth'.format(args.split,strate,current_mode,str(cur_timestep).zfill(2))
                    print("Start of experience: ", experience.current_experience)
                    print("Current Classes: ", experience.classes_in_this_experience)
                    print('current strate is {} {}'.format(strate,current_mode))
                    # offline
                    if(current_mode=='offline'):
                        # train returns a dictionary which contains all the metric values
                        print('current strate is {} {}'.format(strate,current_mode))
                        # test also returns a dictionary which contains all the metric values
                        if(args.eval==False):
                            train_metric[cur_timestep] = cl_strategy.train(experience)
                        print('Training completed')
                        print('Computing accuracy on the whole test set')
                        test_metric[cur_timestep] = cl_strategy.eval(scenario.test_stream)
                        print('current strate is {} {}'.format(strate,current_mode))
                    # online
                    else:
                        print('current strate is {} {}'.format(strate,current_mode))
                        print('Computing accuracy on the future timestamp')
                        test_metric[cur_timestep]=cl_strategy.eval(scenario.test_stream)
                        if(args.eval==False):
                            train_metric[cur_timestep] = cl_strategy.train(experience)
                        # train returns a dictionary which contains all the metric values
                        print('Training completed')
                        print('current strate is {} {}'.format(strate,current_mode))
                    torch.save(model.state_dict(), model_save_path)
                    log_path = '{}/log/'.format(args.split)
                    log_name = 'log_{}.txt'.format("{}_{}".format(strate,current_mode))
                    with open("{}/metric/train_metric_{}.json".format(args.split,strate), "w") as out_file:
                        json.dump(train_metric, out_file, indent = 6)
                    with open("{}/metric/test_metric_{}.json".format(args.split,strate), "w") as out_file:
                        # convert tensor to string for json dump
                        test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream']=\
                        test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream'].numpy().tolist()
                        json.dump(test_metric, out_file, indent = 6)
                    out_file.close()
                    cur_timestep += 1
                    if(args.eval==True):
                        break
                    # move_metric_to_main_node(log_path,log_name,main_server_path='/data/jiashi/metric')
        # Save EER Result
        print("==============> Saving EER Result... <===============")
        # if not os.path.exists("../{}/eer".format(args.split)):
        os.makedirs("result/{}/eer".format(args.split), exist_ok = True)
        save_path = "result/{}/eer/{}_r_{}.txt".format(args.split, strate, args.lora_rank)
        with open(save_path, "w") as file:
            file.write("Eval on ") #Task0 Task1 Task2 Task3 Task4 Task5 Task6 Task7 Task8 Task9\n")
            file.write(" ".join(["Task{}".format(expid + 1) for expid in range(len(cl_strategy.eer_record))]))
            file.write("\n")
            for taskid, line_element in cl_strategy.eer_record.items():
                file.write("Train after Task{} {}\n".format(taskid, line_element))
        print("==============> Saving EER Result on {}".format(save_path))
        plot_eer_dict(cl_strategy.eer_record, "result/{}/eer/{}/".format(args.split, strate))
        print("==============> Ploted EER Result on {}".format("result/{}/eer/{}/".format(args.split, strate)))
        print("==============> Saved <===============")


if __name__ == "__main__": 
    main()
    print('All CL methods finished successfully')
    sys.exit(0)