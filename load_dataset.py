from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip,Resize
from torch.utils.data import Dataset,Subset
import torchvision.transforms as transforms
import os
import numpy as np
from avalanche.benchmarks.utils import AvalancheDataset,AvalancheSubset
from avalanche.benchmarks import NCScenario, nc_benchmark,dataset_benchmark,ni_benchmark
from PIL import Image
import torch
from avalanche.benchmarks  import benchmark_with_validation_stream
from parse_data_path import *
import os
'''
timestamp_index stand for, for each timestamp, the index of instance in the txt file, since each subset represent
one timestamp data, thus we need to have the index of data of each timestamp
timestamp_index[0] == the set of index of data belong to bucket 1
'''
def get_instance_time(args,idx,all_timestamp_index):
    for index,list in enumerate(all_timestamp_index):
        if(idx in list):
            return index
    assert False, "couldn't find timestamp info for data with index {}".format(idx)

def get_feature_extract_loader(args):
    import pdb; pdb.set_trace()
    dataset = CLEARDataset(args,data_txt_path='{}/data_cache/data_all_path.txt'.format(args.split),stage='all')
    all_timestamp_index = dataset.get_timestamp_index()
    return dataset, all_timestamp_index


class FADDataset(Dataset):
    def __init__(self, args, data_txt_path, stage):
        assert stage in ['train','test','all']
        print('Preparing {}'.format(stage))
        self.args = args
        self.n_classes = args.num_classes
        self.n_experiences = args.timestamp
        self.stage = stage
        if(os.path.isfile(data_txt_path)==False):
            print('loading data_list from folder')
            parse_data_path(args)
        else:
            print('loaded exist data_list')
        # data_txt_path: '../temp_folder/data_cache/data_all_path.txt'
        self.prepare_data(data_txt_path)
        self.targets = torch.from_numpy(np.array(self.targets))
        print('Using split {}'.format(self.args.split))
        # self.train_transform,self.test_transform=self.get_transforms()
    
    def get_timestamp_index(self):
        return self.timestamp_index

    def get_targets(self):
        return self.targets
    
    def prepare_data(self,data_txt_path):
        samples, targets = [], []
        timestamp_index = [[] for i in range(self.n_experiences)]
        index = 0
        with open(data_txt_path,'r') as file:
            title=file.readline()
            while (True):
                line = file.readline()
                if(line == ''):
                    break
                line_list=line.split()
                targets.append(int(line_list[1]))
                timestamp_index[int(line_list[2])-1].append(index)
                samples.append(line_list[0])
                index += 1
                
                if(index % 10000 == 0):
                    print('finished processing data {}'.format(index))
        print("Finished processed {} data".format(index))
                
        self.targets = targets
        self.samples = samples
        self.timestamp_index = timestamp_index
        
        os.makedirs('{}/buffered_data/train'.format(self.args.split),exist_ok=True)
        os.makedirs('{}/buffered_data/test'.format(self.args.split),exist_ok=True)
        os.makedirs('{}/buffered_data/all'.format(self.args.split),exist_ok=True)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # file_path='../{}/buffered_data/{}/{}.npy'.format(self.args.split,self.stage,str(index))
        '''
        when using pre-train feature and data_folder_path had already be updated
        When generating pretrain feature, the data_folder_path is original image path
        When finish generating pretrain feature, the data_folder_path is feature path
        '''
        if(self.args.pretrain_feature!='None'):
            sample, label = np.load(self.samples[index], allow_pickle=True), self.targets[index]
            sample = torch.tensor(np.mean(sample, axis=0))
        else:
            sample, label = Image.open(self.samples[index]),self.targets[index]
            array=np.array(sample)
            # some image may have 4 channel (alpha)
            if(array.shape[-1]==4):
                array=array[:,:,:3]
            elif(array.shape[-1]==1):
                array=np.concatenate((array, array, array), axis=-1)
            elif(len(array.shape)==2):
                array=np.stack([array,array,array],axis=-1)
            sample=Image.fromarray(array)
        return sample, label
    
class FADSubset(Dataset):
    def __init__(self, dataset, indices, targets, bucket):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.indices = indices
        self.targets = targets.numpy() # need to be in numpy(thus set of targets have only 10 elem,rather than many with tensor)
        self.bucket = bucket
    
    def get_indice(self):
        return self.indices
    
    def get_bucket(self):
        return self.bucket
    
    def __getitem__(self, idx):
        
        sample = self.dataset[idx][0]
        target = self.targets[idx]
        
        return sample, target
    
    def __len__(self):
        return len(self.targets)

def get_transforms(args):
    # Note that this is not exactly imagenet transform/moco transform for val set
    # Because we resize to 224 instead of 256
    if(args.pretrain_feature != 'None'):
        return None, None
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])  
    return train_transform, test_transform

def get_data_set_offline(args):

    train_Dataset = FADDataset(args,data_txt_path = '{}/data_cache/data_train_path.txt'.format(args.split), stage='train')
    print("Number of train data is {}".format(len(train_Dataset)))
    test_Dataset = FADDataset(args,data_txt_path = '{}/data_cache/data_test_path.txt'.format(args.split), stage='test')
    print("Number of test data is {}".format(len(test_Dataset)))

    n_experiences = args.timestamp
    train_timestamp_index, test_timestamp_index = train_Dataset.get_timestamp_index(), test_Dataset.get_timestamp_index()
    train_transform, test_transform = get_transforms(args)

    list_train_dataset = []
    list_test_dataset = []
    
    # Split All-Dataset to a sequence cl datasets
    for task_id in range(n_experiences):
        bucket_index = train_timestamp_index[task_id]
        train_sub = FADSubset(train_Dataset, bucket_index, train_Dataset.get_targets()[bucket_index], task_id)
        train_subset = AvalancheDataset(train_sub, task_labels = task_id)

        bucket_index = test_timestamp_index[task_id]
        test_sub = FADSubset(test_Dataset, bucket_index, test_Dataset.get_targets()[bucket_index], task_id)
        test_subset = AvalancheDataset(test_sub, task_labels = task_id)

        list_train_dataset.append(train_subset)
        list_test_dataset.append(test_subset)
    
    return dataset_benchmark(
        list_train_dataset, 
        list_test_dataset, 
        train_transform = train_transform,
        eval_transform = test_transform)
    # return ni_benchmark(
    #     list_train_dataset, 
    #     list_test_dataset, 
    #     n_experiences=len(list_train_dataset), 
    #     shuffle=False, 
    #     balance_experiences=True,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)

    # return nc_benchmark(
    #     list_train_dataset,
    #     list_test_dataset,
    #     n_experiences=len(list_train_dataset),
    #     task_labels=True,
    #     shuffle=False,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)

    # return nc_benchmark(
        # list_train_dataset,
        # list_test_dataset,
        # n_experiences=len(list_train_dataset),
        # task_labels=True,
        # shuffle=False,
        # class_ids_from_zero_in_each_exp=True,
        # one_dataset_per_exp=True,
        # train_transform=train_transform,
        # eval_transform=test_transform,
        # seed=args.random_seed)
    # valid_benchmark = benchmark_with_validation_stream(
    #         initial_benchmark_instance, 20, shuffle=False)
    # return valid_benchmark

def get_data_set_online(args):
    all_Dataset=CLEARDataset(args,data_txt_path='../{}/data_cache/data_all_path.txt'.format(args.split),stage='all')
    print("Number of all data is {}".format(len(all_Dataset)))
    n_experiences=args.timestamp
    all_timestamp_index=all_Dataset.get_timestamp_index()
    train_transform,test_transform=get_transforms(args)

    list_all_dataset = []

    for i in range(n_experiences):
        # choose a random permutation of the pixels in the image
        bucket_index=all_timestamp_index[i]
        all_sub = CLEARSubset(all_Dataset,bucket_index,all_Dataset.targets[bucket_index],i)
        all_set=AvalancheDataset(all_sub,task_labels=i)
        list_all_dataset.append(all_set)
    return dataset_benchmark(
        list_all_dataset, 
        list_all_dataset, 
        train_transform=train_transform,
        eval_transform=test_transform)


    # return ni_benchmark(
    #     list_all_dataset, 
    #     list_all_dataset, 
    #     n_experiences=len(list_all_dataset), 
    #     shuffle=False, 
    #     balance_experiences=True,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)


    # return nc_benchmark(
    #     list_all_dataset,
    #     list_all_dataset,
    #     n_experiences=len(list_all_dataset),
    #     task_labels=True,
    #     shuffle=False,
    #     class_ids_from_zero_in_each_exp=True,
    #     one_dataset_per_exp=True,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)

    # return nc_benchmark(
    #     list_all_dataset,
    #     list_all_dataset,
    #     n_experiences=len(list_all_dataset),
    #     task_labels=True,
    #     shuffle=False,
    #     class_ids_from_zero_in_each_exp=True,
    #     one_dataset_per_exp=True,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)
    
if __name__ == '__main__':
    dataset=get_data_set_online()
    import pdb;pdb.set_trace()
    print('finsih')
# # from torchvision.datasets import MNIST
# # from avalanche.benchmarks.datasets import default_dataset_location
# # dataset_root = default_dataset_location('mnist')
# # train_set = MNIST(root=dataset_root,
# #                       train=True, download=True)
# # import pdb;pdb.set_trace()


# from torchvision.datasets import MNIST
# from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
# from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
# train_transform = Compose([
#     RandomCrop(28, padding=4),
#     ToTensor(),
#     Normalize((0.1307,), (0.3081,))
# ])

# test_transform = Compose([
#     ToTensor(),
#     Normalize((0.1307,), (0.3081,))
# ])

# mnist_train = MNIST(
#     './data/mnist', train=True, download=True, transform=train_transform
# )
# mnist_test = MNIST(
#     './data/mnist', train=False, download=True, transform=test_transform
# )
# scenario = ni_benchmark(
#     mnist_train, mnist_test, n_experiences=10, shuffle=True, seed=1234,
#     balance_experiences=True
# )

# train_stream = scenario.train_stream

# for experience in train_stream:
#     t = experience.task_label
#     exp_id = experience.current_experience
#     training_dataset = experience.dataset
#     print('Task {} batch {} -> train'.format(t, exp_id))
#     print('This batch contains', len(training_dataset), 'patterns')
#     print("Current Classes: ", experience.classes_in_this_experience)
# [len(dataset.test_stream[ii].dataset) for ii in range(10)]
