#!/usr/bin/env python
# coding: utf-8

# In[59]:


from utils.utils import gait_dataloader
from utils.utils import init_model,  patients_idx_to_cycles_idx, find_cycles_idx_by_patient_idx
from sklearn.model_selection import train_test_split

from utils.utils import init_model,accuracy_end_to_end, GridSearchCV
import os
import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import seaborn as sn

from sklearn.metrics import f1_score


from utils.utils import *

def load_dataset_v1(dir_dataset,channel_first=True,flatten=True,shuffle=False,random_state=0):
    index = [0,1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,22,23,24,29]
    
    npzfile = np.load(dir_dataset + 'RightPC.npz')
    
    x_train_all = npzfile['Input']
    y_train = npzfile['Output']
    y_train = np.ravel(y_train)
    x_train = x_train_all[:,:,index]
    if channel_first:
        x_train = x_train.transpose(0,2,1)
    
    npzfile = np.load(dir_dataset + 'LeftPC.npz')
    x_val_all = npzfile['Input']
    y_val = npzfile['Output']
    y_val = np.ravel(y_val)
    x_val = x_val_all[:,:,index]
    if channel_first:
        x_val = x_val.transpose(0,2,1)
    
    npzfile = np.load(dir_dataset + 'TD.npz')
    x_test_all = npzfile['Input']
    y_test = npzfile['Output']
    y_test = np.ravel(y_test)
    x_test = x_test_all[:,:,index]
    if channel_first:
        x_test = x_test.transpose(0,2,1)
    
    if flatten == True:
        x_train = x_train.reshape([x_train.shape[0],-1])
        x_val = x_val.reshape([x_val.shape[0],-1])
        x_test = x_test.reshape([x_test.shape[0],-1])
    
    nb_classes = len(np.unique(np.concatenate((y_train, y_val, y_test), axis=0)))
    
    return x_train, y_train ,x_val, y_val, x_test, y_test, nb_classes

dataset = "archives/AQM/TDvsPC_new_no_shuffle/"

x_R, y_R ,x_L , y_L, x_TD, y_TD, nb_classes = load_dataset_v1(dataset,channel_first=True,flatten=False,shuffle=False,random_state=0)

X_d = np.concatenate([x_R,x_L,x_TD])
# X_d = X_d.reshape([len(X_d),-1])
y_d = np.concatenate([y_R,y_L,y_TD])
import  pandas as pd

# R_data = pd.read_csv('DataPC - Right.csv', header=None)
# R_idx = R_data[2].to_numpy()


CP_data = pd.read_csv('DataCP - all.csv', header=None)
CP_idx = CP_data[1].to_numpy()

TD_data = pd.read_csv('DataTD - By Subject.csv', header=None)
TD_idx = TD_data[1].to_numpy()
TD_idx = TD_idx + CP_idx[-1]

cycle_end_idx = np.concatenate([CP_idx,TD_idx])
cycle_end_idx

patient_index_range = np.arange(len(CP_idx)+len(TD_idx))
patient_class = np.concatenate([np.ones(len(CP_idx)), np.zeros(len(TD_idx))])

x_train, x_test, y_train, y_test = train_test_split(patient_index_range, patient_class, test_size=0.4, stratify=patient_class,random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test, test_size=0.75, random_state=0, stratify=y_test)
print(np.unique(y_train,return_counts=True))
print(np.unique(y_val,return_counts=True))
print(np.unique(y_test,return_counts=True))


# In[60]:


from torch.utils.data import Dataset
from utils.utils import gait_dataloader
from utils.utils import init_model
## lime
import torch
import pytorch_lightning as pl
class Dataset_torch(Dataset):

    def __init__(self, data,with_label=True):
        self.with_label =  with_label
        
        if self.with_label:
            self.data_x, self.data_y = data
        else:
            self.data_x = data
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.with_label:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]


        
# batch_predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],'new_train_lrR_ET_TDvsPC_new_inceptiom_full_ResNet/model-epoch=01-val_loss=0.74-val_accuracy=0.91.ckpt')


# In[61]:


def train_model(D_name,Net_name):
    a_list = []
    for i in range(10):
        print("train number -", i)
        classifer = init_model(model_name=Net_name,lr_reduce=True, earlystopping=True, lr_patience=1, batch_size=32,lr=0.001, max_epochs=50,default_root_dir="./new_train_lrR_ET_"+D_name+"_"+Net_name+"/",nb_classes=2)
        classifer.fit(X_d[patients_idx_to_cycles_idx(x_train,cycle_end_idx)], y_d[patients_idx_to_cycles_idx(x_train,cycle_end_idx)], X_d[patients_idx_to_cycles_idx(x_val,cycle_end_idx)], y_d[patients_idx_to_cycles_idx(x_val,cycle_end_idx)],ckpt_monitor='val_accuracy')
#         a_list.append(accuracy_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)], classifer.predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)])))
        print("train number end -", i)

from sklearn.metrics import f1_score

def eval_model(D_name,Net_name):
    
    list_f1 = []
    list_accuracy = []

    import os
    filePath = "./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
    os.listdir(filePath)

    for i in os.listdir(filePath):
        if 'model' in i:
            print(i)
            classifer = init_model(model_name=Net_name).load( "./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+str(i),nb_classes=2)
            list_accuracy.append(accuracy_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)], classifer.predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)])))
            list_f1.append(f1_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)], classifer.predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)])))
    return list_accuracy, "%.4f" %(np.mean(list_accuracy)) +"±"+ "%.4f" %(np.std(list_accuracy)),list_f1 , "%.4f" %(np.mean(list_f1)) +"±"+ "%.4f" %(np.std(list_f1))

def batch_predict(images,model_path):
    pred_set = Dataset_torch(images,with_label=False)
    data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=64,num_workers=4)
    
    
    classifer = init_model(model_name=Net_name,nb_classes=2).load(model_path)
    
    trainer = pl.Trainer()
    pred = trainer.predict(model=classifer.model,dataloaders = data_loader_pred)
    pred = torch.cat(pred)
    return pred.detach().cpu().numpy()

from torch.utils.data import Dataset
from utils.utils import gait_dataloader
from utils.utils import init_model
## lime
import torch
import pytorch_lightning as pl
class Dataset_torch(Dataset):

    def __init__(self, data,with_label=True):
        self.with_label =  with_label
        
        if self.with_label:
            self.data_x, self.data_y = data
        else:
            self.data_x = data
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.with_label:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]

def eval_model_5_bundle(D_name,Net_name,idx=None):
    
    list_f1 = []
    list_accuracy = []

    import os
    filePath = "./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
    model_list = os.listdir(filePath)
    model_list.remove('logs')
    if 'prediction.npy' in model_list:
        model_list.remove('prediction.npy')
    proba_sum_list = []
    if idx==None:
        for index in range(10):
            print(index)
            proba_sum = np.zeros([len(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]),2])
            for j in range(5):

                print(model_list[index])
    #             classifer = init_model(model_name=Net_name).load( "./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+model_list[index*5+j])
                print(batch_predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],"./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+model_list[index*5+j]))
                proba_sum +=batch_predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],"./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+model_list[index*5+j])/5
            proba_sum_list.append(proba_sum)
    else:
            index = idx
            print(index)
            proba_sum = np.zeros([len(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]),3])
            for j in range(5):

                print(model_list[index])
    #             classifer = init_model(model_name=Net_name).load( "./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+model_list[index*5+j])
                print(batch_predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],"./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+model_list[index*5+j]))
                proba_sum +=batch_predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],"./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+model_list[index*5+j])/5
            proba_sum_list = (proba_sum)
    return proba_sum_list


import copy
X_d_original = X_d.copy()

# i_best = -1
best_list = []
list_angle = list(range(22))

#best_list = best_list+[i_best]
#for i in best_list:
    #list_angle.remove(i)

# In[74]:
for list_angle_i in range(len(list_angle)-1):
    
    if list_angle_i!=0:
        best_list = best_list+[i_best]
        list_angle.remove(i_best)
        
    
    if list_angle_i!=0:
        best_list_str = ''
        for i in best_list:
            best_list_str+=str(i)+"_"
    else:
        best_list_str =''


    for i_angle in list_angle:
        X_d_keep = X_d_original[:,best_list+[i_angle],:].copy()
        X_d = np.zeros_like(X_d)
        X_d[:,best_list+[i_angle],:] = X_d_keep

        D_name = "TDvsCPu_full_angle_"+best_list_str+str(i_angle)

        for i in ['ResNet']:
            print('Training angle number: '+best_list_str+str(i_angle))
            print(i)
#            for j in range(5):
#                train_model(D_name=D_name,Net_name=i)

#     print('Finished training angle: ',i_angle)

#        for i_net_name in ['ResNet']:
#            Net_name = i_net_name
#            out = eval_model_5_bundle(D_name,Net_name)
#            np.save('./new_train_lrR_ET_'+"TDvsCPu_full_angle_"+best_list_str+str(i_angle)+'_'+Net_name+'/prediction.npy' ,out)

    list_i_m_accuray_nets = []
    for i_angle in list_angle:

        X_d_keep = X_d_original[:,i_angle:i_angle+1,:].copy()
        X_d = np.zeros_like(X_d)
        X_d[:,i_angle:i_angle+1,:] = X_d_keep

        D_name = "TDvsCPu_full_angle_"+best_list_str+str(i_angle)

        list_accuracy_nets = []

        for i_net_name in ['ResNet']:
            Net_name = i_net_name
            out = np.load( './new_train_lrR_ET_'+D_name+'_'+Net_name+'/prediction.npy')
            results_cycles = print_result_bundle_i_cycles(out, y_d, x_test, cycle_end_idx,label_list=['TD', 'CPu'])
            list_accuracy_nets.append(np.array(results_cycles)[:,0])

        list_i_m_accuray_nets.append(np.mean(list_accuracy_nets,axis=0)[0])

    best_angle = np.argmax(list_i_m_accuray_nets)

    print(best_angle)
    i_best = np.array(list_angle)[np.argsort(-np.array(list_i_m_accuray_nets))][0]
    print(i_best)










