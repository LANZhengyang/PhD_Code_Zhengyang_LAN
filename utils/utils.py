import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

from classifiers import ResNet
from classifiers import LSTM
from classifiers import InceptionTime
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
import seaborn as sn

import pytorch_lightning as pl
from classifiers import ResNet
from classifiers import LSTM
from classifiers import InceptionTime

def load_dataset(dir_dataset,channel_first=True,flatten=False,shuffle=False,random_state=0):
    index = [0,1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,22,23,24,29]
    
    npzfile = np.load(dir_dataset + 'train.npz')
    
    x_train_all = npzfile['Input']
    y_train = npzfile['Output']
    y_train = np.ravel(y_train)
    x_train = x_train_all[:,:,index]
    if channel_first:
        x_train = x_train.transpose(0,2,1)
    
    npzfile = np.load(dir_dataset + 'test.npz')
    x_val_all = npzfile['Input']
    y_val = npzfile['Output']
    y_val = np.ravel(y_val)
    x_val = x_val_all[:,:,index]
    if channel_first:
        x_val = x_val.transpose(0,2,1)
    
    npzfile = np.load(dir_dataset + 'pred.npz')
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
    
    if shuffle == True:
        x_data = np.concatenate([x_train,x_val,x_test])
        y_data = np.concatenate([y_train,y_val,y_test])
        
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=random_state)
        x_test, x_val, y_test, y_val = train_test_split(x_data, y_data, test_size=0.25, random_state=random_state)
    
    return x_train, y_train ,x_val, y_val, x_test, y_test, nb_classes

class Dataset_torch(Dataset):

    def __init__(self, data):
        self.data_x, self.data_y = data
    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
    
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

def gait_dataloader(data_type,dataset,root_path=r'archives/AQM/',batch_size=64,flatten=False,shuffle=False,random_state=0):
    data_all = load_dataset(root_path+dataset+'/',flatten=flatten,shuffle=shuffle,random_state=random_state)
    nb_classes = data_all[6]
    
    if data_type == 'numpy':
        return data_all
    elif data_type == 'torch':
        train_set = Dataset_torch(data_all[0:2])
        test_set = Dataset_torch(data_all[2:4])
        pred_set = Dataset_torch(data_all[4:6])

        data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True,num_workers=4)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True,num_workers=4)
        data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=batch_size,num_workers=4)
        
        return data_loader_train, data_loader_test, data_loader_pred

def accuracy_end_to_end(model,x_train,y_train,x_test,y_test,x_val=None,y_val=None,batch_size=64,max_epochs=50,default_root_dir=None):
    if 'sklearn' in str(type(model)) or 'sktime' in str(type(model)) or 'tslearn' in str(type(model)): 
        return accuracy_score(y_test,model.fit(x_train,y_train).predict(x_test))
    else:
        x_val = x_test
        y_val = y_test        
        
        return accuracy_score(y_test,model.fit(x_train,y_train,x_val,y_val,default_root_dir=default_root_dir).predict(x_test))
    
    
def init_model(model_name=None,model=None,nb_classes=2 ,lr =0.001, lr_factor = 0.5, lr_patience=50,lr_reduce=True, batch_size=64, earlystopping=False, et_patience=10, max_epochs=50, default_root_dir=None):
    if model_name == None:
        return model
    elif model_name != None:
        if model_name == 'ResNet':
            return ResNet.Classifier_ResNet(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience,lr_reduce=lr_reduce, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, default_root_dir=default_root_dir)
        elif model_name == 'LSTM':
            return LSTM.Classifier_LSTM(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience,lr_reduce=lr_reduce, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, default_root_dir=default_root_dir)
        elif model_name == 'InceptionTime':
            return InceptionTime.Classifier_InceptionTime(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience,lr_reduce=lr_reduce, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, default_root_dir=default_root_dir)
        elif model_name == 'KNN_DTW':
            return KNN_DTW.Classifier_KNN_DTW()
        
import multiprocessing

def find_cycles_idx_by_patient_idx(patient_idx,cycle_end_idx):
    
    if patient_idx == 0:
        return np.arange(0,cycle_end_idx[patient_idx])
    else:
        return np.arange(cycle_end_idx[patient_idx-1],cycle_end_idx[patient_idx])
    
def patients_idx_to_cycles_idx(patients_idx,cycle_end_idx):
    cycles_idx = []
    for i in patients_idx:
        cycles_idx.append(find_cycles_idx_by_patient_idx(i,cycle_end_idx))
        
    return np.concatenate(cycles_idx)

def find_patients_idx_by_cycles_idx(cycles_idx,cycle_end_idx):
    patients_idx = []
    for i in cycles_idx:
#         print(cycle_idx_to_patient_idx(i,cycle_end_idx))
        patients_idx.append(cycle_idx_to_patient_idx(i,cycle_end_idx))
#         print(patients_idx)
    return patients_idx

def cycle_idx_to_patient_idx(cycle_idx,cycle_end_idx):
    return np.where(cycle_idx>cycle_end_idx)[0][-1]+1

class GridSearchCV:
    def __init__(self, model,para_list,n_splits=7,n_process=10,default_root_dir=None):
        self.model = model
        
        self.ML = 'sklearn' in str(model) or 'sktime' in str(model) or 'tslearn' in str(model)
        
        self.para_list = list(ParameterGrid(para_list))
        self.n_splits = n_splits
        self.n_process = n_process
        self.default_root_dir = default_root_dir
        
    def fit(self, x_data_cv, y_data_cv, X,y,cycle_end_idx):
#         if __name__ == "__main__":
        print(self.ML)
        if self.ML:
            
            pool = multiprocessing.Pool(processes=self.n_process)
            print('Multi-process:',self.n_process)

        kf = StratifiedKFold(n_splits=self.n_splits)

        self.accuracy_mean_list = []
        self.accuracy_std_list = []

        self.accuracy_list = []

        for para in self.para_list:

            accuracy = []
            for train_index, test_index in kf.split(x_data_cv,y_data_cv):
                
                x_tr = X[patients_idx_to_cycles_idx(x_data_cv[train_index],cycle_end_idx)]
               
                y_tr = y[patients_idx_to_cycles_idx(x_data_cv[train_index],cycle_end_idx)]
                
                x_te = X[patients_idx_to_cycles_idx(x_data_cv[test_index],cycle_end_idx)]
                y_te = y[patients_idx_to_cycles_idx(x_data_cv[test_index],cycle_end_idx)]
                                                                    
#                 print(x_te)
#                 print(self.model)
                if self.ML:
                    accuracy.append( pool.apply_async(accuracy_end_to_end,args=(self.model(**para),x_tr, y_tr,x_te,y_te)))
                elif not self.ML:   
                    accuracy.append( accuracy_end_to_end(self.model(**para),x_tr, y_tr,x_te,y_te,default_root_dir=self.default_root_dir))
            if self.ML:
                accuracy = [i.get() for i in accuracy]

            self.accuracy_list.append(accuracy)

            self.accuracy_mean_list.append(np.mean(accuracy))
            self.accuracy_std_list.append(np.std(accuracy))

            print('parameter:',para)
            print('accuracy:',np.mean(accuracy),"±",np.std(accuracy))
            print()
        if self.ML:
            pool.close() 
            pool.join()

        self.best_accuracy_mean = self.accuracy_mean_list[np.argmax(self.accuracy_mean_list)]
        self.best_accuracy_std = self.accuracy_std_list[np.argmax(self.accuracy_mean_list)]
        self.best_para = self.para_list[np.argmax(self.accuracy_mean_list)]
        print('------------------------')
        print('best parameter:',self.best_para)
        print('accuracy:',self.best_accuracy_mean,"±",self.best_accuracy_std)

        self.accuracy_list = np.array(self.accuracy_list)
        
# def error_patient_cycle(x_test,y_test,y_d,cycle_end_idx,pre_list,threshold=1):
#     cum_list = np.cumsum([len(find_cycles_idx_by_patient_idx(i,cycle_end_idx)) for i in x_test])
    
#     n_subject_l = 0
#     n_subject_r = 0
#     n_cycle = 0
#     n_patient_well_pre = 0
#     for i in range(len(x_test)):
#         print('-----------------------------')
#         print('Subject No.',i)
#         print('cycle_n=',len(find_cycles_idx_by_patient_idx(i,(cum_list))))
#         compre_list,compre_list_count = np.unique(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][find_cycles_idx_by_patient_idx(i,cum_list)].astype(int) == pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy(),return_counts=True)
        
#         n_error = 0
#         if False in compre_list:
#             n_error = compre_list_count[np.where(compre_list==False)[0][0]]
        
        
#         print('n_error=',n_error)
#         print()
        
#         n_subject_l_one = 0
#         n_subject_r_one = 0
        
#         if n_error:
#             if y_test[i] == 1:
#                 n_subject_r+=1
#                 n_subject_r_one+=n_error

#             elif y_test[i] == 0:
#                 n_subject_l+=1
#                 n_subject_l_one+=n_error

#             n_cycle += n_error
  
#         print('r',n_subject_r_one)  
#         print('l',n_subject_l_one)
#         print('pre_list',pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy())
#         print(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][find_cycles_idx_by_patient_idx(i,cum_list)].astype(int))
        
#         print('y_test[i]',y_test[i])
#         print('majr', np.argmax(np.bincount(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy())))
        
#         if y_test[i] == 1 and np.argmax(np.bincount(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy()))==1 and n_error!=0:
#             n_patient_well_pre+=1
            
# #             print(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][find_cycles_idx_by_patient_idx(i,cum_list)])
# #             print(np.unique(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy()))
#             print('print 1')
        
#         elif y_test[i] == 0 and np.argmax(np.bincount(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy()))==0 and n_error!=0:
#             n_patient_well_pre+=1
            
#             print('print 0')
            
#         elif y_test[i] == 2 and np.argmax(np.bincount(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy()))==2 and n_error!=0:
#             n_patient_well_pre+=1   
#             print('print 2')
            
#         elif n_error==0:
#             n_patient_well_pre+=1
            
#             print('print 0 err')

#     print('- Origin -')    
#     print('n_subject_1:',len(y_test[y_test==1]))
#     print('n_subject_0:',len(y_test[y_test==0]))
#     print('n_subject:',len(y_test))
#     print('n_patient_well_pre_majority:',n_patient_well_pre)
#     print('n_cycles:',len(patients_idx_to_cycles_idx(x_test,cycle_end_idx)))

#     print('- Error -')    
#     print('n_subject_1:',n_subject_r)
#     print('n_subject_0:',n_subject_l)
#     print('n_subject:',n_subject_r+n_subject_l)
#     print('n_cycles:',n_cycle)


def error_patient_cycle(x_test,y_test,y_d,cycle_end_idx,pre_list,threshold=1,nb_classes=2):
    
    
    error_order_idx = [[],[]]
    error_cycles_idx = []

    y_test = y_test.astype(int)

    cum_list = np.cumsum([len(find_cycles_idx_by_patient_idx(i,cycle_end_idx)) for i in x_test])
    
    
    n_subject = np.zeros(nb_classes,dtype=int)
    
    
    n_cycle = 0
    n_patient_well_pre = [0,0]
    
#     n_patient_well_pre_0 = 0
#     n_patient_well_pre_1 = 0
    
#     n_patient_bad_pre_0 = 0
#     n_patient_bad_pre_1 = 0
    
    for i in range(len(x_test)):
        
        if isinstance(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)],np.ndarray):
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].astype(int)
        else:
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy().astype(int)


        y_true = y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][find_cycles_idx_by_patient_idx(i,cum_list)].astype(int)
        
        print('-----------------------------')
        print('Subject No.',i)
        print('cycle_n=',len(find_cycles_idx_by_patient_idx(i,(cum_list))))
        compre_list,compre_list_count = np.unique(y_true ==y_pre,return_counts=True)
        
        n_error = 0
        if False in compre_list:
            n_error = compre_list_count[np.where(compre_list==False)[0][0]]
            
        print('n_error=',n_error)
        print()
        
        n_subject_one = np.zeros(nb_classes,dtype=int)
        
        
        if n_error:
            n_subject[int(y_test[i])]+=1
            n_subject_one[int(y_test[i])] += n_error

            n_cycle += n_error
        
        print('error in label '+str(y_test[i])+'=', n_subject_one[int(y_test[i])])    
        print('pre_list',y_pre)

        print('y_test[i]',y_test[i])
        print('majority:', np.argmax(np.bincount(y_pre)))
        
        if y_test[i] == np.argmax(np.bincount(y_pre)) and n_error!=0:
            n_patient_well_pre[y_test[i]]+=1
            
            print('well predict for class '+str(y_test[i]))
            
        elif n_error==0:
            n_patient_well_pre[y_test[i]]+=1
            
            print('well predict for no error')
        else:
            print('majority error for label No.',i)
            error_order_idx[y_test[i]].append(i)
            error_cycles_idx.append(np.where(y_pre!=y_test[i]))
            
        

    print('- Origin -')   
    for i in range(nb_classes):
        print('n_subject_'+str(i)+'=',len(y_test[y_test==i]))
    print('n_subject:',len(y_test))
    print('n_patient_well_pre_majority:',n_patient_well_pre)
    print('n_cycles:',len(patients_idx_to_cycles_idx(x_test,cycle_end_idx)))

    print('- Error -')
    for i in range(nb_classes):
        print('n_subject_'+str(i)+'=',n_subject[i])

    print('n_subject:',np.sum(n_subject))
    print('n_cycles:',n_cycle)
    print('error patient idx:', error_order_idx)
    print('error_cycles_idx:', error_cycles_idx)
    return error_order_idx, n_patient_well_pre

def prediction_subject(x_test,y_test,y_d,cycle_end_idx,pre_list):
    
    y_test = y_test.astype(int)
    cum_list = np.cumsum([len(find_cycles_idx_by_patient_idx(i,cycle_end_idx)) for i in x_test])
    pre_subject_list = []

    for i in range(len(x_test)):

        if isinstance(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)],np.ndarray):
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].astype(int)
        else:
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy().astype(int)


        pre_subject_list.append(np.argmax(np.bincount(y_pre)))

        
    return pre_subject_list
    
    
def print_result_bundle_i_cycles(out,y_d,x_test,cycle_end_idx,label_list,save=None,n_model=10,binary=True):
    ac_list = []
    f1_list = []
    recall_list = []

    recall_list = []

    specificity_list = []

    auc_list = []
    c_matrix_list = [] 

    for i in range(n_model):
        pre_lab_test = [ np.argmax(i) for i in out[i]]
        ac_list.append(accuracy_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))
        c_matrix_list.append(confusion_matrix(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))
        
        if binary==True:

            f1_list.append(f1_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))

            recall_list.append(recall_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))

            specificity_list.append(recall_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test, pos_label=0))

            auc_list.append(roc_auc_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))




    print("accuracy")
    print(ac_list)
    print(np.mean(ac_list))
    print(np.std(ac_list))
    
    print("confusion matrix")
    print(c_matrix_list)
    print(np.mean(c_matrix_list, axis=0))
    print(np.std(c_matrix_list, axis=0))
    
    if binary==True:

        print("F1")
        print((f1_list))
        print(np.mean(f1_list))
        print(np.std(f1_list))

        print("sensitivity")
        print(recall_list)
        print(np.mean(recall_list))
        print(np.std(recall_list))

        print("specificity")
        print(specificity_list)
        print(np.mean(specificity_list))
        print(np.std(specificity_list))


        print("AUC")
        print(auc_list)
        print(np.mean(auc_list))
        print(np.std(auc_list))


    
    

    df_cm = pd.DataFrame(np.mean(c_matrix_list, axis=0),index = label_list, columns = label_list )
    plt.figure(dpi=600)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}) # font size
    # plt.title("Confusion matrix ResNet", fontsize =20)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save!=None:
        plt.savefig(save+"_mean_cycles",bbox_inches = 'tight',facecolor ="w")
        
    
    plt.show()
    plt.close()

    df_cm = pd.DataFrame(np.std(c_matrix_list, axis=0),index = label_list, columns = label_list )
    plt.figure(dpi=600)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}) # font size
    # plt.title("Confusion matrix ResNet", fontsize =20)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save!=None:
        plt.savefig(save+"_std_cycles",bbox_inches = 'tight',facecolor ="w")
   
    plt.show()
    plt.close()
    
    if binary==True:
    
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(recall_list), np.std(recall_list)], [np.mean(specificity_list), np.std(specificity_list)], [np.mean(f1_list),np.std(f1_list)],[np.mean(auc_list),np.std(auc_list)]]
    else: 
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(f1_list),np.std(f1_list)]]

def print_result_bundle_i_subjects(out,y_d,x_test, y_test,cycle_end_idx,label_list,save=None ,n_model=10, binary=True):
    pre_subject_list = []
    for i in range(n_model):
        pre_subject = prediction_subject(x_test,y_test,y_d,cycle_end_idx,np.array([ np.argmax(j) for j in out[i]]))
        pre_subject_list.append(pre_subject)
        com = pre_subject == y_test
        print(np.where(com==False))
        
    
    ac_list = []
    f1_list = []
    recall_list = []

    recall_list = []

    specificity_list = []

    auc_list = []
    c_matrix_list = [] 

    for i in range(n_model):
        pre_lab_test = pre_subject_list[i]
        ac_list.append(accuracy_score(y_test,pre_lab_test))
        c_matrix_list.append(confusion_matrix(y_test,pre_lab_test))
        
        if binary==True:
            f1_list.append(f1_score(y_test,pre_lab_test))

            recall_list.append(recall_score(y_test,pre_lab_test))

            specificity_list.append(recall_score(y_test,pre_lab_test, pos_label=0))

            auc_list.append(roc_auc_score(y_test,pre_lab_test))

        
        
            


    print("accuracy")
    print(ac_list)
    print(np.mean(ac_list))
    print(np.std(ac_list))
    
    print("confusion matrix")
    print(c_matrix_list)
    print(np.mean(c_matrix_list, axis=0))
    print(np.std(c_matrix_list, axis=0))
    
    if binary==True:

        print("F1")
        print((f1_list))
        print(np.mean(f1_list))
        print(np.std(f1_list))

        print("sensitivity")
        print(recall_list)
        print(np.mean(recall_list))
        print(np.std(recall_list))

        print("specificity")
        print(specificity_list)
        print(np.mean(specificity_list))
        print(np.std(specificity_list))


        print("AUC")
        print(auc_list)
        print(np.mean(auc_list))
        print(np.std(auc_list))


        
    df_cm = pd.DataFrame(np.mean(c_matrix_list, axis=0),index = label_list, columns = label_list )
    plt.figure(dpi=600)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}) # font size
    # plt.title("Confusion matrix ResNet", fontsize =20)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    if save!=None:
        plt.savefig(save+"_mean_subjects",bbox_inches = 'tight',facecolor ="w")
        
    plt.show()
    plt.close()
    
    df_cm = pd.DataFrame(np.std(c_matrix_list, axis=0),index = label_list, columns = label_list )
    plt.figure(dpi=600)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}) # font size
    # plt.title("Confusion matrix ResNet", fontsize =20)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    if save!=None:
        plt.savefig(save+"_std_subjects",bbox_inches = 'tight',facecolor ="w")
        
        
    plt.show()
    plt.close()
    
    if binary==True:
    
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(recall_list), np.std(recall_list)], [np.mean(specificity_list), np.std(specificity_list)], [np.mean(f1_list),np.std(f1_list)],[np.mean(auc_list),np.std(auc_list)]]
    else: 
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(f1_list),np.std(f1_list)]]
    
def merge_mean_std(mean_std):
    return str('%.2f'%mean_std[0])+"±"+str('%.2f'%mean_std[1])

def output_csv(output_list):
    list_str = []
    for i in output_list:
        list_str.append(merge_mean_std(i))
    return list_str

def load_model_5_bundle(D_name,Net_name,index=0):
    
    
    class model_5_bundle:
        def __init__(self, D_name,Net_name):
            self.D_name = D_name
            self.Net_name = Net_name
            self.index = index
            
            self.filePath = "./new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
            self.model_list_brut = os.listdir(self.filePath)
            self.model_list = []
            for idx_tmp,i in enumerate(self.model_list_brut):
                if i.startswith('model'):
                    self.model_list.append(i)
#                     print(i)
#             print(len(self.model_list))
                    
            
        def predict(self,X,nb_classes=2):
            
            proba_sum = np.zeros([len(X),nb_classes])
#             print(self.model_list[])
#             print('-------------------------------')
            for j in range(5):
#                 print(self.model_list[self.index*5+j])
                proba_sum +=batch_predict(D_name,Net_name, X,"./new_train_lrR_ET_"+self.D_name+"_"+self.Net_name+"/"+self.model_list[self.index*5+j],nb_classes=nb_classes)/5
            
            return np.array(proba_sum)
#             return np.sum(np.array(proba_sum),axis=1)
    return model_5_bundle(D_name,Net_name)

def batch_predict(D_name,Net_name, images,model_path,nb_classes=2):
    pred_set = Dataset_torch(images,with_label=False)
    data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=64,num_workers=4)
    
    
    classifer = init_model(model_name=Net_name,nb_classes=nb_classes).load(model_path, nb_classes=nb_classes)
    
    trainer = pl.Trainer()
    pred = trainer.predict(model=classifer.model,dataloaders = data_loader_pred)
    pred = torch.cat(pred)
    return pred.detach().cpu().numpy()
