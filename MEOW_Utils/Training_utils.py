import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from MEOW_Models.MT_models import MEOW_MTM
from pandas import DataFrame
from transformers import BertTokenizer
from MEOW_Utils.Data_utils import count_the_TKbeg_and_TKend
import collections 
from MEOW_Utils.config import*

def plot_diagram(H, epoch_num):
    # tensor to float
    Train_loss = [float(i) for i in H['train_loss']]
    Test_loss = [float(i) for i in H['test_loss']]

    Train_acur = [float(i) for i in H['train_acc']]
    Test_acur = [float(i) for i in H['test_acc']]

    # loss
    plt.figure()
    plt.title("Loss")
    plt.xlabel("EPOCH")
    plt.ylabel("Loss")
    plt.plot(Train_loss, label="train_loss")
    plt.plot(Test_loss, label="test_loss")
    plt.xticks(np.arange(epoch_num), range(1,epoch_num+1,1))
    plt.show()
    
    # accuracy
    plt.figure()
    plt.title("Test Accuracy")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.plot(Train_acur, label="train_acc")
    plt.plot(Test_acur, label="test_acc")
    plt.xticks(np.arange(epoch_num), range(1,epoch_num+1,1))
    plt.show()

def count_correct_num(prob : torch.tensor, label : torch.tensor):
    predict = torch.argmax(prob, dim=1)
    label = torch.argmax(label, dim=1)
    correct_num = (predict == label).type(torch.int).sum()
    return correct_num

def QA_running(MEOW_model : MEOW_MTM, 
                iter,
                device,
                dataset_ind,
                do_optimize = False,
                return_toks = False
                ):
        input_ids, mask, token, label, SEPind, Start_pos, End_pos = next(iter)

        orgdevice = input_ids.device

        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        label = label.to(device)
        
        loss, prob = MEOW_model.mt_forward(dataset_ind = dataset_ind,
                                            input_ids = input_ids, 
                                            mask = mask, 
                                            token_type_ids = token,
                                            SEPind = SEPind,
                                            label=label,
                                            start_pos = Start_pos,
                                            end_pos = End_pos,
                                            eval = return_toks)
        
        correct_num = count_correct_num(prob, label)
        if do_optimize:
            MEOW_model.mt_optimize(loss=loss, dataset_ind=dataset_ind)

        # to provent the cuda from out of memory
        # use to orgdevice to releace the memory allocated to tensor
        input_ids = input_ids.to(orgdevice)
        mask = mask.to(orgdevice)
        token = token.to(orgdevice)
        label = label.to(orgdevice)
        
        return loss, prob, correct_num

def Classifiaction_running(MEOW_model : MEOW_MTM, 
                            iter,
                            device, 
                            dataset_ind,
                            do_optimize = False
                            ):
        input_ids, mask, token, label, SEPind = next(iter)

        orgdevice = input_ids.device
        
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        label = label.to(device)
        
        loss, prob = MEOW_model.mt_forward(dataset_ind = dataset_ind,
                                           input_ids = input_ids,
                                           mask = mask,
                                           token_type_ids = token,
                                           SEPind = SEPind,
                                           label = label
                                           )
        
        correct = count_correct_num(prob, label)
        if do_optimize:
            MEOW_model.mt_optimize(loss = loss, dataset_ind = dataset_ind)


        # to provent the cuda from out of memory
        # use to orgdevice to releace the memory allocated to tensor
        input_ids = input_ids.to(orgdevice)
        mask = mask.to(orgdevice)
        token = token.to(orgdevice)
        label = label.to(orgdevice)
        
        return loss, prob, correct

def compute_f1(targ_toks : list, pred_toks : list):    
    common = collections.Counter(targ_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(targ_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(targ_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(targ_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

class Record:
    def __init__(self):
        self.training_loss = 0.0
        self.training_round_num = 0
        self.training_correct = 0
        self.training_data_num = 0
        
        self.test_loss = 0.0
        self.test_round_num = 0
        self.test_correct = 0
        self.test_data_num = 0
    
    def get_training_accuracy(self):
        return self.training_correct / self.training_data_num

    def get_training_average_loss(self):
        return self.training_loss / self.training_round_num

    def get_test_accuracy(self):
        return self.test_correct / self.test_data_num
    
    def get_test_average_loss(self):
        return self.test_loss / self.test_round_num
    
    def add_training_loss(self, loss):
        self.training_loss += loss
        self.training_round_num += 1
    
    def add_training_acur(self, correct, num):
        self.training_correct += correct
        self.training_data_num += num

    def add_test_loss(self, loss):
        self.test_loss += loss
        self.test_round_num += 1
    
    def add_test_acur(self, correct, num):
        self.test_correct += correct
        self.test_data_num += num
