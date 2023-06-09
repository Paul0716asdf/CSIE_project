import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 

import torch 
import transformers 
from transformers import BertTokenizer 
from typing import * 

from MEOW_Models.MT_models import MEOW_MTM 
from MEOW_Models.Kernel_model import BertWithoutEmbedding 

from MEOW_Utils.Data_utils import* 
from MEOW_Utils.Training_utils import* 
from MEOW_Utils.config import * 

from torch.utils.data import DataLoader

def SCRIPT_SET_TOKENIZER():
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODULE_NAME)
    print("SET TOKENIZER successfully")
    return tokenizer

def SCRIPT_SET_QAandSUP(check_data_loader = False, path = ''):

    df_CoLA = pd.read_csv(path + INPUT_FILE_PATH_CoLA, index_col=[0])
    df_MNLI = pd.read_csv(path + INPUT_FILE_PATH_MNLI, index_col=[0])
    df_SQuAD = pd.read_csv(path + INPUT_FILE_PATH_SQuAD, index_col=[0])
    df_QNLI = pd.read_csv(path + INPUT_FILE_PATH_QNLI, index_col=[0])
    df_RTE = pd.read_csv(path + INPUT_FILE_PATH_RTE, index_col=[0]) 

    df_SQuAD_HA = df_SQuAD[df_SQuAD.answer_start != -1]
    df_SQuAD_NA = df_SQuAD[df_SQuAD.answer_start == -1]
    
    SQuAD_NA_databox = DataBox(
                 dataset_name = 'SQuAD',
                 df_Data = df_SQuAD_NA,
                 test_size = TEST_SIZE,
                 tokenizer = tokenizer,
                 label_nums = 2,
                 batch_size = Target_BATCH_SIZE,
                 )

    SQuAD_HA_databox = DataBox(
                    dataset_name = 'SQuAD',
                    df_Data = df_SQuAD_HA,
                    test_size = TEST_SIZE,
                    tokenizer = tokenizer,
                    label_nums = 2,
                    batch_size = Target_BATCH_SIZE,
                    )

    CoLA_databox = DataBox(
                dataset_name = 'CoLA',
                df_Data = df_CoLA,
                test_size = TEST_SIZE,
                tokenizer = tokenizer,
                label_nums = 2,
                batch_size = Support_BATCH_SIZE
                )

    MNLI_databox = DataBox(
                dataset_name = 'MNLI',
                df_Data = df_MNLI,
                test_size = TEST_SIZE,
                tokenizer = tokenizer,
                label_nums = 3,
                batch_size = Support_BATCH_SIZE
                )

    QNLI_databox = DataBox(
                dataset_name = 'QNLI',
                df_Data = df_QNLI,
                test_size = TEST_SIZE,
                tokenizer = tokenizer,
                label_nums = 2,
                batch_size = Support_BATCH_SIZE
                )
    
    RTE_databox = DataBox(
                dataset_name = 'RTE',
                df_Data = df_RTE,
                test_size = TEST_SIZE,
                tokenizer = tokenizer,
                label_nums = 2,
                batch_size = Support_BATCH_SIZE
                )

    #### check no wrong in dataloader ####
    if check_data_loader : 
        for i in SQuAD_NA_databox.test_dataloader:
            0
        for i in SQuAD_HA_databox.test_dataloader:
            0
        for i in MNLI_databox.test_dataloader:
            0
        for i in CoLA_databox.test_dataloader:
            0
        for i in QNLI_databox.test_dataloader:
            0
        for i in RTE_databox.test_dataloader:
            0
    ################

    global Training_round

    Training_round = min(len(SQuAD_HA_databox.training_dataloader),
                          len(SQuAD_NA_databox.training_dataloader),
                          len(MNLI_databox.training_dataloader),
                          len(CoLA_databox.training_dataloader),
                          len(QNLI_databox.training_dataloader),
                          len(RTE_databox.training_dataloader))

    global Test_round

    Test_round = min(len(SQuAD_HA_databox.test_dataloader),
                 len(SQuAD_NA_databox.test_dataloader),
                 len(MNLI_databox.test_dataloader),
                 len(CoLA_databox.test_dataloader),
                 len(QNLI_databox.test_dataloader),
                 len(RTE_databox.test_dataloader))

    global SQuAD_list
    SQuAD_list = [SQuAD_HA_databox, SQuAD_NA_databox]

    global SUP_list
    SUP_list = [CoLA_databox, MNLI_databox, QNLI_databox, RTE_databox]

    print("SET DATASET successfully")

def SCRIP_GET_TRAINTEST_ROUND():
    print(f'Per epoc round\'s num is {Training_round}')
    print(f'Per epoc round\'s num is {Test_round}')

def SCRIPT_SET_MODEL(do_mtl = True, path = None, qa_optim_path = None):
    global Helper
    Helper = layer_helper(pretrained_module_name=PRETRAINED_MODULE_NAME, device=DEVICE)

    global MEOW_model
    MEOW_model = MEOW_MTM(Helper,
                      qa_databox = SQuAD_list[0],  #pass HA or NA is same
                      support_databox_list = SUP_list,
                      do_mtl=do_mtl,
                      device=DEVICE
                      )
    MEOW_model.to(DEVICE)

    if path != None:
        MEOW_model.load_state_dict(torch.load(path))
        MEOW_model.eval()
        print("load the model successfully")
    else:
        print("initial the model successfully") 
    
    if qa_optim_path:
        MEOW_model.SQuAD_optimizer.load_state_dict(torch.load(qa_optim_path)) 
        print("load the optimizer successfully") 

    return MEOW_model

def SCRIPT_THE_UNANSWER_FAULT():
    #### This input the pair that has no anwer
    #### and output the pair that the model predict this pair to has answer
    #### mean print the model false prediction
    
    df_na = SQuAD_list[1].df_test
    ind = []

    for i in range(len(df_na)):
        
        context = df_na.iloc[i]['context']
        question = df_na.iloc[i]['question']
        EC = tokenizer.encode_plus(context, question)
        
        SEPind = [len(tokenizer.tokenize(context)) + 1]

        input_ids = torch.tensor([EC['input_ids']])  # 要讓他升一個維度 表示batch
        mask = torch.tensor([EC['attention_mask']])
        token = torch.tensor([EC['token_type_ids']])

        input_ids = input_ids.to(DEVICE)
        mask = mask.to(DEVICE)
        token = token.to(DEVICE)

        toks, prob = MEOW_model.mt_forward(dataset_ind=DATA_IND['SQuAD'],
                                            input_ids=input_ids,
                                            mask=mask,
                                            token_type_ids=token,
                                            SEPind=SEPind,
                                            eval=True)
        
        hasanswer = torch.argmax(prob[0], dim=0) == 1

        # pred_toks = tokenizer.convert_ids_to_tokens(toks[0])
        if hasanswer == True:
            ind.append(i)

    return ind

def SCRIPT_THE_ANSWER_FAULT():
    #### This input the pair that has anwer
    #### and output the pair that the model predict this pair to has no answer
    #### mean print the model negative true prediction
    
    df_ha = SQuAD_list[0].df_test
    ind = []

    for i in range(len(df_ha)):
        
        context = df_ha.iloc[i]['context']
        question = df_ha.iloc[i]['question']
        EC = tokenizer.encode_plus(context, question)
        
        SEPind = [len(tokenizer.tokenize(context)) + 1]

        input_ids = torch.tensor([EC['input_ids']])  # 要讓他升一個維度 表示batch
        mask = torch.tensor([EC['attention_mask']])
        token = torch.tensor([EC['token_type_ids']])

        input_ids = input_ids.to(DEVICE)
        mask = mask.to(DEVICE)
        token = token.to(DEVICE)

        toks, prob = MEOW_model.mt_forward(dataset_ind=DATA_IND['SQuAD'],
                                            input_ids=input_ids,
                                            mask=mask,
                                            token_type_ids=token,
                                            SEPind=SEPind,
                                            eval=True)
        
        hasanswer = ( torch.argmax(prob[0], dim=0) == 1 )

        if i == 4 :
            print(hasanswer)

        # pred_toks = tokenizer.convert_ids_to_tokens(toks[0])
        if hasanswer == False:
            ind.append(i)

    return ind

def SCRIPT_EVALUATE_MODEL():
    df_HA = SQuAD_list[0].df_test
    dset_HA = QA_evalaute_dataset(df_HA, tokenizer, 2)
    dloader_HA = DataLoader(dset_HA, batch_size=8, shuffle=False, collate_fn=QA_evaluate_collate_batch)

    df_NA = SQuAD_list[1].df_test
    dset_NA = QA_evalaute_dataset(df_NA, tokenizer, 2)
    dloader_NA = DataLoader(dset_NA, batch_size=8, shuffle=False, collate_fn=QA_evaluate_collate_batch)

    HA_total_correct = 0
    HA_score = 0.0
    NA_total_correct = 0
    HA_score = 0.0

    NA_total_correct = 0
    NA_score = 0.0
    NA_total_correct = 0
    NA_score = 0.0

    for d in dloader_HA:
        input_ids, mask, token, label, SEPind, i_batch = d
        
        orgdevice = input_ids.device

        input_ids = input_ids.to(DEVICE)
        mask = mask.to(DEVICE)
        token = token.to(DEVICE)
        label = label.to(DEVICE)
        
        toks, prob = MEOW_model.mt_forward(dataset_ind = DATA_IND['SQuAD'],
                                            input_ids = input_ids, 
                                            mask = mask, 
                                            token_type_ids = token,
                                            SEPind = SEPind,
                                            eval = True)
        
        correct_num = count_correct_num(prob, label)
        HA_total_correct += correct_num

        # to provent the cuda from out of memory
        # use to orgdevice to releace the memory allocated to tensor
        input_ids = input_ids.to(orgdevice)
        mask = mask.to(orgdevice)
        token = token.to(orgdevice)
        label = label.to(orgdevice)

        for k in range(len(input_ids)):
            ans_toks = tokenizer.tokenize(df_HA['text'][i_batch[k]])
            # print(ans_toks)
            pred_toks = tokenizer.convert_ids_to_tokens(toks[k])
            # print(pred_toks)
            # print('')
            HA_score += compute_f1(ans_toks, pred_toks)

    for d in dloader_NA:
        input_ids, mask, token, label, SEPind, i_batch = d
        
        orgdevice = input_ids.device

        input_ids = input_ids.to(DEVICE)
        mask = mask.to(DEVICE)
        token = token.to(DEVICE)
        label = label.to(DEVICE)
        
        toks, prob = MEOW_model.mt_forward(dataset_ind = DATA_IND['SQuAD'],
                                            input_ids = input_ids, 
                                            mask = mask, 
                                            token_type_ids = token,
                                            SEPind = SEPind,
                                            eval = True)
        
        correct_num = count_correct_num(prob, label)
        NA_total_correct += correct_num

        # to provent the cuda from out of memory
        # use to orgdevice to releace the memory allocated to tensor
        input_ids = input_ids.to(orgdevice)
        mask = mask.to(orgdevice)
        token = token.to(orgdevice)
        label = label.to(orgdevice)

        for k in range(len(input_ids)):
            ans_toks = tokenizer.tokenize(df_NA['text'][i_batch[k]])
            # print(ans_toks)
            pred_toks = tokenizer.convert_ids_to_tokens(toks[k])
            # print(pred_toks)
            # print('')
            NA_score += compute_f1(ans_toks, pred_toks)
    
    print("HAS ANSWER :")
    print("data num : {:d}", len(df_HA))
    print("f1 score : {:.4f}", HA_score / len(df_HA))
    print("Precision : {:d}", HA_total_correct / len(df_HA))
    print("Recall : {:d}", HA_total_correct / (HA_total_correct + len(df_NA) - NA_total_correct))

    print("")

    print("NO ANSWER :")

    print("data num : {:d}", len(df_NA))
    print("f1 score : {:.4f}", NA_score / len(df_NA))
    print("Precision : {:d}", NA_total_correct / len(df_NA))
    print("Recall : {:d}", NA_total_correct / (NA_total_correct + len(df_HA) - HA_total_correct))

    print("")

def SCRIPT_TRAIN_SUPPORT(epoch_num):
    # 訓練
    for epoch in range(epoch_num):
        print("the {:d} iter :".format(epoch+1))

        Sup_record = [Record() for _ in range(SUPPORT_DATA_NUM)]

        #### training now 
        ################################################################################################################
        MEOW_model.train()

        support_data_training_iter = [iter(db.training_dataloader) for db in SUP_list]

        for _ in range(Training_round):
            #### support data train -------------------------------------
            for i in range(SUPPORT_DATA_NUM):
                loss, prob, correct = Classifiaction_running(MEOW_model, support_data_training_iter[i], DEVICE, i, do_optimize=True)
                Sup_record[i].add_training_acur(correct, prob.size(0))
                Sup_record[i].add_training_loss(loss.item())
            #### --------------------------------------------------------
        
        #### record ############################
        for i in range(SUPPORT_DATA_NUM):
            x = Sup_record[i].get_training_average_loss()
            y = Sup_record[i].get_training_accuracy()
            SUP_list[i].H['train_loss'].append(x)
            SUP_list[i].H['train_acc'].append(y)
            print("{} train loss: {:.6f}, correct_rate: {:.4f}".format(DATA_NAME[i], x, y))
        ########################################

        ################################################################################################################


        #### test now
        ################################################################################################################
        MEOW_model.eval()

        support_data_test_iter = [iter(db.test_dataloader) for db in SUP_list]

        for _ in range(Test_round):  
            # Sup data 
            for i in range(SUPPORT_DATA_NUM):
                loss, prob, correct = Classifiaction_running(MEOW_model, support_data_test_iter[i], DEVICE, i, do_optimize=False)
                Sup_record[i].add_test_acur(correct, prob.size(0))
                Sup_record[i].add_test_loss(loss.item())
            
        #### record ################################
        for i in range(SUPPORT_DATA_NUM):
            x = Sup_record[i].get_test_average_loss()
            y = Sup_record[i].get_test_accuracy()
            SUP_list[i].H['test_loss'].append(x)
            SUP_list[i].H['test_acc'].append(y)
            print("{} test loss: {:.6f}, correct_rate: {:.4f}".format(DATA_NAME[i], x, y))
        #############################################
        print(" ")
        ################################################################################################################

def SCRIPT_TRAIN_QA(epoch_num):
    # 訓練
    for epoch in range(epoch_num):
        print("the {:d} iter :".format(epoch+1))

        HA_record = Record()
        NA_record = Record() 

        #### training now 
        ################################################################################################################
        MEOW_model.train()

        SQuAD_HA_training_iter = iter(SQuAD_list[0].training_dataloader)
        SQuAD_NA_training_iter = iter(SQuAD_list[1].training_dataloader)

        for _ in range(Training_round):
            # SQuAD has_answer
            loss, prob, correct = QA_running(MEOW_model, SQuAD_HA_training_iter, DEVICE, dataset_ind = DATA_IND['SQuAD'], do_optimize=True)
            HA_record.add_training_acur(correct, prob.size(0))
            HA_record.add_training_loss(loss.item())

            # SQuAD no_answer
            loss, prob, correct = QA_running(MEOW_model, SQuAD_NA_training_iter, DEVICE, dataset_ind = DATA_IND['SQuAD'], do_optimize=True)
            NA_record.add_training_acur(correct, prob.size(0))
            NA_record.add_training_loss(loss.item())
        
        #### record ############################
        SQuAD_list[0].H['train_loss'].append(HA_record.get_training_average_loss())
        SQuAD_list[0].H['train_acc'].append(HA_record.get_training_accuracy())
        print("SQuAD has answer train loss: {:.6f}, correct_rate: {:.4f}".format(HA_record.get_training_average_loss(), HA_record.get_training_accuracy()))

        SQuAD_list[1].H['train_loss'].append(NA_record.get_training_average_loss()) 
        SQuAD_list[1].H['train_acc'].append(NA_record.get_training_accuracy())
        print("SQuAD no answer train loss: {:.6f}, correct_rate: {:.4f}".format(NA_record.get_training_average_loss(), NA_record.get_training_accuracy()))
        ########################################

        ################################################################################################################


        #### test now
        ################################################################################################################
        MEOW_model.eval()

        SQuAD_HA_test_iter = iter(SQuAD_list[0].test_dataloader)
        SQuAD_NA_test_iter = iter(SQuAD_list[1].test_dataloader)

        for _ in range(Test_round):
            # SQuAD has_answer
            loss, prob, correct = QA_running(MEOW_model, SQuAD_HA_test_iter, DEVICE, dataset_ind = DATA_IND['SQuAD'], do_optimize=False)
            HA_record.add_test_acur(correct, prob.size(0))
            HA_record.add_test_loss(loss.item())

            # SQuAD no_answer
            loss, prob, correct = QA_running(MEOW_model, SQuAD_NA_test_iter, DEVICE, dataset_ind = DATA_IND['SQuAD'], do_optimize=False)
            NA_record.add_test_acur(correct, prob.size(0))
            NA_record.add_test_loss(loss.item())
            
        #### record ################################
        SQuAD_list[0].H['test_loss'].append(HA_record.get_test_average_loss())
        SQuAD_list[0].H['test_acc'].append(HA_record.get_test_accuracy())
        print("SQuAD has answer test loss: {:.6f}, correct_rate: {:.4f}".format(HA_record.get_test_average_loss(), HA_record.get_test_accuracy()))

        SQuAD_list[1].H['test_loss'].append(NA_record.get_test_average_loss()) 
        SQuAD_list[1].H['test_acc'].append(NA_record.get_test_accuracy())
        print("SQuAD no answer test loss: {:.6f}, correct_rate: {:.4f}".format(NA_record.get_test_average_loss(), NA_record.get_test_accuracy()))
        #############################################
        print(" ")
        
        ################################################################################################################

def SCRIPT_TRAIN_BOTH_SIMU(epoch_num):
    # 訓練
    for epoch in range(epoch_num):
        print("the {:d} iter :".format(epoch+1))

        HA_record = Record()
        NA_record = Record()
        Sup_record = [Record() for _ in range(SUPPORT_DATA_NUM)]

        #### training now 
        ################################################################################################################
        MEOW_model.train()

        support_data_training_iter = [iter(db.training_dataloader) for db in SUP_list]
        SQuAD_HA_training_iter = iter(SQuAD_list[0].training_dataloader)
        SQuAD_NA_training_iter = iter(SQuAD_list[1].training_dataloader)

        for _ in range(Training_round):
            # SQuAD has_answer
            loss, prob, correct = QA_running(MEOW_model, SQuAD_HA_training_iter, DEVICE, dataset_ind = DATA_IND['SQuAD'], do_optimize=True)
            HA_record.add_training_acur(correct, prob.size(0))
            HA_record.add_training_loss(loss.item())

            # SQuAD no_answer
            loss, prob, correct = QA_running(MEOW_model, SQuAD_NA_training_iter, DEVICE, dataset_ind = DATA_IND['SQuAD'], do_optimize=True)
            NA_record.add_training_acur(correct, prob.size(0))
            NA_record.add_training_loss(loss.item())

            #### support data train -------------------------------------
            for i in range(SUPPORT_DATA_NUM):
                loss, prob, correct = Classifiaction_running(MEOW_model, support_data_training_iter[i], DEVICE, i, do_optimize=True)
                Sup_record[i].add_training_acur(correct, prob.size(0))
                Sup_record[i].add_training_loss(loss.item())
            #### --------------------------------------------------------
        
        #### record ############################
        SQuAD_list[0].H['train_loss'].append(HA_record.get_training_average_loss())
        SQuAD_list[0].H['train_acc'].append(HA_record.get_training_accuracy())
        print("SQuAD has answer train loss: {:.6f}, correct_rate: {:.4f}".format(HA_record.get_training_average_loss(), HA_record.get_training_accuracy()))

        SQuAD_list[1].H['train_loss'].append(NA_record.get_training_average_loss()) 
        SQuAD_list[1].H['train_acc'].append(NA_record.get_training_accuracy())
        print("SQuAD no answer train loss: {:.6f}, correct_rate: {:.4f}".format(NA_record.get_training_average_loss(), NA_record.get_training_accuracy()))
        
        for i in range(SUPPORT_DATA_NUM):
            x = Sup_record[i].get_training_average_loss()
            y = Sup_record[i].get_training_accuracy()
            SUP_list[i].H['train_loss'].append(x)
            SUP_list[i].H['train_acc'].append(y)
            print("{} train loss: {:.6f}, correct_rate: {:.4f}".format(DATA_NAME[i], x, y))
        ########################################

        ################################################################################################################


        #### test now
        ################################################################################################################
        MEOW_model.eval()

        support_data_test_iter = [iter(db.test_dataloader) for db in SUP_list]
        SQuAD_HA_test_iter = iter(SQuAD_list[0].test_dataloader)
        SQuAD_NA_test_iter = iter(SQuAD_list[1].test_dataloader)

        for _ in range(Test_round):
            # SQuAD has_answer
            loss, prob, correct = QA_running(MEOW_model, SQuAD_HA_test_iter, DEVICE, dataset_ind = DATA_IND['SQuAD'], do_optimize=False)
            HA_record.add_test_acur(correct, prob.size(0))
            HA_record.add_test_loss(loss.item())

            # SQuAD no_answer
            loss, prob, correct = QA_running(MEOW_model, SQuAD_NA_test_iter, DEVICE, dataset_ind = DATA_IND['SQuAD'], do_optimize=False)
            NA_record.add_test_acur(correct, prob.size(0))
            NA_record.add_test_loss(loss.item())
            
            # Sup data 
            for i in range(SUPPORT_DATA_NUM):
                loss, prob, correct = Classifiaction_running(MEOW_model, support_data_test_iter[i], DEVICE, i, do_optimize=False)
                Sup_record[i].add_test_acur(correct, prob.size(0))
                Sup_record[i].add_test_loss(loss.item())
            
        #### record ################################
        SQuAD_list[0].H['test_loss'].append(HA_record.get_test_average_loss())
        SQuAD_list[0].H['test_acc'].append(HA_record.get_test_accuracy())
        print("SQuAD has answer test loss: {:.6f}, correct_rate: {:.4f}".format(HA_record.get_test_average_loss(), HA_record.get_test_accuracy()))

        SQuAD_list[1].H['test_loss'].append(NA_record.get_test_average_loss()) 
        SQuAD_list[1].H['test_acc'].append(NA_record.get_test_accuracy())
        print("SQuAD no answer test loss: {:.6f}, correct_rate: {:.4f}".format(NA_record.get_test_average_loss(), NA_record.get_test_accuracy()))
        
        for i in range(SUPPORT_DATA_NUM):
            x = Sup_record[i].get_test_average_loss()
            y = Sup_record[i].get_test_accuracy()
            SUP_list[i].H['test_loss'].append(x)
            SUP_list[i].H['test_acc'].append(y)
            print("{} test loss: {:.6f}, correct_rate: {:.4f}".format(DATA_NAME[i], x, y))
        #############################################
        print(" ")
        ################################################################################################################

def SCRIPT_ACK_QUESTION(context, question):
    EC = tokenizer.encode_plus(context, question)
    
    SEPind = [len(tokenizer.tokenize(context)) + 1]

    input_ids = torch.tensor([EC['input_ids']])  # 要讓他升一個維度 表示batch
    mask = torch.tensor([EC['attention_mask']])
    token = torch.tensor([EC['token_type_ids']])

    input_ids = input_ids.to(DEVICE)
    mask = mask.to(DEVICE)
    token = token.to(DEVICE)

    toks, prob = MEOW_model.mt_forward(dataset_ind=DATA_IND['SQuAD'],
                                        input_ids=input_ids,
                                        mask=mask,
                                        token_type_ids=token,
                                        SEPind=SEPind,
                                        eval=True)
    
    print("has answer probabiliy : {:4f}".format(prob[0][1]))
    hasanswer = torch.argmax(prob[0], dim=0) == 1

    # pred_toks = tokenizer.convert_ids_to_tokens(toks[0])
    if hasanswer == True:
        print(tokenizer.decode(toks[0]))
    else:
        print("NO ANSWER")
    print('')

def SCRIPT_GET_DFHA_TEST():
    return SQuAD_list[0].df_test
    
def SCRIPT_GET_DFNA_TEST():
    return SQuAD_list[1].df_test
