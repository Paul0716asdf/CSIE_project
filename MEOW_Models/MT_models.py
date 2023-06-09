import torch
from MEOW_Models.ST_model import Bert_classification, Bert_QA
from MEOW_Models.Kernel_model import BertWithoutEmbedding, ModelingQA, ModelingCLF
from MEOW_Utils.Data_utils import DataBox, layer_helper
from typing import*

class MEOW_MTM(torch.nn.Module):
    def __init__(
        self,
        Helper : layer_helper,
        qa_databox : DataBox,
        support_databox_list : List[DataBox] = None,
        do_mtl = True,
        device = None):

        super(MEOW_MTM, self).__init__()
                
        self.device = device
        self.kernel_model = Helper.get_kernel_model()
        self.support_data_num = len(support_databox_list)

        self.support_key = torch.nn.Linear(768, 768)
        self.target_query = torch.nn.Linear(768, 768) ## use the output of SQuAD's output

        #### initial all model
        #### ---------------------------------------------------------------------------------
        self.support_modulelist = torch.nn.ModuleList()
        self.optimizer_list = []     # optimizer_list[-1] is QA optimizer

        #### for support data ----------------------
        for i in range(self.support_data_num):
            self.support_modulelist.append( Bert_classification(self.kernel_model, 
                                                                 embedding_layer = Helper.get_embedding_layer(individual=False),
                                                                 modeling_layer = Helper.get_modelings_layer_clf(individual_pooler=True),
                                                                 num_labels = support_databox_list[i].label_nums,
                                                                 device = device) )
            self.optimizer_list.append(torch.optim.SGD(self.support_modulelist[i].parameters(), lr=0.00005, momentum=0.9, nesterov=True))
        
        
        #### ---------------------------------------
        #### ---------------------------------------
        
        #### for target data -----------------------
        self.SQuAD_model = Bert_QA(self.kernel_model,
                                    Helper.get_embedding_layer(individual=False), 
                                    Helper.get_modelings_layer_clf(individual_pooler=True),
                                    Helper.get_modelings_layer_qa(),
                                    num_labels = 2,
                                    support_modulelist = self.support_modulelist,
                                    support_key = self.support_key,
                                    target_query = self.target_query,
                                    do_mtl = do_mtl,
                                    device = device)
        self.SQuAD_optimizer = torch.optim.SGD(self.SQuAD_model.parameters(), lr=0.00005, momentum=0.9, nesterov=True)
        #### ---------------------------------------
        #### ---------------------------------------

    def mt_forward(self,
                   dataset_ind : int,
                   input_ids : torch.tensor, 
                   mask : torch.tensor, 
                   token_type_ids : torch.tensor, 
                   SEPind : List,
                   label : torch.tensor = None, # if inference, don't need it
                   start_pos : List = None,  #for qa
                   end_pos : List = None,  #for qa
                   eval : bool = False # for qa
                   ):
        
        if(dataset_ind < self.support_data_num): ## is clf task
            return self.support_modulelist[dataset_ind](input_ids, mask, token_type_ids, SEPind, label)
        return self.SQuAD_model(input_ids, mask, token_type_ids, SEPind, label, start_pos, end_pos, eval)

    def mt_optimize(self, loss, dataset_ind):
        if dataset_ind == self.support_data_num :
            optimizer = self.SQuAD_optimizer
        else :
            optimizer = self.optimizer_list[dataset_ind]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
