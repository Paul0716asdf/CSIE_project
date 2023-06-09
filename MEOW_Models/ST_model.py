import torch
from MEOW_Models.Kernel_model import BertWithoutEmbedding, ModelingQA, ModelingCLF
from typing import*

from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler
from MEOW_Utils.model_utils import get_retrieve_context_matrix, pading_empty_tensor, Highway_layer, CLS_pooler_layer
from torch.nn.parameter import Parameter
import random
import math

class Bert_classification(torch.nn.Module):
    def __init__(
        self,
        kernel_model : BertWithoutEmbedding,
        embedding_layer : BertEmbeddings,
        modeling_layer : ModelingCLF,
        num_labels : int,
        device
        ):
        super(Bert_classification, self).__init__()
        
        self.kernel_model = kernel_model
        self.embedding_layer = embedding_layer
        self.modeling_layer = modeling_layer

        self.device = device

        self.modeling_output_clasifier = torch.nn.Linear(768, num_labels)
        self.bert4lastlayer_clasifier = torch.nn.Linear(768*4, num_labels)
        
        self.loss_function = torch.nn.CrossEntropyLoss() #會自己 softmax
        self.softmax = torch.nn.Softmax(dim=1)
        
           
    # return 出(batch, seq_len)
    def forward(
        self, 
        input_ids : torch.tensor, 
        attention_mask : torch.tensor, 
        token : torch.tensor,
        SEPind : List,
        label : torch.tensor = None
        ) -> Tuple[torch.tensor]: #loss and probability

        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token)
        bert_output = self.kernel_model(embedding_output=embedding_output, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token)

        #### -- bert modeling_layer output
        output_clf = self.modeling_layer(SEPind, bert_output)  # (batch_size, 768)        
        modeling_score = self.modeling_output_clasifier(output_clf) # (batch_size, num_labels)
        #### --------------------------------------------------

        final_score = modeling_score
        loss = self.loss_function(final_score, label)
        prob = self.softmax(final_score)
        
        return loss, prob

class Bert_QA(torch.nn.Module):
    def get_1d_init_tensor(self,m):
        w = [0] * m
        for i in range(m):
            w[i] = random.uniform(-0.8, 0.8)
        w = torch.tensor(w, requires_grad=True)
        return w

    def __init__(
        self,
        kernel_model : BertWithoutEmbedding,
        embedding_layer : BertEmbeddings,
        modeling_layer_for_clf : ModelingCLF,
        modeling_layer_for_qa : ModelingQA,
        num_labels : int,
        support_modulelist : torch.nn.ModuleList,
        support_key : torch.nn.Linear, 
        target_query : torch.nn.Linear,
        do_mtl = True,
        device = None
        ):
        super(Bert_QA, self).__init__()

        self.embedding_layer = embedding_layer
        self.kernel_model = kernel_model
        self.modeling_layer_clf = modeling_layer_for_clf
        self.modeling_layer_qa = modeling_layer_for_qa
      
        self.do_mtl = do_mtl

        self.device = device
        self.hid_size = kernel_model.config.hidden_size

        self.start_clasifier = Parameter(self.get_1d_init_tensor(768))
        self.end_clasifier = Parameter(self.get_1d_init_tensor(768))
        self.clf_clasifier = torch.nn.Linear(self.hid_size, 2)

        # self.start_clasifier = torch.nn.Linear(hid_size, hid_size)
        # self.end_clasifier = torch.nn.Linear(hid_size, hid_size)

        #### this is for clf attention mechanism
        self.support_dataset_num = support_modulelist.__len__()
        self.support_modulelist = support_modulelist
        self.support_key = support_key
        self.target_query = target_query
        #### ---------------------------------------------------------------------------

        
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss_function_label = torch.nn.CrossEntropyLoss()
        self.loss_function = torch.nn.CrossEntropyLoss()
        
    def forward(
        self,
        input_ids : torch.tensor,
        attention_mask : torch.tensor,
        token : torch.tensor,
        SEPind : List,
        label : torch.tensor = None, # reference dont need
        start_pos : List = None,  # reference dont need
        end_pos : List = None, # reference dont need
        return_toks : bool = False
        ) -> Tuple[torch.Tensor]:

        this_batch_size = input_ids.size(0)
        
        ####-----------------------------------------------------------------------------
        # this is the embedding output, the QA has answer and no answer use the same embedding layer
        # but they use the different modeling layer
        # the embedding layer is depended on which dataset you are
        # but the modeling layer is which task
        ####-----------------------------------------------------------------------------


        #### FIRST AND SECOND LAYER
        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token)
        bert_output = self.kernel_model(embedding_output=embedding_output, input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token)
        ####-----------------------------------------------------------------------------


        #### GET THE QA CLF OUTPUT #########################################################################
        ####################################################################################################

        ## methoed (1) BY SELF ATTENTION
        ##  ------------------------------------------------------------------------------------------------
        if self.do_mtl : 
            value_list = []
            key_list = []
            
            output_clf_target = self.modeling_layer_clf(SEPind, bert_output)  # (batch_size, 768)
            # value_list.append(output_clf_target)
            # key_list.append(self.support_key(output_clf_target))

            for model in self.support_modulelist:
                output_clf_support = model.modeling_layer(SEPind, bert_output)
                value_list.append(output_clf_support)
                key_list.append(self.support_key(output_clf_support))

            val = torch.stack(value_list).transpose(0,1)  #### batchsize * dataset_num * 768
            key = torch.stack(key_list).transpose(0,1)  #### batchsize * dataset_num * 768
            query = self.target_query(output_clf_target)[:,None,:] #### batchsize * 1 * 768

            attention_score = torch.matmul(query, key.transpose(-1,-2)).squeeze(dim=1) #### batchsize * 1 * 768 =>(squeeze) batchsize * datasetnum
            attention_score = attention_score / math.sqrt(768)
            attention_prob = torch.nn.functional.softmax(attention_score, dim=-1) # batchsize * dataset_num

            # print(attention_prob)

            output_clf = torch.stack([torch.matmul(attention_prob[i], val[i]) for i in range(this_batch_size)]) #(batch_size, 768)
        ## -------------------------------------------------------------------------------------------------
        
        ## method (2) USE ONLY 1 MOELING -------------------------------------------------------------------
        else:
            output_clf = self.modeling_layer_clf(SEPind, bert_output)
        ## -------------------------------------------------------------------------------------------------

        ####################################################################################################
        ####################################################################################################
        

        #### GET THE QA OUTPUT ##############################################################
        output_qa = self.modeling_layer_qa(SEPind, bert_output) # (batch_size, context_length, 768)
        #####################################################################################


        #### COUNT THE PROBABILITY OF IT HAS ANSWER OR NOT ##################################
        # CLS_list = [bert_output[i,0, :] for i in range(len(SEPind))]
        # CLS_output = torch.stack(CLS_list)
        clf_score = self.clf_clasifier(output_clf) # (batch_size, 2)
        prob = self.softmax(clf_score)
        #####################################################################################
 

        #### COUNT THE START AND END ########################################################
        start_score = (output_qa * self.start_clasifier).sum(dim=2) # (batch_size, context_length)
        end_score = (output_qa * self.end_clasifier).sum(dim=2) # (batch_size, context_length)
        #####################################################################################


        #### ONLY REFERENCE AND DON'T NEED LOSS #################################################
        ####-----------------------------------------------------------------------------
        if return_toks == True :
            start_tok = start_score.argmax(dim=1)
            end_tok = end_score.argmax(dim=1)

            batch_toks = []

            for i in range (this_batch_size) :
                if clf_score[i].argmax() == 0 : # predict it has no answer
                    batch_toks.append([])
                else :
                    batch_toks.append(input_ids[i, start_tok[i]+1 : end_tok[i]+2])  # +1 +2 because of [CLS]    
            
            return batch_toks, prob
        #########################################################################################
        

        #### NEED LOSS ##########################################################################
        ####-----------------------------------------------------------------------------
        loss_for_label = self.loss_function_label(clf_score, label)
        if label[0][1] == 1 : 
            # this batch data has answer, need the start and end position loss

            start_1hot = torch.zeros(this_batch_size, output_qa.size(1)).to(self.device)
            end_1hot = torch.zeros(this_batch_size, output_qa.size(1)).to(self.device)

            # the startpos nedd -1 because of [CLS] is get rid of during the modeling layer
            for i in range(this_batch_size):
                start_1hot[i][start_pos[i]-1] = 1 
                end_1hot[i][end_pos[i]-1] = 1 

            loss_start = self.loss_function(start_score, start_1hot)
            loss_end = self.loss_function(end_score, end_1hot)

            total_loss =  loss_for_label + (loss_start + loss_end)
        else :
            total_loss = loss_for_label

        return total_loss, prob
        #########################################################################################
