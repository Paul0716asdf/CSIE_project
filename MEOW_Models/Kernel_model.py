from transformers import BertModel, BertTokenizer, BertPreTrainedModel
import torch
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import *
import random
from MEOW_Utils.model_utils import Highway_layer, get_retrieve_context_matrix, pading_empty_tensor, get_context_from_LHL
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class BertWithoutEmbedding(BertPreTrainedModel):
    
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        embedding_output,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        # we don't use pooler output for kernel model
        pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class ModelingQA(torch.nn.Module):  # LSTM + Highway_layer
    def __init__(
        self, 
        hid_size,
        device
        ):
        super(ModelingQA, self).__init__()
        
        self.device = device
        self.highway_layer = Highway_layer()
        # self.LSTM = torch.nn.LSTM(input_size=hid_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.LSTM = torch.nn.LSTM(input_size=hid_size, hidden_size=hid_size, num_layers=2, batch_first=True, dropout = 0.4, bidirectional = True)

    def forward(
        self,
        SEPind : List,
        outputs : BaseModelOutputWithPoolingAndCrossAttentions # this is the output of BertWithoutEmbedding
        ) -> Tuple[torch.Tensor]: # return loss only
        
        last_hidden_layer = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size:768)
        hidden_size = last_hidden_layer.size(2)

        context_LHL = get_context_from_LHL(last_hidden_layer, SEPind) # (batch_size, max_context_length, 768)
        
        #### highway --------------------------------
        highway_output = self.highway_layer(context_LHL) # (batch_size, max_context_length, 768)
        ## we need the mtx tor get retrieve the context because after the highway later the zero tensor will add the constant tensor
        mtx = get_retrieve_context_matrix(SEPind, max_context_len = highway_output.size(1), hidden_layer_size = hidden_size)
        mtx = mtx.to(self.device)
        highway_output = highway_output * mtx
        #### ----------------------------------------

        #### LSTM -----------------------------------
        context_PACk = pading_empty_tensor(highway_output) # this is an object
        output, (hn,cn) = self.LSTM(context_PACk)
        output, input_sizes = pad_packed_sequence(output, batch_first=True) # output is (batch_size, context_padding_length, 768)

        out_split = output.view(output.size(0), output.size(1), 2, int(output.size(2)/2))
        out_forward = out_split[:, :, 0, :]
        out_backward = out_split[:, :, 1, :]
        concat_output = out_forward + out_backward
        #### ----------------------------------------
        
        return concat_output

class ModelingCLF(torch.nn.Module):
    def get_2d_init_tensor(self,m,n):
        w = torch.empty(m,n, requires_grad=True)
        w = torch.nn.init.xavier_normal_(w)
        return w

    def get_1d_init_tensor(self,m):
        w = [0] * m
        for i in range(m):
            w[i] = random.uniform(-0.8, 0.8)
        w = torch.tensor(w, requires_grad=True)
        return w

    def __init__(self, pooler, hidden_size, device) -> None:
        super(ModelingCLF, self).__init__()
        # self.Wcls = Parameter(self.get_2d_init_tensor(768,num_labels))
        # self.bcls = Parameter(self.get_1d_init_tensor(num_labels))
        
        self.pooler = pooler
        self.dropout = torch.nn.Dropout(p=0.2)
        self.hidden_size = hidden_size

        self.linear_4bertlastlayer = torch.nn.Linear(in_features=768*4, out_features=768)
        self.dropout = torch.nn.Dropout(p=0.3)

        self.device = device
        

    def forward(
        self,
        SEPind : List,
        bert_output : BaseModelOutputWithPoolingAndCrossAttentions # this is the output of BertWithoutEmbedding
        ) -> Tuple[torch.Tensor]: # return loss only
        
        #### the output of orignial pooler output
        LHL = bert_output.last_hidden_state
        pooler_outupt = self.pooler(LHL) # (batch_size, hidden_size:768)
        pooler_outupt = self.dropout(pooler_outupt)
        ####---------------------------------------------------

        #### --- bert 4 last hidden concat and output 
        last4layer = bert_output.hidden_states
        v1 = last4layer[-1][:,0,:]
        v2 = last4layer[-2][:,0,:]
        v3 = last4layer[-3][:,0,:]
        v4 = last4layer[-4][:,0,:]

        vec = torch.cat([v1,v2,v3,v4], dim=1)
        linear_output = self.linear_4bertlastlayer(vec)
        linear_output = self.dropout(linear_output)
        ####---------------------------------------------------

        # cls_output = last_hidden_layer[:,0]
        # cls_score = torch.matmul(cls_output, self.Wcls)
        # cls_score = self.activation(cls_score)
        
        return pooler_outupt + linear_output
