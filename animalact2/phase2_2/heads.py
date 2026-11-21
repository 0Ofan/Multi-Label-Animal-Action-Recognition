"""New 768-input Proposal heads; no legacy learned weights or added projection."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from animalact2.phase1_5_1.common import validate_sequence,masked_mean


class VideoMAEProposalHead(nn.Module):
    def __init__(self,kind,hidden_size=128):
        super().__init__()
        if kind not in ("mean","bigru_mean","bigru_attention"):raise ValueError("Unknown head")
        self.kind=kind
        if kind=="mean":self.classifier=nn.Linear(768,119)
        else:
            self.gru=nn.GRU(768,hidden_size,num_layers=1,bidirectional=True,batch_first=True)
            if kind=="bigru_mean":self.classifier=nn.Linear(2*hidden_size,119)
            else:
                self.attention=nn.Linear(2*hidden_size,119,bias=False)
                self.class_weight=nn.Parameter(torch.empty(119,2*hidden_size))
                self.class_bias=nn.Parameter(torch.zeros(119));nn.init.xavier_uniform_(self.class_weight)

    def forward(self,features,lengths,valid_frame_mask,*,return_aux=False):
        validate_sequence(features,lengths,valid_frame_mask)
        if features.shape[-1]!=768:raise ValueError("Expected 768-dimensional VideoMAE features")
        if self.kind=="mean":
            logits=self.classifier(masked_mean(features,valid_frame_mask));aux={}
        else:
            packed=pack_padded_sequence(features,lengths.cpu(),batch_first=True,enforce_sorted=False)
            encoded,_=self.gru(packed);encoded,_=pad_packed_sequence(encoded,batch_first=True,total_length=features.shape[1])
            if self.kind=="bigru_mean":logits=self.classifier(masked_mean(encoded,valid_frame_mask));aux={}
            else:
                scores=self.attention(encoded).transpose(1,2).float()
                weights=scores.masked_fill(~valid_frame_mask[:,None,:],float("-inf")).softmax(-1)
                pooled=torch.bmm(weights.to(encoded.dtype),encoded)
                logits=(pooled*self.class_weight[None]).sum(-1)+self.class_bias
                aux=dict(temporal_attention=weights)
        return dict(logits=logits,**aux) if return_aux else logits
