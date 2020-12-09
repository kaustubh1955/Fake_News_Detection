import torch
import torch.nn as nn

from ConvS2S import ConvEncoder
from Attention import MultiHeadAttention, PositionFeedforward



class Encoder(nn.Module): # 1 Mark
    def __init__(self, conv_layers, hidden_dim, feed_forward_dim=2048):
        super(Encoder, self).__init__()
        # Your code here
        
        self.conv = ConvEncoder(hidden_dim, conv_layers)
        self.attention = MultiHeadAttention(hidden_dim,16)
        self.feed_forward = PositionFeedforward(hidden_dim,feed_forward_dim)
        
    def forward(self, input):
        """
        Forward Pass of the Encoder Class
        :param input: Input Tensor for the forward pass.
        """
        out_lst = self.conv(input)
        outt=self.attention(out_lst,out_lst,out_lst)
        ans=self.feed_forward(outt)
        return ans
        # Your code here