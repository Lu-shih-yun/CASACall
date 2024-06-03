import torch.nn as nn
import numpy as np
import torch
#from TorchCRF import CRF

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, input):
        encoder_outputs = self.encoder(input)
        output = self.decoder(encoder_outputs)
        return output

def build_model(encoder, decoder, device):
  # 建構模型
  model = Seq2Seq(encoder, decoder)
  #print(model)

  model = model.to(device)
  return model

class Seq2Seq_CRF(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, input):
        encoder_outputs = self.encoder(input)
        #print(np.shape(encoder_outputs))
        output = self.decoder(encoder_outputs)
        return output
    
def build_model_CRF(encoder, decoder, device):
  # 建構模型
  model = Seq2Seq_CRF(encoder, decoder)
  #print(model)

  model = model.to(device)
  return model