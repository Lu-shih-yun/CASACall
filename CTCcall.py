import torch
import torch.nn as nn
import numpy as np
from build_model import *
import decoder
import encoder
import time
import math


Time_start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


#test data path
input_data_total = torch.from_numpy(np.load("Chunk_ecoli.npy"))


input_length = len(input_data_total[0])
model_length = input_length//8
output_length = 900
input_chennels = 256
SALayers = 5
head_num = 8
class_num = 5
conv_kernel = 33

encoder_name = encoder.Encoder(model_length,input_chennels,SALayers,head_num,conv_kernel)
decoder_name = decoder.CTCDecoder(model_length,output_length,input_chennels,class_num)
model = build_model(encoder_name,decoder_name,device)

#load model
path = input("model path :")
#Saved_model/Saved_best_model_CASACall.pth
model.load_state_dict(torch.load(path))
model.eval()

classes = [
    "A",
    "C",
    "G",
    "T"
]

arry_total = []


for data_times in range(math.ceil(len(input_data_total)/100000)): 
    source = input_data_total[data_times*100000:(data_times+1)*100000].to(device,dtype=torch.float)
    
    #convert log probability into argmax and remove repeated symbols
    arry = []
    with torch.no_grad():
        for i in range(len(source)):
            output = model(source[i:i+1]).log_softmax(1)
            probability = torch.exp(output)
            pred_raw = torch.argmax(output,dim=1).squeeze(dim=0).detach().cpu().numpy()
            del_repeat = pred_raw
            for j in range(len(del_repeat)-1):
                if del_repeat[j] == del_repeat[j+1]:
                    del_repeat[j] = 0
            arry.append(pred_raw)
    arry_total.extend(arry)


#covert 0,1,2,3,4 into A,T,C,G as 0 = blank, 1 = A, 2 = C, 3 = G, 4 = T
seq=[]
for i in range(len(arry_total)):
    seq.append(">"+str(i)+"\n")
    for j in range(len(arry_total[i])):
        if arry_total[i][j]>0:
            seq.append(classes[arry_total[i][j]-1])
    seq.append("\n")
#save as fasta file
def seq2fasta(seq):
    with open('CASACall_Ecoli_test.fasta', 'w') as f:
        f.write("".join(seq))
seq2fasta(seq)
Time_end = time.time()
Time_spend = Time_end-Time_start
print(Time_spend)
