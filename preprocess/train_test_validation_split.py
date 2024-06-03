import numpy as np
import h5py
import sklearn
from sklearn.model_selection import train_test_split


chunk = np.load("Chunk_test_data.npy",mmap_mode='r')
ref = np.load("Reference_test_data.npy",mmap_mode='r')
ref_len = np.load("Reference_length_test_data.npy",mmap_mode='r')


print(len(chunk),len(ref),len(ref_len))

chunk_train,chunk_test,ref_train,ref_test,len_train,len_test = train_test_split(chunk,ref,ref_len,test_size=0.2,random_state=66)
print(len(chunk_train),len(ref_train),len(len_test))


chunk_train = np.float32(chunk_train)
chunk_test = np.float32(chunk_test)
#chunk_vali = np.float32(chunk_vali)



np.save('bonito_training_data/Chunk',chunk_train)
np.save('bonito_training_data/Reference',ref_train)
np.save('bonito_training_data/Reference_length',len_train)
np.save('bonito_training_data/validation/Chunk',chunk_test)
np.save('bonito_training_data/validation/Reference',ref_test)
np.save('bonito_training_data/validation/Reference_length',len_test)

