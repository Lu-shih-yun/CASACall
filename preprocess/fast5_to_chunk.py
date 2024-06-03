import h5py
import os
import numpy as np
from statsmodels import robust
import gc

thisdir = 'fast5_raw_signal/'
#find all file path of hdf5 file in dir
ovlplen = 100

def find_HDF5(folder_dir):
    filepathes =  []
    for r, d, f in os.walk(folder_dir):
        for file in f:
            if file.endswith(".fast5") :
                filepathes.append(os.path.join(r, file))
    return filepathes

def split_signal(fast5_data,chunk_size = 3600):
    split_raw = []
    rawpath = 'Raw/Reads/'
    try:
        raw_start = fast5_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs['read_start_rel_to_raw']
        raw = fast5_data['Raw/Reads/']
        row_end = fast5_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'][-1][2]+fast5_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'][-1][3]
        for name in raw.keys():
            rawpath = rawpath+name

        signal = fast5_data[rawpath+'/Signal'][raw_start:]

        signal = (signal - np.median(signal)) / np.float64(robust.mad(signal))
        for x in range(0,row_end,chunk_size):
            Chunk = signal[()][x:x+chunk_size]

            if len(Chunk)==chunk_size:
                split_raw.append(Chunk)

    except:
        pass

    return split_raw
def split_ref(fast5_data,chunk_size = 3600,padded_length = 486):

    #container for chunked reference, single chunk of reference, reference lengthes and signal
    ref = []
    ref_row = []
    ref_lengths = []
    Chunks = []
    fast5_all = []
    switch = False
    fast5_count = 0
    start_base = 0

    base_dict = {b'A':1,b'C':2,b'G':3,b'T':4}
    end_of_chunk = chunk_size

    try:
        rawpath = 'Raw/Reads/'
        raw = fast5_data['Raw/Reads/']
        
        for name in raw.keys():
            rawpath = rawpath+name
        raw_start = fast5_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs['read_start_rel_to_raw']
        signal = fast5_data[rawpath+'/Signal'][raw_start:]
        
        #normalize signal
        signal = (signal - np.median(signal)) / np.float32(robust.mad(signal))
    except:
        pass

    try:
        for row in fast5_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']:
            fast5_all.append(row)

        
        while fast5_count < len(fast5_all):
            signal_current = fast5_all[fast5_count][2]
            if (signal_current >= end_of_chunk - ovlplen) and (switch == False):
                start_signal = fast5_all[fast5_count][2]
                start_base = fast5_count
                switch = True
            
            if fast5_all[fast5_count][2] >= end_of_chunk -1:
                ref.append(np.pad(ref_row,(0,padded_length-len(ref_row)),mode='constant',constant_values=5))
                ref_lengths.append(len(ref_row))
                ref_row.clear()
                switch = False
                Chunks.append(signal[()][end_of_chunk-chunk_size:end_of_chunk])
                end_of_chunk = chunk_size + start_signal
                fast5_count = start_base
                #print(end_of_chunk)
            else:
                ref_row.append(base_dict.get(fast5_all[fast5_count][4]))
                fast5_count += 1

    except:
        pass

    return ref_lengths,ref,Chunks

def run(folder_dir):
    filepathes = find_HDF5(folder_dir)
    Chunk = []
    Reference = []
    Reference_length = []
    i = 0
    for filepath in filepathes:
        
        try:
            fast5_data = h5py.File(filepath, 'r')
        except IOError:
            assert IOError('Error opening file. Likely a corrupted file.')
            
        #filter with guppy basecall mean qcosre
        if fast5_data['Analyses/Basecall_1D_000/Summary/basecall_1d_template'].attrs['mean_qscore'] > 14:
            ref_length,ref,signal = split_ref(fast5_data)
            Chunk += signal
            Reference += ref
            Reference_length += ref_length

            i += 1
            if i % 100 == 0:
                gc.collect()   
                print(filepath)
                print(len(Reference_length),len(Reference),len(Chunk))
        
            
    np.save("Chunk_test_data",Chunk)
    np.save("Reference_test_data",Reference)
    np.save("Reference_length_test_data",Reference_length)

run(thisdir)
