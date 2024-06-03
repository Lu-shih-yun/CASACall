# CASACall

Shih-Yun Lu, Jopu Sun, Chie-Pein Chen, Tong-Miin Liou and Chien-Chong Hong

## Preparation
### Requirement
python == 3.9.0
pytorch == 1.11.0
CUDA == 11.6
numpy == 1.26.4
h5py == 3.8.0

optional software:
Minimap2 == 2.23
Rebalar == 0.2.0

### Datasets
In our paper, we used the Bonito dataset as the training dataset, which can be downloaded from here: https://github.com/nanoporetech/bonito
If you want to test your own fast5 format data, you can put the data in preprocess/fast5_raw_signal and use fast5_to_chunk.py to generate .npy format files for testing.



