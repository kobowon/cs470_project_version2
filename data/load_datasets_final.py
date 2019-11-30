import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  )
from data.create_inputs_utils import *
from data.create_inputs_utils_test import convert_examples_to_features, load_and_cache_examples


data_dir = '../preprocessed_data'

processor = CaptionProcessor()
task_name = "sst-2"#임의 설정 (필요없는 값)
output_mode = "classification" #임의 설정 (필요없는 값)
model_name_or_path = "bert-base-uncased" 
max_seq_length = 50
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
label_list = processor.get_labels()
num_labels = len(label_list)

config_class = BertConfig
model_class = BertForSequenceClassification
tokenizer_class = BertTokenizer

config = config_class.from_pretrained(model_name_or_path, num_labels=num_labels, finetuning_task = task_name)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case = True)
#model = model_class.from_pretrained(model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path), config=config)

#model.to(device)      

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        self.processor = CaptionProcessor()
        assert self.split in {'TRAIN', 'VAL','TEST'}

        # Open hdf5 file where images are stored
        self.train_hf = h5py.File(data_folder + '/train36.hdf5', 'r')
        self.train_features = self.train_hf['image_features']
        self.val_hf = h5py.File(data_folder + '/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']

        # Captions per image
        self.cpi = 5
        
        # Load encoded captions  #[0][0:idx,1,2,3]
        #self.captions = load_and_cache_examples(self.processor, tokenizer,max_seq_length,model_name_or_path,[None],data_dir,self.split)
        self.captions = load_and_cache_examples(self.processor, tokenizer,max_seq_length,model_name_or_path,[None],data_folder,self.split)
        #print("@@@@@@@@@@@")

        #with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
        #    self.captions = json.load(j)

        # Load caption lengths 
        
        #with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.pickle'), 'rb') as f:
        #    self.caplens = pickle.load(f)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
            
        # Load bottom up image features distribution
        with open(os.path.join(data_folder, self.split + '_GENOME_DETS_' + data_name + '.json'), 'r') as j:
            self.object = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        
        #return 값 : img와 caption[input_ids:0,position_ids:1,toklen_type_ids:2,labels:3], caption 5개(VAL, TEST만) 
        
        # The Nth caption corresponds to the (N // captions_per_image)th image
        #i : caption index라고 생각
        #0~4 index에   image index는 0을 대응해서, 하나의 image에 5개의 caption을 대응시킴
        object = self.object[i // self.cpi]
        
        # Load bottom up image features
        if object[0] == "v":
            img = torch.FloatTensor(self.val_features[object[1]])
        else:
            img = torch.FloatTensor(self.train_features[object[1]])

        
        #이렇게하면 len이랑 caption이랑 매칭 안됨 매칭하려면 LEN 다시 저장해야함
        caption = self.captions[i]    #captions[i][0:idx,1,2,3]
        #caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])
        
        if self.split is 'TRAIN': #train일 때는 캡션 하나씩만 주고
            return img, caption, caplen
        else:#test일 때는 캡션 전체도 줌
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            #captions_test[0][0] [batch][input_ids:0,position_ids:1,toklen_type_ids:2,labels:3]
            all_captions = self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)]
            return img, caption, caplen, all_captions

        
    def __len__(self):
        return self.dataset_size
