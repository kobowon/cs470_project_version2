import torch
from torch import nn
# import torchvision
from models.attention import Attention
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, features_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        
        self.processor = CaptionProcessor()
        self.task_name = "sst-2"#임의 설정 (필요없는 값)
        self.output_mode = "classification" #임의 설정 (필요없는 값)
        self.model_name_or_path = "bert-base-uncased" 
        self.max_seq_length = 50
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)

        self.config_class = BertConfig
        self.model_class = BertForSequenceClassification
        self.tokenizer_class = BertTokenizer

        self.config = self.config_class.from_pretrained(self.model_name_or_path, num_labels=self.num_labels, finetuning_task = self.task_name)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name_or_path, do_lower_case = True)
        self.model = self.model_class.from_pretrained(self.model_name_or_path, from_tf=bool('.ckpt' in self.model_name_or_path), config=self.config)

        #self.model = self.model.to(device)    
        
        
        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(features_dim, decoder_dim, attention_dim)  # attention network

        #bowon
        #self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.embedding = self.model.bert.embeddings.to(device)
        #self.embedding = self.model.bert.to(device)
        self.bert_encoder = self.model.bert.encoder.layer[0].to(device)
        
        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = nn.LSTMCell(embed_dim + features_dim + decoder_dim, decoder_dim, bias=True) # top down attention LSTMCell
        self.language_model = nn.LSTMCell(features_dim + decoder_dim, decoder_dim, bias=True)  # language model LSTMCell
        self.fc1 = nn.utils.weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.fc = nn.utils.weight_norm(nn.Linear(decoder_dim, vocab_size))  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        #self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self,batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size,self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size,self.decoder_dim).to(device)
        return h, c

    def forward(self, image_features, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        #encoded_captions : [0:input_ids,1:attention_mask,2:token_type_ids,3:labels] X batch_size [[x],[x],[x]..]
        #encoded_captions[0:batch_size]

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size

        # Flatten image
        image_features_mean = image_features.mean(1).to(device)  # (batch_size, num_pixels, encoder_dim)

        # Sort input data by decreasing lengths; why? apparent below
        
        #print(encoded_captions)
        
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        #print("sort_ind : ",sort_ind)
        #print("caption_lengths : ",caption_lengths)
        
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        
        #이걸 구현해야 함 - 완료
        #encoded_captions = encoded_captions[sort_ind]
        
        input_ids_tensor = []
        attention_mask_tensor = []
        token_type_ids_tensor = []
        labels_tensor = torch.tensor([0]*batch_size,dtype=torch.long).to(device)    
        
        for i in sort_ind:
            input_ids_tensor.append(encoded_captions[0][i.item()].tolist())
            attention_mask_tensor.append(encoded_captions[1][i.item()].tolist())
            token_type_ids_tensor.append(encoded_captions[2][i.item()].tolist())
            
        
        input_ids_tensor = torch.tensor(input_ids_tensor,dtype=torch.long).to(device)    
        attention_mask_tensor = torch.tensor(attention_mask_tensor,dtype=torch.long).to(device)    
        token_type_ids_tensor = torch.tensor(token_type_ids_tensor,dtype=torch.long).to(device)    
        
        encoded_captions = input_ids_tensor

        # Embedding
        
        
        #embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        
        inputs = {'input_ids':      input_ids_tensor,
                  'attention_mask': attention_mask_tensor,
                  'token_type_ids': token_type_ids_tensor, 
                  'labels':         labels_tensor}
        
        embedding_inputs = {'input_ids':inputs['input_ids'],
                            'position_ids':None,
                            'token_type_ids':inputs['token_type_ids']}
        
        bert_inputs = {'input_ids': input_ids_tensor,
                       'attention_mask':attention_mask_tensor,
                       'token_type_ids':token_type_ids_tensor,}
        
        
        #embeddings = self.embedding(**embedding_inputs)
        embeddings_output = self.embedding(**embedding_inputs)
        embeddings_output = embeddings_output.type(torch.FloatTensor).to(device)
        
        
        extended_attention_mask = attention_mask_tensor.unsqueeze(1).unsqueeze(2).type(torch.FloatTensor).to(device)
        encoder_inputs = {'hidden_states':embeddings_output,
                          'attention_mask':extended_attention_mask,}
        
        layer_outputs = self.bert_encoder(**encoder_inputs)
        #print("@@@@@@@@@@@@@@@@@@@@@@@@")
        #print("@@@@@@@@@@@@@@@@@@@@@@@@")
        #print("@@@@@@@@@@@@@@@@@@@@@@@@")
        #print("@@@@@@@@@@@@@@@@@@@@@@@@")
        #print(layer_outputs[0].size())
        embeddings = layer_outputs[0]
        
        #output = self.embedding(**bert_inputs)
        #embeddings = output[0]
        #print(embeddings.size())
        
        
        
        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths-1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        predictions1 = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        
        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up 
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model 
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h1,c1 = self.top_down_attention(
                torch.cat([h2[:batch_size_t],image_features_mean[:batch_size_t],embeddings[:batch_size_t, t, :]], dim=1),(h1[:batch_size_t], c1[:batch_size_t]))
            attention_weighted_encoding = self.attention(image_features[:batch_size_t],h1[:batch_size_t])
            preds1 = self.fc1(self.dropout(h1))
            h2,c2 = self.language_model(
                torch.cat([attention_weighted_encoding[:batch_size_t],h1[:batch_size_t]], dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]))
            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            predictions1[:batch_size_t, t, :] = preds1
        
        #print('predictions : ', predictions)
        #print('predictions size : ',predictions.size())
        #print('decode_lengths : ', decode_lengths)
        #print('decode_lengths size : ',len(decode_lengths)

        return predictions, predictions1,encoded_captions, decode_lengths, sort_ind

