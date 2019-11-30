# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
1
# coding=utf-8
2
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
3
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
4
#
5
# Licensed under the Apache License, Version 2.0 (the "License");
6
# you may not use this file except in compliance with the License.
7
# You may obtain a copy of the License at
8
#

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
#import tensorboardX
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  )

from transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
    }




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    #tensor board 용도
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    #총 batch_size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # Samples elements randomly
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #데이터를 배치단위로 가져오는 iterator
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    #총 optimization step 수 (t_total) 결정
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
    #bowonko <2019.11.1>
    #1 epoch마다 accuracy를 출력
    args.eval_steps = 200#len(train_dataloader)   
    args.logging_steps = args.eval_steps
    
    #bowonko <2019.10.7>
    #('bert.embeddings' in n) or ('bert.encoder.layer.0.' in n) or ('bert.encoder.layer.1.' in n) or ('bert.pooler.' in n)
    #freeze할 parameter를 제외한 parameter list 
    params_list_remove_freeze_params= []
    for n,p in model.named_parameters():
        params_list_remove_freeze_params.append((n,p))

    #print([n for n,p in params_list_remove_freeze_params])

    #decay되면 안되는 element
    no_decay = ['bias', 'LayerNorm.weight']
 
    optimizer_grouped_parameters = [
        #모델의 파라미터 중에 decay안되는 것들이 속해있지 않으면 weight_decay 설정해도되고
        {'params': [p for n, p in params_list_remove_freeze_params if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #모델의 파라미터 중에 decay되지 말아야할 파라미터들(bias, layernorm의 가중치)이 포함되어 있으면 weight_decay하지 않는다
        {'params': [p for n, p in params_list_remove_freeze_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]    
    
    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #총 optimization step(t_total) 설정할 수 있음
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    #생략/ detail : apex from https://www.github.com/nvidia/apex
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    
    model.zero_grad()

    sep_idx_list = []
    
    #epoch iterator
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]) 
    for curr_epoch in train_iterator:
        
        #early stopping bowonko 2019.11.14
        if args.current_patient == 0:
            epoch_iterator.close()
            break        
        
        #one epoch(전체 데이터)에 대한 batch iterator
        epoch_iterator = tqdm(train_dataloader, desc="[current batch/1 epoch total batch]", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3],
                      'all_sep_ids':    batch[4]}
            #bowonko <2019.11.05~06> 각 batch data의 sep 위치(mid,end)저장 - sep_idx_list = [(),()...()]
        
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            
            #매 step(batch) 마다 optimization (gradient_accumulation_step이 1일 때)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                optimizer.step() # update parameter
                scheduler.step()  # Update learning rate schedule
              
                model.zero_grad() 
                global_step += 1
                
                
                #print("@@@@@@@@@@@@@@@@@@@@@@1@@@@@@@@@@@@@@")
               
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print("@@@@@@@@@@@@@@@@@@@@@@2@@@@@@@@@@@@@@")
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss
                    
                    if args.local_rank == -1 and args.evaluate_during_training and global_step % args.eval_steps == 0:  # Only evaluate when single GPU otherwise metrics may not average well
                        print("@@@@@@@@@@@@@@@@@@@@@@3@@@@@@@@@@@@@@")
                        print("[Evaluation Start]")
                        results = evaluate(args, model, tokenizer,global_step,curr_epoch+1) #2019.09.23 global step 추가
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                                                
                
                #저장을 일정한 간격으로 하고 싶을 때 사용
                '''if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)'''

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            #early stopping bowonko 2019.11.14
            '''if args.current_patient == 0:
                epoch_iterator.close()
                break
           ''' 
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
            
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="",curr_epoch=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                          'labels':         batch[3],
                          'all_sep_ids':    batch[4]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)

        ##bowonko
        if(args.task_name == 'mrpc' or args.task_name =='qqp' or args.task_name =='paws'):
            print('@@@@@@@@@@@@@@MRPC OR QQP OR PAWS@@@@@@@@@@@@@@@@')
            print("Task_name : ",args.task_name)
            curr_acc = result['acc']
            curr_f1 = result['f1']
            best_acc = args.best_acc 
            best_f1 = args.best_f1
            if((curr_acc > best_acc or curr_f1 > best_f1) and args.current_patient > 0):
                #성능향상이 있었으면 patient값 초기화
                args.current_patient = args.patient_value
                if(curr_acc > best_acc):
                    best_acc = curr_acc
                    args.best_acc = best_acc
                    args.best_epoch = curr_epoch
                if(curr_f1 > best_f1):
                    best_f1 = curr_f1
                    args.best_f1 = best_f1
                    args.best_epoch = curr_epoch
                
                output_eval_file = os.path.join(eval_output_dir, "eval_result.txt")
                with open(output_eval_file,"w") as writer:
                    logger.info("***** Eval result {} *****".format(prefix))
                    for key in sorted(result.keys()):
                        logger.info(" %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                    writer.write("epoch : %s\n" %args.best_epoch)
                    writer.write("seed : %d\n" %args.seed)
                    
                global_step = prefix #bowon
                output_checkpoint_dir = os.path.join(eval_output_dir, 'checkpoint-{}-{}'.format(args.best_epoch,global_step))
                if not os.path.exists(output_checkpoint_dir):
                    os.makedirs(output_checkpoint_dir)
                model_to_save = model.module if hasattr(model, 'module') else model # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_checkpoint_dir)
                torch.save(args, os.path.join(output_checkpoint_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_checkpoint_dir)
            
            else:#성능향상이 없거나 patient가 0이면
                if args.current_patient == 0:
                    print("학습 종료(early stopping)\n")
                    #eval끝 -train 종료
                else:#성능향상이 없고 , current_patient > 0
                    args.current_patient -= 1
                    print("patient 1 감소, patient : ",args.current_patient,"\n\n")
                
            #1 epoch마다 뜨는 거
            print('[Record]\n','acc : %.4f' %curr_acc, 'curr_f1 : %.4f'%curr_f1)
            print('[Current Best Record]\n','curr_acc: %.4f' %args.best_acc, 'best f1 : %.4f'%args.best_f1, 'best_epoch : ',args.best_epoch, 'seed : ',args.seed)
                
        elif(args.task_name == 'qnli'):
            print('@@@@@@@@@@@@@@QNLI@@@@@@@@@@@@@@@@')
            curr_acc = result['acc']
            best_acc = args.best_acc
            if(curr_acc > best_acc):
                args.current_patient = args.patient_value
                best_acc = curr_acc
                args.best_acc = best_acc
                args.best_epoch = curr_epoch
                
                output_eval_file = os.path.join(eval_output_dir, "eval_result.txt")
                with open(output_eval_file,"w") as writer:
                    logger.info("***** Eval result {} *****".format(prefix))
                    for key in sorted(result.keys()):
                        logger.info(" %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                    writer.write("epoch : %s\n" %args.best_epoch)
                    writer.write("seed : %d\n" %args.seed)

                global_step = prefix #bowon
                output_checkpoint_dir = os.path.join(eval_output_dir, 'checkpoint-{}-{}'.format(args.best_epoch,global_step))
                if not os.path.exists(output_checkpoint_dir):
                    os.makedirs(output_checkpoint_dir)
                model_to_save = model.module if hasattr(model, 'module') else model # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_checkpoint_dir)
                torch.save(args, os.path.join(output_checkpoint_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_checkpoint_dir)
                        
            else:#성능향상이 없거나 patient가 0이면    
                if args.current_patient == 0:
                    print("학습 종료(early stopping)\n")
                    #eval끝 -train 종료
                else:#성능향상이 없고 , current_patient > 0
                    args.current_patient -= 1
                    print("patient 1 감소\n")                    
            print('[Record]\n','curr acc : %.4f' %curr_acc)    
            print('[Current Best Record]\n','best acc : %.4f' %args.best_acc, 'best_epoch : ',args.best_epoch)
       

        elif(args.task_name == 'sts-b'):
            print('@@@@@@@@@@@@@@STS-B@@@@@@@@@@@@@@@@')
            curr_pearson = result['pearson']
            curr_spearmanr = result['spearmanr']
            best_pearson = args.best_pearson
            best_spearmanr = args.best_spearmanr
            
            if(curr_pearson > best_pearson):
                best_pearson = curr_pearson
                args.best_pearson = best_pearson
                args.best_epoch = curr_epoch
                
            if(curr_spearmanr > best_spearmanr):
                best_spearmanr = curr_spearmanr
                args.best_spearmanr = best_spearmanr
                args.best_epoch = curr_epoch
            
            if(best_pearson == curr_pearson or best_spearmanr == curr_spearmanr):
                #둘 중하나 update
                args.current_patient = args.patient_value
                #evaluation file 작성
                output_eval_file = os.path.join(eval_output_dir, "eval_result.txt")
                with open(output_eval_file,"w") as writer:
                    logger.info("***** Eval result {} *****".format(prefix))
                    for key in sorted(result.keys()):
                        logger.info(" %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                    writer.write("epoch : %s\n" %args.best_epoch)

                global_step = prefix #bowon
                output_checkpoint_dir = os.path.join(eval_output_dir, 'checkpoint-{}-{}'.format(args.best_epoch,global_step))
                if not os.path.exists(output_checkpoint_dir):
                    os.makedirs(output_checkpoint_dir)
                model_to_save = model.module if hasattr(model, 'module') else model # Take care of distributed/parallel training
                #모델을 output dir에 저장
                model_to_save.save_pretrained(output_checkpoint_dir)
                #arguments도 저장
                torch.save(args, os.path.join(output_checkpoint_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_checkpoint_dir)
            
            else:#성능향상이 없거나 patient가 0이면
                if args.current_patient == 0:
                    print("학습 종료(early stopping)\n")
                    #eval끝 -train 종료
                else:#성능향상이 없고 , current_patient > 0
                    args.current_patient -= 1
                    print("patient 1 감소\n")                    
            
            print('[Current Best Record] \n','best pearson : %.4f'%args.best_pearson, 'best spearmanr :  %.4f'%args.best_spearmanr, 'best_epoch : ',args.best_epoch)
                
        results.update(result)

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
    #bunbun bowonko 2019. 11. 06

    all_input_ids_list = [f.input_ids for f in features]
    
    sep_idx_list = []
    for sen in all_input_ids_list:
        sep_item = ()
        for i, tok in enumerate(sen):
            #102 : SEP
            if(tok == 102):
                sep_item += (i,)
        sep_idx_list.append(sep_item)
                
    #print(all_input_ids_list[0])
    #print(sep_idx_list[0])

    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_sep_ids = torch.tensor(sep_idx_list, dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_sep_ids)
    return dataset


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=10000,
                        help="Evaluate every X updates steps.")
    #안씀
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    
    #bowon default qqp 용도
    parser.add_argument('--best_acc', type=float, default=0.,
                        help="best accuracy")
    parser.add_argument('--best_f1', type=float, default=0.,
                        help="best f1 score")    
    
    parser.add_argument('--best_pearson', type=float, default=0., help ="best pearson")
    parser.add_argument('--best_spearmanr', type=float, default=0., help ="best spearmanr")
    parser.add_argument('--best_epoch', type=int, default=0,help="best epoch")    
    parser.add_argument('--current_patient', type=int, default=2, help="early stopping")
    parser.add_argument('--patient_value', type=int, default=2, help="early stopping")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    args.seed = 42
    #args.seed = random.randint(0,100)
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    print("@@@@@@@@@@@@")
    print("@@@@@@@@@@@@")
    print("@@@@@@@@@@@@")
    print("@@@@@@@@@@@@Task Name : ",args.task_name)
    print("@@@@@@@@@@@@")
    print("@@@@@@@@@@@@")
    print("@@@@@@@@@@@@")
    
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    
    #bowonko <2019.11.1> from_pretrained를 바꿔야함 - 완료 - classifier만 제외
    
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    
    

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        
        #sep위치를 어떻게 뽑아오지
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        
        #bowonko
        '''print(train_dataset[1][0])
        print(train_dataset[2][0])
        print(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))
        return
        '''
        #bowonko
        for param in model.parameters():
            param.requires_grad = False
        
        for p in model.bert.embeddings.parameters():
            p.requires_grad = True
        
        for p in model.classifier.parameters():
            p.requires_grad = True
        
        for p in model.bert.pooler.parameters():
            p.requires_grad = True
        #bunbun
        for num, layer in enumerate(model.bert.encoder.layer):
            for n, p in layer.named_parameters():
                if num in range(0,12):
                    p.requires_grad = True

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        #학습이 다 끝났을 때의 상태를 저장
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation 
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    
    #bowonko <2019.11.1> from_pretrained를 바꿔야함 - 완료 - classifier만 제외
    
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        model.to(args.device)
        
        #tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.model_name_or_path]
        if args.eval_all_checkpoints:#안함
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            #model = model_class.from_pretrained(checkpoint)
            #model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
            print("◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈result◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈")
            #checkpoint 결과만 보여주는 거 
            
        if(args.task_name == 'mrpc' or args.task_name == 'qqp' or args.task_name == 'paws'):
            print('best acc : ',args.best_acc, 'best f1 : ',args.best_f1, 'best_epoch : ',args.best_epoch)

        elif(args.task_name == 'sts-b'):
            print('best_pearson : ',args.best_pearson,'best_spearmanr : ',args.best_spearmanr, 'best_epoch : ',args.best_epoch )
    
    return results


if __name__ == "__main__":
    main()
