from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from model import *
from .utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

import json
import numpy as np
from abc import *
from pathlib import Path
import random


class AdversarialAttacker(metaclass=ABCMeta):
    def __init__(self, args, model, test_loader,max_predictions,mask_prob):
        self.args = args
        self.device = args.device
        self.num_items = args.num_items
        self.max_len = args.bert_max_len
        self.model = model.to(self.device)
        ##
        self.max_predictions = max_predictions
        
        self.mask_prob = mask_prob
        
        
        
        
        
        ##
        
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.test_loader = test_loader
        self.mask_token = args.num_items + 1
        self.num_max_iters = min(len(self.test_loader),
            args.num_attack_eval // args.test_batch_size)
        self.adv_ce = nn.CrossEntropyLoss(ignore_index=0)

        if isinstance(self.model, Locker) or isinstance(self.model, BERT):
            self.item_embeddings = self.model.embedding.token.weight.detach().cpu().numpy()[1:-1]
        else:
            self.item_embeddings = self.model.embedding.token.weight.detach().cpu().numpy()[1:]
        self.item_embeddings = torch.tensor(self.item_embeddings).to(self.device)

    def substitution_attack(self, target=None, num_attack=10, min_cos_sim=0.5, repeated_search=10):
        if target is None:
            print('## Untargeted Substitution Attack ##')
        else:
            print('## Substitution Attack on Item {} ##'.format(str(target)))
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.test_loader, total=self.num_max_iters)
        Trueseq=[]
        changeseq=[]
        allseq=[]
        labels1=torch.zeros(64,200)
        #a=torch.empty()
        for batch_idx, batch in enumerate(tqdm_dataloader):
            if batch_idx >= self.num_max_iters:
                break
            self.model.eval()
            with torch.no_grad():
                if isinstance(self.model, Locker) or isinstance(self.model, BERT) or isinstance(self.model, SASRec):
                    seqs, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    #print('seqs :{}'.format(str(seqs)),'batch_idx:{}'.format(str(batch_idx)),'target:{}'.format(str(target)))
                    perturbed_seqs = seqs.clone()
                    embeddings, mask = self.model.embedding(perturbed_seqs.long())#扰动项嵌入
                elif isinstance(self.model, NARM):
                    seqs, lengths, candidates, labels = batch
                    lengths = lengths.flatten()
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    perturbed_seqs = seqs.clone()
                    embeddings, mask = self.model.embedding(perturbed_seqs.long(), lengths)
            #np.savetxt('ori.csv',seqs.cpu().numpy(),fmt='%.2f',delimiter=',')
            self.model.train()
            embeddings = embeddings.detach().clone()
            embeddings.requires_grad = True
            if embeddings.grad is not None:
                embeddings.grad.zero_()
            if isinstance(self.model, Locker) or isinstance(self.model, BERT) or isinstance(self.model, SASRec):
                x =  self.model.embedding.token.weight
                scores = self.model.model(embeddings,x, mask)[:, -1, :]
            elif isinstance(self.model, NARM):
                scores = self.model.model(embeddings, self.model.embedding.token.weight, lengths, mask)
            
            if target is None:
                y= scores.argmax(-1)
                loss = self.adv_ce(scores,y )
            else:
                loss = self.adv_ce(scores, torch.tensor([target] * perturbed_seqs.size(0)).to(self.device))
            self.model.zero_grad()
            loss.backward()
            embeddings_grad = embeddings.grad.data
            importance_scores = torch.norm(embeddings_grad, dim=-1)

            self.model.eval()
            with torch.no_grad():
                attackable_indicies = (perturbed_seqs != self.mask_token)
                x = perturbed_seqs != 0
                attackable_indicies = (perturbed_seqs != 0) * attackable_indicies#找出容易攻击的项目
                importance_scores = importance_scores * attackable_indicies#对应容易攻击项的重要性分数
                _, descending_indicies = torch.sort(importance_scores, dim=1, descending=True)#对分数进行从大到小排序
                descending_indicies = descending_indicies[:, :num_attack]#取前num_attack个项对应的编号（0-199）
                
                best_seqs = perturbed_seqs.clone().detach()
                #print('best_seqs:{}'.format(str(best_seqs)))
                for num in range(num_attack):
                    row_indices = torch.arange(seqs.size(0))#64
                    col_indices = descending_indicies[:, num]

                    current_embeddings = embeddings[row_indices, col_indices]#拿到嵌入信息（64，64）
                    current_embeddings_grad = embeddings_grad[row_indices, col_indices]#拿到梯度信息
                    all_embeddings = self.item_embeddings.unsqueeze(1).repeat_interleave(current_embeddings.size(0), 1)#（3416，64，64）
                    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                    
                    if target is None:
                        multipication_results = torch.t(cos(current_embeddings+current_embeddings_grad.sign(), all_embeddings))#（64，3416）
                    else:
                        multipication_results = torch.t(cos(current_embeddings-current_embeddings_grad.sign(), all_embeddings))
                    cos_filter_results = cos(all_embeddings, current_embeddings)#（64，3416）
                    cos_filter_results = torch.t(cos_filter_results >= min_cos_sim)
                    multipication_results = multipication_results * cos_filter_results#剔除余弦相似度小于0.5的
                    _, candidate_indicies = torch.sort(multipication_results, dim=1, descending=True)#（64，3416）

                    if target is not None:
                        if_prev_target = (best_seqs[row_indices, col_indices-1] == target)
                        multipication_results[:, target-1] = multipication_results[:, target-1] + (if_prev_target * -1e9)
                        _, candidate_indicies = torch.sort(multipication_results, dim=1, descending=True)
                    
                    best_seqs[row_indices, col_indices] = candidate_indicies[:, 0] + 1
                    
                    if isinstance(self.model, Locker) or isinstance(self.model, BERT) or isinstance(self.model, SASRec):
                        logits = F.softmax(self.model(best_seqs)[:, -1, :], dim=-1)
                    elif isinstance(self.model, NARM):
                        logits = F.softmax(self.model(best_seqs, lengths), dim=-1)
                    
                    if target is None:
                        best_scores = torch.gather(logits, -1, candidates[:, 0:1]).squeeze()
                    else:
                        best_scores = torch.gather(logits, -1, torch.tensor([target] * best_seqs.size(0)).unsqueeze(1).to(self.device)).squeeze()

                    for time in range(1, repeated_search):#取前十列
                        temp_seqs = best_seqs.clone().detach()
                        temp_seqs[row_indices, col_indices] = candidate_indicies[:, time] + 1#替换原始序列中

                        if isinstance(self.model, Locker) or isinstance(self.model, BERT) or isinstance(self.model, SASRec):
                            logits = F.softmax(self.model(temp_seqs)[:, -1, :], dim=-1)
                        elif isinstance(self.model, NARM):
                            logits = F.softmax(self.model(temp_seqs, lengths), dim=-1)

                        if target is None:
                            temp_scores = torch.gather(logits, -1, candidates[:, 0:1]).squeeze()#计算
                            best_seqs[row_indices, col_indices] = temp_seqs[row_indices, col_indices] * (temp_scores <= best_scores) + best_seqs[row_indices, col_indices] * (temp_scores > best_scores)
                            best_scores = temp_scores * (temp_scores <= best_scores) + best_scores * (temp_scores > best_scores)
                            best_seqs = best_seqs.detach()
                            best_scores = best_scores.detach()
                            del temp_scores
                        else:
                            temp_scores = torch.gather(logits, -1, torch.tensor([target] * best_seqs.size(0)).unsqueeze(1).to(self.device)).squeeze()
                            best_seqs[row_indices, col_indices] = temp_seqs[row_indices, col_indices] * (temp_scores >= best_scores) + best_seqs[row_indices, col_indices] * (temp_scores < best_scores)
                            best_scores = temp_scores * (temp_scores >= best_scores) + best_scores * (temp_scores < best_scores)
                            best_seqs = best_seqs.detach()
                            best_scores = best_scores.detach()
                            del temp_scores

                
            perturbed_seqs = best_seqs.detach()
            ##取出对抗性样本
            # if batch_idx==94:
            #     labels1=torch.zeros(24,200)
            #     labels1[:,199]=candidates[:,0]
            # else:
            #     labels1[:,199]=candidates[:,0]
            # p = labels1
            # OP = perturbed_seqs
            # with open('seqs.csv','ab') as f:
            #     np.savetxt(f,OP.cpu().numpy(),fmt='%i',delimiter=',')
            # f.close
            # with open('labels1.csv','ab') as f:
            #     np.savetxt(f,p.cpu().numpy(),fmt='%i',delimiter=',')
            # f.close
            
            

            
            # #统计被攻击项被攻击的次数
            #C=torch.cat((a,seqs),0) 
            # tt = seqs.reshape(-1)
            # for num in Trueseq:
            #     for i in tt:
            #         if i.item()==num:
            #             allseq.append(num)
            #print('perturbed_seqs:{}'.format(str(perturbed_seqs)),'seqs:{}'.format(str(seqs)))
            if isinstance(self.model, Locker) or isinstance(self.model, BERT) or isinstance(self.model, SASRec):
                perturbed_scores = self.model(perturbed_seqs)[:, -1, :]
            elif isinstance(self.model, NARM):
                perturbed_scores = self.model(perturbed_seqs, lengths)
            
            if target is not None:
                candidates[:, 0] = torch.tensor([target] * candidates.size(0)).to(self.device)    
            perturbed_scores = perturbed_scores.gather(1, candidates)
            metrics = recalls_and_ndcgs_for_ks(perturbed_scores, labels, self.metric_ks)
            self._update_meter_set(average_meter_set, metrics)
            self._update_dataloader_metrics(tqdm_dataloader, average_meter_set)
            # d = seqs.eq(perturbed_seqs)
            # dim0, dim1 = d.shape
            # for i in range(dim0):
            #     for j in range(dim1):
            #         if d[i][j] == False:
            #             Trueseq.append(seqs[i][j])
            #             #changeseq.append(perturbed_seqs[i][j])
            # tt = seqs.reshape(-1)
            # for i in tt:
            #     allseq.append(i.item())
        #c = np.array(Trueseq)
        #d = np.array(allseq)
        #ar,num=np.unique(c,return_counts=True)
        #ar1,num1=np.unique(d,return_counts=True)
        #dic=dict(zip(ar,num))
        # dic1=dict(zip(ar1,num1))
        #a = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        # b = sorted(dic1.items(), key=lambda x: x[1], reverse=True)
        #Filename = str(target)+'.txt'
        # Filename1 = str(target)+'appear.txt'
        # doc1=open(Filename1,'w')
        #doc = open(Filename,'w')
        #print(a,file=doc)
        # print(b,file=doc1)
        #doc.close
        # doc1.close
        average_metrics = average_meter_set.averages()
        return average_metrics

    def test(self, target=None):
        if target is not None:
            print('## Black-Box Targeted Test on Item {} ##'.format(str(target)))
        else:
            print('## Black-Box Untargeted Test ##')
        
        self.model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader, total=self.num_max_iters)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                if batch_idx >= self.num_max_iters:
                    break
                if isinstance(self.model, Locker) or isinstance(self.model, BERT) or isinstance(self.model, SASRec):
                    seqs, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    scores = self.model(seqs)[:, -1, :]
                elif isinstance(self.model, NARM):
                    seqs, lengths, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    lengths = lengths.flatten()
                    scores = self.model(seqs, lengths)
                
                if target is not None:
                    candidates[:, 0] = torch.tensor([target] * seqs.size(0)).to(self.device)
                scores = scores.gather(1, candidates)
                metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            return average_metrics

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]
                               ] + ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Eval: ' + \
            ', '.join(s + ' {:.3f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(
            *(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)
