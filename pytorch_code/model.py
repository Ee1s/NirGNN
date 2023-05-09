#!/usr/bin/env python36
# -*- coding: utf-8 -*-


import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import scipy.spatial.distance as dist
from torch import distributions

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):#构造session 图
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size) #num_embeddings, embedding_dim
        self.gnn = GNN(self.hidden_size, step=opt.step) 
        self.weight = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True) #？
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
        self.nn = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        self.nn1 = nn.Linear(in_features=4 * self.hidden_size, out_features=self.hidden_size)
        self.nnb = nn.Linear(in_features=3 * self.hidden_size, out_features=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.logic = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(self.hidden_size, 512)
        self.bn1_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, self.hidden_size)
        self.bn_fc2 = nn.BatchNorm1d(self.hidden_size)
        self.W = nn.Parameter(torch.zeros(size=(self.hidden_size, self.hidden_size)))
        # self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
#计算分数
    def compute_scores(self, hidden, mask,topk):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)#sg
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1)) 
        # b = self.embedding.weight[1:]  # n_nodes x latent_size
        b = topk
        scores = torch.matmul(a, b.transpose(1, 0))

        return scores
#计算意图
    def compute_intent(self, item):
        q2 =self.linear_two(item)
        alpha = self.linear_three(torch.sigmoid(q2))
        intent = alpha*item
        # intent = torch.sum(alpha * item, 1)  # sg
        return intent

    def compute_beta_scores(self, hidden, mask,topk,beta):
        beta_std = torch.std(beta,dim=2).unsqueeze(dim=2)
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        beta1 = beta[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q2 = beta*hidden
        q1 = (beta1*ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
          # batch_size x seq_length x latent_size
        avg =  torch.mean(q1+q2,dim=2)
        avg = torch.unsqueeze(avg,2)
        alpha = self.linear_three((q1+q2-avg)/beta_std)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)#sg
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1)) #公式7sh
        # b=torch.sum(a,dim=-1)
        # print(b[18])
        # b = self.embedding.weight[1:]  # n_nodes x latent_size
        b = topk
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores


    def beta_intent(self,hidden,taxo1,taxo2,taxo3):
        taxo = self.nnb(torch.cat((taxo1,taxo2,taxo3),dim =-1))
        taxo = torch.nn.functional.softplus(taxo)
        hidden = torch.nn.functional.softplus(hidden)
        B = torch.distributions.beta.Beta(hidden, taxo)
        beta = B.sample()
        return beta

    def forward(self, inputs, A, attr1, attr2, taxo1, taxo2,taxo3,ca1,ca2):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        attr1 = self.embedding(attr1) #100,14,100
        attr2 = self.embedding(attr2)
        attr1_intent = self.compute_intent(attr1)
        attr2_intent = self.compute_intent(attr2)
        taxo1 = self.embedding(taxo1) #
        taxo2 = self.embedding(taxo2)
        taxo3 = self.embedding(taxo3)
        ca1 =self.embedding(ca1)
        ca2 = self.embedding(ca2)
        hidt_beta = self.beta_intent(hidden,taxo1,taxo2,taxo3)
        hidt = self.nn1(torch.cat((hidden,taxo1,taxo2,taxo3), dim=-1))




        attr_intent = self.nn(torch.cat((attr1_intent, attr2_intent), dim=-1))

        # attr_intent = attr1_intent #torch.cat((attr1_intent,attr2_intent),2)
        ca = self.nn(torch.cat((ca1,ca2),dim=-1))

        return hidden, hidt, attr_intent,ca, hidt_beta


    def map(self,x):
        # x = torch.cat((x,attr),1)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc2(x)
        return x

    def bhattacharyya_distance_1(self, vector1, vector2):
        # 点与样本集的巴氏距离：
        a = vector1.detach().cpu().numpy() * vector2.detach().cpu().numpy()
        b = np.sqrt(a)
        for i in range(len(b)):
            for j in range(len(b[i])):
                if math.isnan(b[i][j]):
                    b[i][j] = 0
        # for i in range(len(b)):
        #     if math.isnan(b[i][j]):
        #         b[i][j] = 0

        bc = np.sum(b,axis =1)
        # BC = np.sum(np.sqrt(vector1.detach().numpy() * vector2.detach().numpy()))
        # dt = np.ones(100, dtype=float)#
        return -np.log(bc)#, np.sqrt(dt-bc)

    def bhattacharyya_distance_2(self, vector1, vector2):
        # 点与样本集的巴氏距离：
        a = vector1.detach().cpu().numpy() * vector2.detach().cpu().numpy()
        b = np.sqrt(a)
        # for i in range(len(b)):
        #     for j in range(len(b[i])):
        #         if math.isnan(b[i][j]):
        #             b[i][j] = 0
        for i in range(len(b)):
            if math.isnan(b[i]):
                b[i] = 0

        bc = np.sum(b, axis=0)
        # BC = np.sum(np.sqrt(vector1.detach().numpy() * vector2.detach().numpy()))
        # dt = np.ones(100, dtype=float)#
        return np.log(bc)  # , np.sqrt(dt-bc)

    def Mahalanobis_distance_2(self,x, y):
        x =x.detach().numpy()
        y = y.detach().numpy()
        X = np.vstack([x, y])
        XT = X.T
        d2 = dist.pdist(XT, 'mahalanobis')
        return d2

    def zeroshot(self, intent_item , intent_attribute, candidate_attribute,mask,alias_inputs):
        get = lambda i: intent_item[i][alias_inputs[i]]
        seq_item = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        oriitem = torch.sum(seq_item * mask.view(mask.shape[0], -1, 1).float(), 1)

        get = lambda i: intent_attribute[i][alias_inputs[i]]
        seq_intent = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        intent = torch.sum(seq_intent * mask.view(mask.shape[0], -1, 1).float(), 1)

        # xx = self.linear_zero(intent)


        distanceb = self.bhattacharyya_distance_1(self.linear_zero(intent),oriitem)
        # distancem = self.Mahalanobis_distance_2(self.linear_zero(intent),oriitem)
        lossB = torch.LongTensor(distanceb)
        # lossh = torch.LongTensor(distanceh)

        loss = torch.sum(lossB)#+torch.sum(lossh)
        #------------topk做法------------------
        # intent2 = torch.unsqueeze(intent, 1)
        # intent2 = intent2.repeat(1,4,1)
        # sim_score = F.cosine_similarity(intent2, candidate_attribute, dim=2).clamp(min=-1, max=1)
        # score = torch.Tensor(sim_score)
        # top_val, idx = torch.topk(score, 1) #candidate_value
        # getcan = lambda i:candidate_attribute[i][idx[i]]
        # seq_can = torch.stack([getcan(i) for i in torch.arange(len(alias_inputs)).long()])
        # seq_can= torch.squeeze(seq_can,1)
        #----------------no topk----------------
        #candidate 换0

        candidate_item_embedding = self.linear_zero(candidate_attribute)
        return candidate_item_embedding, loss

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, attr_data, taxo_data): #getslice 没有家属性 model slices,traindata
    alias_inputs, A, items, mask, targets, attr1, attr2, taxo1, taxo2, taxo3, ca_a1,ca_a2 = data.get_slice(i, attr_data,taxo_data) # 前两个没看懂 target是目标分数
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long()) #放到cuda里
    items = trans_to_cuda(torch.Tensor(items).long()) #放到cuda里
    A = trans_to_cuda(torch.Tensor(A).float())#放到cuda里
    mask = trans_to_cuda(torch.Tensor(mask).long())

    attr1 = trans_to_cuda(torch.LongTensor(attr1))
    attr2 = trans_to_cuda(torch.LongTensor(attr2))
    taxo1 = trans_to_cuda(torch.Tensor(taxo1).long())
    taxo2 = trans_to_cuda(torch.Tensor(taxo2).long())
    taxo3 = trans_to_cuda(torch.Tensor(taxo3).long())
    ca_a1 = trans_to_cuda(torch.LongTensor(ca_a1))
    ca_a2 = trans_to_cuda(torch.LongTensor(ca_a2))


    hidden, hidt, attr_intent,ca,beta = model(items, A, attr1,attr2,taxo1,taxo2,taxo3,ca_a1,ca_a2)


    get = lambda i: hidt[i][alias_inputs[i]]
    # get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    get2 = lambda i: beta[i][alias_inputs[i]]
    seq_beta = torch.stack([get2(i) for i in torch.arange(len(alias_inputs)).long()])

    candidate_item_embedding, zeroloss = model.zeroshot(hidden,attr_intent,ca,mask,alias_inputs)
    score1 = model.compute_scores(seq_hidden, mask, candidate_item_embedding)
    score2 = model.compute_beta_scores(seq_hidden, mask, candidate_item_embedding,seq_beta)
    score = score1*0.5+score2*0.5
    return targets, score, zeroloss








def train_test(model, train_data, test_data, attr_data, taxo_data): 
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size) 
    for i, j in zip(slices, np.arange(len(slices))):  # 123
        model.optimizer.zero_grad() 
        targets, scores, zeroloss = forward(model, i, train_data, attr_data, taxo_data)
        targets = trans_to_cuda(torch.Tensor(targets).long()) 
        loss = model.loss_function(scores, targets - 1)*0.7+ 0.3*zeroloss 
        # loss = zeroloss
        loss.backward()

        # loss.backward(torch.ones_like(zeroloss))
        model.optimizer.step()

        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
   # print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr, hit10, mrr10 = [], [], [], []
    slices = test_data.generate_batch(model.batch_size)
    # subtestslice = slices[0:10]
    # for i in subtestslice:
    for i in slices:
        targets, scores, zeroloss = forward(model, i, test_data, attr_data, taxo_data)#model, i, train_data
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
        sub_scores10 = scores.topk(10)[1]
        sub_scores10 = trans_to_cpu(sub_scores10).detach().numpy()
        for score, target, mask in zip(sub_scores10, targets, test_data.mask):
            hit10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100

    return hit, mrr,hit10,mrr10


