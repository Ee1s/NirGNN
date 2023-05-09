#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import torch
import random


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)+1 
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)] 
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] 
    return us_pois, us_msks, len_max #扩充补0


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    # np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data(): #确保能对应上attr和trainset
    def __init__(self, data, attr, shuffle=False, graph=None):
        inputs = data[0] #(tr_seqs, tr_labs)
        max_num = max(max(row) for row in inputs)
        # self.attr = attr
        inputs, mask, len_max = data_masks(inputs, [0]) 
        self.inputs = np.asarray(inputs) 
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph




    def generate_batch(self, batch_size):
        if self.shuffle: 
            shuffled_arg = np.arange(self.length) 

            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i, attr_data, taxo_data):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        candidate_attrbitue1 = []
        candidate_attrbitue2 = []
        attr_data[0]=[0,0]
        for i in range(len(attr_data)):
            candidate_attrbitue1.append(attr_data[i][0])
            candidate_attrbitue2.append(attr_data[i][1])
        # candidate_attrbitue1 = attr_data[:,0]
        # candidate_attrbitue2 = attr_data[:,1]
        # targets =
        max_num = np.max(np.array(inputs))
        items, n_node, A, alias_inputs = [], [], [], []
        zero_attr1 = []
        zero_taxo1 = []
        zero_attr2 = []
        zero_taxo2 = []
        zero_taxo3 = []

        c_attr1 = []
        # c_taxo1 = []
        c_attr2 = []
        # c_taxo2 = []
        # c_taxo3 = []
        ca_item = []
        # zero_attr.append([0])
        # zero_taxo.append([0])
        # x = np.unique(zero_attr)
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))#去除重复数字,这一列多少数 函数是去除数组中的重复数字,并进行排序之后输出。
        max_n_node = np.max(n_node)#包含25 这个切片里最多这些
        for u_input in inputs: #每一个序列
            candiate = []
            node = np.unique(u_input)
            # i = 0
            # while i<4:
            #     ret = random.randint(1, max_n_node)
            #     if ret in node or ret in candiate:
            #         continue
            #     else:
            #         candiate.append(ret)
            #         i = i+1
            # print("在[a, b]之间产生随机整数：random.randint(1000, 9999)=", ret)
            items.append(node.tolist() + (max_n_node - len(node)) * [0]) #变成本组最大范围了
            u_A = np.zeros((max_n_node, max_n_node)) #构建邻接矩阵一共48个不重复点
            u_attr = {}
            u_taxo = {}

            for i in np.arange(len(u_input) - 1): #从0开始
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0] #i的位置
                v = np.where(node == u_input[i + 1])[0][0]#i邻居的位置
                u_A[u][v] = 1 #重新编号作为邻接矩阵
            # for i in np.unique(u_input):
                # x = self.attr[u_input[i]][1]
                # list_a[u] = x #self.attr[u_input[i][1]]
                try :
                    u_attr[u] = attr_data[u_input[i]]
                except:
                    print('0')
                u_attr[v] = attr_data[u_input[i + 1]]
                u_taxo[u] = taxo_data[u_input[i]]
                u_taxo[v] = taxo_data[u_input[i+ 1]]


            # value_list = list(u_attr.values())
            can_attr1 =[]
            can_attr2 = []
            # candtaxo = {}
            for i in np.arange(len(candiate)):
                can_attr1.append(attr_data[candiate[i]][0])
                can_attr2.append(attr_data[candiate[i]][1])



            attr_value_list1 = list(range(len(u_attr)+1))# 0要空着？
            taxo_value_list1 = list(range(len(u_attr)+1))
            attr_value_list2 = list(range(len(u_attr)+1))
            taxo_value_list2 = list(range(len(u_attr)+1))
            taxo_value_list3 = list(range(len(u_attr)+1))
            #candidate
            # can_attr1 = list(range(len(candattr)+1))
            # can_attr2 = list(range(len(candattr) + 1))
            # can_taxo1 = list(range(len(candattr) + 1))
            # can_taxo2 = list(range(len(candattr) + 1))
            # can_taxo3 = list(range(len(candattr) + 1))

            attr_value_list1[0] = 0
            taxo_value_list1[0] = 0
            attr_value_list2[0] = 0
            taxo_value_list2[0] = 0
            taxo_value_list3[0] = 0

            # can_attr1[0] = 0
            # can_attr2[0] = 0
            # can_taxo1[0] = 0
            # can_taxo2[0] = 0
            # can_taxo3[0] = 0

            for i in u_attr:
                attr_value_list1[i] = u_attr[i][0]
                attr_value_list2[i] = u_attr[i][1]

            for i in u_taxo:
                taxo_value_list1[i] = u_taxo[i][0]
                taxo_value_list2[i] = u_taxo[i][1]
                taxo_value_list3[i] = u_taxo[i][2]

            for i in range(max_n_node-len(u_attr)-1): #max node
                attr_value_list1.append(0)
                attr_value_list2.append(0)
            # for i in range(max_n_node - len(u_attr) - 1):
                taxo_value_list1.append(0)
                taxo_value_list2.append(0)
                taxo_value_list3.append(0)


            # for i in candattr:
            #     can_attr1[i] = candattr[i][0]
            #     can_attr2[i] = candattr[i][1]

            # for i in candtaxo:
            #     can_taxo1[i] = candtaxo[i][0]
            #     can_taxo2[i] = candtaxo[i][1]
            #     can_taxo3[i] = candtaxo[i][2]

            # u_A1 = np.array([
            #     [0,1,0,1],
            #     [0,0,1,0],
            #     [1,0,0,0],
            #     [0,0,0,0]
            # ])
            # # u_A1 = np.array([
            # #     [0,1,0,0],
            # #     [0,0,1,1],
            # #     [0,1,0,0],
            # #     [0,0,0,0]
            # # ])


            u_sum_in = np.sum(u_A, 0) #按行相加并保持特性
            u_sum_in[np.where(u_sum_in == 0)] = 1 #等于0的地方变成等于1
            u_A_in = np.divide(u_A, u_sum_in)#归一化
            u_sum_out = np.sum(u_A, 1) #按列
            u_sum_out[np.where(u_sum_out == 0)] = 1 #归一化
            u_A_out = np.divide(u_A.transpose(), u_sum_out)#相除
            u_A = np.concatenate([u_A_in, u_A_out]).transpose() #入度和出度？拼在一起 横着拼
            A.append(u_A) #18*36的矩阵

            zero_attr1.append(attr_value_list1) # u_attr
            zero_taxo1.append(taxo_value_list1)
            zero_attr2.append(attr_value_list2) # u_attr
            zero_taxo2.append(taxo_value_list2)
            zero_taxo3.append(taxo_value_list3)

            ca_item.append(candiate)

            c_attr1.append(can_attr1)
            c_attr2.append(can_attr2)
            # c_taxo1.append(can_taxo1)
            # c_taxo2.append(can_taxo2)
            # c_taxo3.append(can_taxo3)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) #记录每个item的在邻接矩阵里顶点的顺序

        # try:
        #     zero_attr1 = torch.LongTensor(zero_attr1)
        # except:
        #     print(zero_attr1)
        return alias_inputs, A, items, mask, targets,zero_attr1,zero_attr2,zero_taxo1,zero_taxo2,zero_taxo3,candidate_attrbitue1,candidate_attrbitue2#,c_taxo1,




