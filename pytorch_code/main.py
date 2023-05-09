#!/usr/bin/env python36
# -*- coding: utf-8 -*-


import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yelpsmall', help='dataset name')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=5, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    attr_data = pickle.load(open('../datasets/' + opt.dataset + '/all_attr.txt', 'rb'))
    taxo_data = pickle.load(open('../datasets/' + opt.dataset + '/all_taxo.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, attr_data, shuffle=True) #test 和train里面包含的东西不一样
    test_data = Data(test_data, attr_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'yelpsmall':
        n_node = 743601
    elif opt.dataset == 'mt_all' or opt.dataset == 'yoochoose1_4':
        n_node = 684
    else:
        n_node = 18889 #一共有多少个顶点，这里得改

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch) #data[1]是目标分数
        hit, mrr, hit10, mrr10= train_test(model, train_data, test_data,attr_data,taxo_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        if hit10 >= best_result[2]:
            best_result[2] = hit10
            best_epoch[2] = epoch
            flag = 1
        if mrr10 >= best_result[3]:
            best_result[3] = mrr10
            best_epoch[3] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d' % (best_result[2], best_result[3], best_epoch[2], best_epoch[3]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
