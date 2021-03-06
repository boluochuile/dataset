import sys
import copy
import random
import numpy as np
from collections import defaultdict


def data_partition(fname, flag='train'):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    # assume user/item index starting from 1
    if flag == 'train':
        f = open('/content/dataset/SASRec/data/%s_data/%s_train.txt' % (fname, fname), 'r')
    elif flag == 'valid':
        f = open('/content/dataset/SASRec/data/%s_data/%s_valid.txt' % (fname, fname), 'r')
    else:
        f = open('/content/dataset/SASRec/data/%s_data/%s_test.txt' % (fname, fname), 'r')
    for line in f:
        # 0,773,0
        u, i, timestamp = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    return [User, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    [users, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    for u in users:

        if len(users[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)

        user_len = len(users[u])
        item_idx = [users[u][user_len - 1]]

        idx = args.maxlen - 1
        for i in reversed(users[u][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(users[u])
        rated.add(0)

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # return sess.run(self.test_logits,
        #                         {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print ('.'),
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

# dataset = data_partition('ml-1m')
# [user_train, user_valid, user_test, usernum, itemnum] = dataset
# print(len(user_train), '_', usernum)
# print(user_test)