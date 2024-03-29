import datetime
import logging
import os
import sys
import time
import tensorflow as tf
import numpy as np
from sklearn import metrics
import math
from HyperParameter import *


starttime = time.time()
datafolder = os.path.join(dataset, "Data")
modelfolder = os.path.join(dataset, 'Model')
with open(os.path.join(datafolder, 'num_pro.txt'), 'r') as f:
    num_pro = eval(f.read())
with open(os.path.join(datafolder, 'num_skill.txt'), 'r') as f:
    num_skill = eval(f.read())
with open(os.path.join(datafolder, 'num_stu.txt'), 'r') as f:
    num_stu = eval(f.read())

if dataset == "Assist09":
    with open(os.path.join(datafolder, 'max_skill_len.txt'), 'r') as f:
        max_skill_len = eval(f.read())
else:
    max_skill_len = 1

final_joint_embed = np.load(os.path.join(modelfolder, "final_joint_embed.npz"), allow_pickle=True)["final_joint_embed"]
final_true_corr = np.load(os.path.join(modelfolder, "final_true_corr.npz"), allow_pickle=True)["final_true_corr"]

data_num = len(final_joint_embed)
split_point = int(data_num * split_rate)
train_joint_data, train_corr_data = final_joint_embed[:split_point], final_true_corr[:split_point]
test_joint_data, test_corr_data = final_joint_embed[split_point:], final_true_corr[split_point:]

tf_embed_target = tf.placeholder(tf.float32, [None, None], name='tf_data_embed')
tf_corr_target = tf.placeholder(tf.float32, [None], name='tf_data_corr')
tf_keep_rate = tf.placeholder(tf.float32, None, name="tf_keep_rate")

pred_w = tf.get_variable('pred_w', [1, 1 + max_skill_len], initializer=tf.truncated_normal_initializer(stddev=0.1))
pred_b = tf.get_variable('pred_b', [1, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))

tf_corr_pred = tf.matmul(tf_embed_target, tf.transpose(pred_w)) + pred_b
tf_corr_pred = tf.layers.dropout(tf_corr_pred, tf_keep_rate)
tf_corr_logits = tf.reshape(tf_corr_pred, [-1])
tf_corr_labels = tf.reshape(tf_corr_target, [-1])
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_corr_labels, logits=tf_corr_logits))

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

logfile = os.path.join(modelfolder, "trainModel.txt")
f = open(logfile, "wb+")
f.truncate()
logging.basicConfig(filename=logfile, level="DEBUG")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

startTraintime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.info(os.linesep + '-' * 45 + ' BEGIN: ' + startTraintime + ' ' + '-' * 45)
logging.info('dataset %s, problem number %d, skill number %d, student number %d' % (dataset, num_pro, num_skill, num_stu))
logging.info("data_num {0},train_data_num {1},test_data_num {2}".format(data_num, len(train_joint_data), len(test_joint_data)))

train_steps = int(math.ceil(len(train_joint_data) / float(bs)))
test_steps = int(math.ceil(len(test_joint_data) / float(bs)))

logging.info("begin training....")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    best_auc = best_acc = 0
    tmp, Loss = 0, np.zeros(epochs)
    for i in range(epochs):
        epochstarttime = time.time()
        train_loss = 0
        for j in range(train_steps):
            b, e = j * bs, min((j + 1) * bs, len(train_joint_data))
            batch_train_embed = train_joint_data[b:e:]
            batch_train_corr = train_corr_data[b:e:]
            feed_dict = {tf_embed_target: batch_train_embed, tf_corr_target: batch_train_corr, tf_keep_rate: keep_rate}
            _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += batch_loss
        train_loss /= train_steps
        test_preds, test_trues = [], []
        for j in range(test_steps):
            b, e = j * bs, min((j + 1) * bs, len(test_joint_data))
            batch_test_embed = test_joint_data[b:e:]
            batch_test_corr = test_corr_data[b:e:]
            feed_dict = {tf_embed_target: batch_test_embed, tf_corr_target: batch_test_corr, tf_keep_rate: 1}
            pred_corr, true_corr = sess.run([tf_corr_logits, tf_corr_labels], feed_dict=feed_dict)
            test_preds.append(pred_corr)
            test_trues.append(true_corr)
        test_preds = np.concatenate(test_preds, axis=0)
        test_trues = np.concatenate(test_trues, axis=0)
        test_auc = metrics.roc_auc_score(test_trues, test_preds)
        test_preds[test_preds >= 0.5] = 1.
        test_preds[test_preds < 0.5] = 0.
        test_acc = metrics.accuracy_score(test_trues, test_preds)
        epochendtime = time.time()
        records = 'Epoch %d/%d, train loss:%.4f, test acc:%.4f, test auc:%.4f, epoch time:%f' % \
                  (i + 1, epochs, train_loss, test_acc, test_auc, epochendtime - epochstarttime)
        logging.info(records)
        if best_acc + best_auc <= test_acc + test_auc:
            best_acc = test_acc
            best_auc = test_auc
        Loss[i] = round(train_loss, 4)
        if i >= early_stop:
            if all(x <= y for x, y in zip(Loss[tmp:tmp + early_stop], Loss[tmp + 1:tmp + 1 + early_stop])):
                logging.info("Early stop at %d based on loss result." % (i + 1))
                break
            tmp += 1
    logging.info("best acc:%.4f   best auc:%.4f" % (best_acc, best_auc))

endTraintime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.info(os.linesep + '-' * 45 + ' END: ' + endTraintime + ' ' + '-' * 45)

endtime = time.time()
logging.info("total time %f" % (endtime - starttime))
