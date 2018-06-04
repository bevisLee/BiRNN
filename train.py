#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import datetime
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from Bidirectional_LSTM import Bi_LSTM, Word2Vec


# ==================================================
# Data Preparatopn
# ==================================================

# Load data
W2V = Word2Vec.Word2Vec()

print("Loading data...")
start_time = time.time()
train_data = W2V.read_data("./Word2Vec/Movie_rating_data/ratings_train.txt")
test_data = W2V.read_data("./Word2Vec/Movie_rating_data/ratings_test.txt")

## tokenize the data we have
tokens = [[W2V.tokenize(row[1]),int(row[2])] for row in train_data if W2V.tokenize(row[1]) != []]
tokens = np.array(tokens)
train_X = tokens[:,0]
train_Y = tokens[:,1]

tokens = [[W2V.tokenize(row[1]),int(row[2])] for row in test_data if W2V.tokenize(row[1]) != []]
tokens = np.array(tokens)
test_X = tokens[:,0]
test_Y = tokens[:,1]

## Convert to One-hot
# import word2vec model where you have trained before
train_Y_ = W2V.One_hot(train_Y)
train_X_ = W2V.Convert2Vec("./Word2Vec/Word2vec.model",train_X)  

test_Y_ = W2V.One_hot(test_Y)  
test_X_ = W2V.Convert2Vec("./Word2Vec/Word2vec.model",test_X)  

duration = time.time() - start_time
minute = int(duration / 60)
second = int(duration) % 60
print("\nData PreProcessing : %d min %d sec" % (minute,second))
print("Data split - Train : {:d} / Dev : {:d}".format(len(train_Y), len(test_Y)))


# ==================================================
# Train & Test
# ==================================================

training_epochs = 3
result = []

for i in range(1,training_epochs+1):
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess :
        
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "./runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Train Summaries
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # test summaries
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

           print("model - ",timestamp)

            # Parameters
            Batch_size = 32
            Vector_size = 300
            Maxseq_length = 95   ## Max length of training data
            learning_rate = 0.001
            lstm_units = 128
            num_class = 2

            ## train sess
            Total_size = len(train_X)
            total_batch = int(Total_size / Batch_size)
            seq_length = [len(x) for x in train_X]
            keep_prob = 0.75

            # Tensor
            X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
            Y = tf.placeholder(tf.float32, shape = [None, num_class], name = 'Y')
            seq_len = tf.placeholder(tf.int32, shape = [None])

            BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, Maxseq_length, num_class, keep_prob)

            with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
                logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len, Maxseq_length)
                loss, optimizer, merged = BiLSTM.model_build(logits, Y, learning_rate)

            prediction = tf.nn.softmax(logits)
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # global_step counter
            global_step = tf.Variable(0, trainable=False, name='global_step')
            increment_global_step = tf.assign_add(global_step,1,name = 'increment_global_step')

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            for epoch in range(i) :
                print("\nepoch : ",epoch+1,"- BiRNN Train.....\n")
                avg_loss = 0
                for step in range(total_batch):
                    train_batch_X = train_X_[step*Batch_size : step*Batch_size+Batch_size]
                    train_batch_Y = train_Y_[step*Batch_size : step*Batch_size+Batch_size]
                    batch_seq_length = seq_length[step*Batch_size : step*Batch_size+Batch_size]
                    
                    train_batch_X = W2V.Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size)

                    summary, _ = sess.run([merged,optimizer], feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})

                    # Compute average loss
                    loss_ = sess.run(loss, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
                    avg_loss += sess.run(loss, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})/total_batch
                    acc = sess.run(accuracy , feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
                    train_writer.add_summary(summary, step)

                    current_step = sess.run(increment_global_step) 
                    if (step > 10 and step % 100 == 0) or (i == training_epochs and step == total_batch-1) :
                        time_str = datetime.datetime.now().isoformat()
                        print("loss = {:.6f} accuracy = {:.6f}".format(loss_, acc),"/ epoch =",epoch,"step =",step,"/",time_str)

                    if current_step % 1000 == 0 : 
                        # checkpoint save
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("checkpoint:",current_step,"- Saved model : {}\n".format(path))

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("\nepoch : ",epoch+1,"/ Finish - Saved model : {}\n".format(path))

            ## test sess
            print("\nepoch : ",epoch+1,"- BiRNN Test.....\n")

            test_size = len(test_X)
            test_batch = int(test_size / Batch_size)
            seq_length = [len(x) for x in test_X]
            keep_prob = 1.0 
        
            total_acc = 0
            for step in range(test_batch):
                test_batch_X = test_X_[step*Batch_size : step*Batch_size+Batch_size]
                test_batch_Y = test_Y_[step*Batch_size : step*Batch_size+Batch_size]
                batch_seq_length = seq_length[step*Batch_size : step*Batch_size+Batch_size]
                test_batch_X = W2V.Zero_padding(test_batch_X, Batch_size, Maxseq_length, Vector_size)

                summary, _ = sess.run([merged,optimizer], feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length})
                acc = sess.run(accuracy , feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length})
                total_acc += acc/test_batch

                # prediction
                pred_v = sess.run(prediction , feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length})

                if pred_v is None :
                    pred_v = np.zeros(shape=(32,2))

                X_batch = test_X[step*Batch_size : step*Batch_size+Batch_size]
                pred_df_row = pd.DataFrame(list(zip(X_batch, test_batch_Y, pred_v)),columns = ["test_X",'test_Y','pred'])

                if step == 0 :
                    pred_df = pred_df_row
                else :
                    pred_df = pd.concat([pred_df,pred_df_row],axis=0)

                # test acc print
                if (step > 10 and step % 100 == 0) or (step == test_batch-1) :
                    loss_ = sess.run(loss, feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length})
                    acc = sess.run(accuracy , feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length})

                    time_str = datetime.datetime.now().isoformat()
                    print("loss = {:.6f} accuracy = {:.6f}".format(loss_, acc),"/ step =",step,"/",time_str)
                    test_writer.add_summary(summary, step)

            # result
            print("\nepoch : ",epoch+1,"- Total Accuracy : {}".format(total_acc))
            result.append(total_acc)

            # pred_value save
            pred_df_file = out_dir+"/pred_value_"+str(epoch+1)+'.csv'
            pred_df.to_csv(pred_df_file, index=False, header=False)
            print("epoch : ",epoch+1,"- Saved pred_value : {}\n".format(pred_df_file))

# Best result
print("----------------------------------------")
print("BiRNN Finish.......")
for i in range(0,len(result)) :
    print("epoch : ",i+1," / accuracy : ",result[i])

print("\nBest Result epoch : ",result.index(max(result))+1," / accuracy : ",max(result))
