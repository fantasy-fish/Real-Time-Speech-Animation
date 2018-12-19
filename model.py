import tensorflow as tf

from utl.load_param import *
from utl.utl import _parse_function
import os,math
import time
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import logfbank, mfcc, ssc

class Model:
    def __init__(self,sess):
        self.sess = sess
        self.build_model()

    def build_model(self):
        # dataset input pipeline
        with tf.name_scope('dataset'):
            dataset_tr = [os.path.join(data_dir,data+'.tfrecords') for data in dataset_train]
            dataset_va = [os.path.join(data_dir,data+'.tfrecords') for data in dataset_validation]
            dataset_te = [os.path.join(data_dir,data+'.tfrecords') for data in dataset_test]
            self.training_dataset = tf.data.TFRecordDataset(dataset_tr).shuffle(train_buffer_size)
            self.training_dataset = self.training_dataset.apply(
                                    tf.contrib.data.map_and_batch(_parse_function,batch_size,drop_remainder=True))
            self.validation_dataset = tf.data.TFRecordDataset(dataset_va)
            self.validation_dataset = self.validation_dataset.apply(
                tf.contrib.data.map_and_batch(_parse_function, batch_size, drop_remainder=True))
            self.testing_dataset = tf.data.TFRecordDataset(dataset_te).shuffle(test_buffer_size)
            self.testing_dataset = self.testing_dataset.apply(
                tf.contrib.data.map_and_batch(_parse_function, batch_size, drop_remainder=True))
            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                self.handle, self.training_dataset.output_types, self.training_dataset.output_shapes)
            self.x, self.y = iterator.get_next()

        #  tf graph input
        with tf.name_scope('input'):
            #self.batch_size = tf.placeholder(tf.int32,[],name="batch_size")
            #self.x = tf.placeholder("float32", [None, n_steps, n_input])
            #self.y = tf.placeholder("float32", [None, n_phoneme])
            self.phase = tf.placeholder(tf.bool, name='phase')
            self.weights = tf.placeholder("float32", [n_phoneme,1])
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.dropout = tf.placeholder("float32")
            self.seq_len = tf.placeholder(tf.int16,[batch_size])

        # fully connected layer weights and bias
        with tf.name_scope('net1_fc'):
            n_out_phoneme_fc2 = n_phoneme
            w1_pho = tf.Variable(tf.truncated_normal([n_hidden , n_out_fc1],
                        stddev=np.sqrt(2.0 / (n_hidden  + n_out_fc1)), dtype=tf.float32), name='net1_w1_pho')
            w2_pho = tf.Variable(tf.truncated_normal([n_out_fc1, n_out_phoneme_fc2],
                        stddev=np.sqrt(2.0 / (n_out_fc1  + n_out_phoneme_fc2)),dtype=tf.float32), name='net1_w2_pho')
            b1_pho = tf.Variable(tf.truncated_normal([n_out_fc1],
                        stddev=np.sqrt(2.0/n_hidden), dtype=tf.float32), name='net1_b1_pho_noreg')
            b2_pho = tf.Variable(tf.truncated_normal([n_out_phoneme_fc2],
                        stddev=np.sqrt(2.0/n_out_fc1),dtype=tf.float32), name='net1_b2_pho_noreg')

            # w1_pho = tf.Variable(
            #     tf.truncated_normal([n_hidden , n_out_fc1], stddev=2 / (n_hidden  + n_out_fc1),
            #                         dtype=tf.float32), name='net1_w1_pho')
            # w2_pho = tf.Variable(
            #     tf.truncated_normal([n_out_fc1, n_out_phoneme_fc2], stddev=2 / (n_out_fc1 + n_out_phoneme_fc2),
            #                         dtype=tf.float32), name='net1_w2_pho')
            # b1_pho = tf.Variable(tf.ones([n_out_fc1], dtype=tf.float32) * 0.1, name='net1_b1_pho')
            # b2_pho = tf.Variable(tf.zeros([n_out_phoneme_fc2], dtype=tf.float32), name='net1_b2_pho')


        # LSTM model
        with tf.name_scope('net1_shared_rnn'):

            if (kernel_type == 'rnn'):
                cell_func = tf.contrib.rnn.BasicRNNCell
            elif (kernel_type == 'lstm'):
                # cell_func = tf.contrib.rnn.BasicLSTMCell
                cell_func = tf.contrib.rnn.LSTMCell
            elif (kernel_type == 'gru'):
                cell_func = tf.contrib.rnn.GRUCell

            def one_layer_lstm_kernel(x, dropout, n_hidden):
                lstm_cell = cell_func(n_hidden, initializer=tf.glorot_normal_initializer())
                cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0 - dropout)
                return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

            def n_layer_rnn_kernel(x, dropout, n_layers, n_hidden):
                cells = []
                for _ in range(n_layers):
                    lstm_cell = cell_func(n_hidden)
                    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0 - dropout)
                    #state = lstm_cell.zero_states(batch_size, tf.float32)
                    cells.append(lstm_cell)
                cell = tf.contrib.rnn.MultiRNNCell(cells)
                lstm_state_as_tensor_shape = [n_layers, 2, batch_size, n_hidden]
                initial_state = tf.zeros(lstm_state_as_tensor_shape)
                unstack_state = tf.unstack(initial_state, axis=0)
                self.initial_state = tuple(
                    [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1]) for idx in
                     range(n_layers)])
                return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32,initial_state=self.initial_state,
                                         sequence_length=self.seq_len,scope='net1_rnn')

            net1_rnn_output, self.final_state = n_layer_rnn_kernel(x=self.x, dropout=self.dropout, n_layers=n_layers,
                                                    n_hidden=n_hidden)  # x in [n_batch x n_step x n_feature]
            # outputs = net1_rnn_output
            #print net1_rnn_output.get_shape()
        with tf.name_scope('net1_output'):
            self.outputs = net1_rnn_output[:, -1, :]
            #self.outputs.shape = (batch_size,n_step=24,n_hidden=256)
            #take the output at the last timestep
            p1 = tf.matmul(self.outputs, w1_pho) + b1_pho
            p2 = tf.contrib.layers.batch_norm(p1, center=True, scale=True, is_training=self.phase, scope='net1_bn1_noreg')
            p2 = tf.nn.dropout(p2,keep_prob=1.0-self.dropout)
            p3 = tf.nn.relu(p2, name='net1_relu1')
            p4 = tf.matmul(p3, w2_pho) + b2_pho
            # p4 = tf.contrib.layers.batch_norm(p4, center=True, scale=True, is_training=self.phase, scope='net1_bn2')
            # p4 = tf.nn.relu(p4, name='net1_relu2')
            # p5 = tf.matmul(p4, w3_pho) + b3_pho
            self.pred = p4


        # error
        with tf.name_scope('net1_pho_err'):
            mistakes = tf.not_equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1))
            net1_pho_err = tf.reduce_mean(tf.cast(mistakes, tf.float32))

        # loss
        with tf.name_scope('net1_loss'):
            self.cost = dict()
            self.y_ = tf.nn.softmax(self.pred)
            self.cost['net1_pho'] = tf.reduce_mean(tf.matmul(-self.y * tf.log(self.y_+1e-10), self.weights))
            #self.cost['net1_pho'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.pred))
            self.cost['net1_pho_err'] = net1_pho_err

            t_vars = tf.trainable_variables()
            reg_vars = [var for var in t_vars if not 'bias' in var.name and not 'noreg' in var.name]
            reg_losses_1 = [tf.reduce_sum(tf.nn.l2_loss(var)) for var in reg_vars]
            self.cost['net1_regularization'] = sum(reg_losses_1) / len(reg_losses_1)

            self.cost['net1'] = self.cost['net1_pho'] + p_alpha * self.cost['net1_regularization']
            #self.cost['net1'] = self.cost['net1_pho']

        self.saver = tf.train.Saver()

    def save(self,step):
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, "predictor"),global_step=step)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

    def predict(self,rate,sig,group_list):
        #---------------------------------------------------------------------------#
        #test
        #print len(sig)
        fps = 25
        # print('FPS: {:}'.format(fps))
        winstep = 1.0 / fps / up_sample_rate
        mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=winstep, numcep=13)
        logfbank_feat = logfbank(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
        ssc_feat = ssc(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
        full_feat = np.concatenate([mfcc_feat, logfbank_feat, ssc_feat], axis=1)
        # full_feat = logfbank_feat

        aligned_length_wav = full_feat
        npWav = np.array(aligned_length_wav)
        n_samples = len(npWav)

        # normalize wav-raw
        mean, std = np.loadtxt('utl/mean_std.txt')
        wav_raw = npWav
        wav_raw = (wav_raw - mean) / std
        #wav_raw = wav_raw[:, sel_id]

        # grouping
        x = list()
        x.append(wav_raw)
        n_batch =1
        n_sample_needed = n_batch*batch_size - len(x)
        x += [x[-1] for _ in range(n_sample_needed)]
        x = np.array(x)
        #print x.shape

        state_test = self.sess.run(self.initial_state)
        batch_x = x
        seq_len = np.array([n_steps]+[0]*(batch_size-1))
        feed = {self.x: batch_x,
                self.phase: False,
                self.dropout: 0,
                #self.batch_size: batch_size,
                self.initial_state: state_test,
                self.seq_len:seq_len}
        batch_y,_ = self.sess.run([self.pred,self.final_state],
                                feed_dict=feed)

        y = batch_y[0]

        y = np.array(y)
        # np.savetxt("prediction/{}.txt".format(step),y)
        id = np.argmax(y)

        phonemes = group_list[id]
        #print phonemes
        #consider the up sample rate
        return phonemes
