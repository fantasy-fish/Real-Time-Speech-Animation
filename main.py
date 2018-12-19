import tensorflow as tf
from utl.load_param import *
from model import Model
from record import *
from message import *
import os
import shutil
import time
import stomp
import scipy.io.wavfile as wavfile

def main(_):
    if os.path.isfile("tmp/prediction"):
        os.remove("tmp/prediction")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        model = Model(sess=sess)
        if model.load():
            #record_to_file("tmp/tmwp.wav")
            (rate, sig) = wavfile.read(test_dir)
            if (sig.ndim > 1):
                sig = sig[:, 0]  # pick mono-acoustic track
            win_len = 160*23+400
            win_step = 160
            n_pieces = (len(sig)-win_len)/win_step+1
            with open('utl/phoneme_list') as f:
                phoneme_list = f.readlines()
                phoneme_list = [ph[:-1] for ph in phoneme_list]
                group_list = phoneme_list
            #prediction is on 10 ms base
            frame_len = 12
            #frame_len = 4
            for i in range(n_pieces):
                if i%frame_len != 0:
                    continue
                t1 = time.time()
                cur_sig = sig[win_step*i:win_step*i+win_len]
                print cur_sig[-10:]
                cur_mean = np.mean(cur_sig)
                rms = np.sqrt(np.sum((cur_sig-cur_mean)*(cur_sig-cur_mean))/len(cur_sig))
                #print rms
                pr = model.predict(rate,cur_sig,group_list)
                if rms<1000:
                    pr = '_'
                #print pr
                sendMessageToSmartBody("sb scene.getDiphoneManager().setPhonemesRealtime('foo', '{}')".format(pr))
                print time.time()-t1
        else:
            print "Model not loaded"


if __name__ == '__main__':
    tf.app.run()
    