import numpy as np

# Dataset Setting
mfcc_win_step_per_frame = 1     # wav
mfcc_dim = 26 + 26 + 13
up_sample_rate = 4
is_up_sample = True
hasPhoneme = True

kernel_type = 'lstm'


# kernel param setting
win_size = 24  # 8
#n_layers = 3
n_layers = 3
#n_steps = 8
n_steps = 24
n_input = int(mfcc_dim * mfcc_win_step_per_frame * win_size / n_steps)
n_hidden = 256

n_out_fc1 = 256

weighted = True

n_phoneme = 20

# just removing the beginning and ending 'sp's
# if clean:
#     n_phoneme -= 1 #remove 'sp'

initial_learning_rate = 1e-4
#1e-4,bat_size increases,no_regularization
#every 1000 steps save it
p_alpha = 0

dropout = 0.2

#deal with the 0 std
sel_id = range(65)
#del sel_id[39]
#deal with 0 weights
lr_decay_epoch = 100
lr_decay_rate = 0.5
total_epoch_num = 15

batch_size = 256
n_sample = 1407444
train_buffer_size = int(n_sample*0.7)
test_buffer_size = int(2e5)
mean,std = np.loadtxt('utl/mean_std.txt')

save_step_num = 1000
test_step_num = 2500
save_ckpt = True
#z_dim = 100

is_train = False
#sr = 16000

# DIR setting
data_dir = "../data/tfrecord"
#dataset_train = ["grid_1","grid_2","lombardgrid_1","lombardgrid_2","lombardgrid_3","savee"]
dataset_train = ["timit_train_1","timit_train_2","timit_train_3","timit_train_4"]
dataset_validation = ["timit_test"]
dataset_test = ['kb_2k_cl']
checkpoint_dir = "checkpoint"
log_dir = "logs"
val_dir = "test_audio/train_SA1.wav"#first one
test_dir = "test_audio/demo_tr_vo.wav"
#test_dir = "test_audio/demo_tr_vo.wav"#last one