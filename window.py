import tensorflow as tf
from utl.load_param import *
from model import Model
from message import *
import shutil
import os
import scipy.io.wavfile as wavfile

if os.path.isfile("tmp/prediction"):
		os.remove("tmp/prediction")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
model = Model(sess=sess)
if not model.load():
	print "Model loading failed"
	exit()

with open('utl/phoneme_list') as f:
		phoneme_list = f.readlines()
		phoneme_list = [ph[:-1] for ph in phoneme_list]
		group_list = phoneme_list

import Tkinter as tk
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave
import time

CHUNK_SIZE = 160
FRAME_SIZE = 24
#WIN_LEN = CHUNK_SIZE*FRAME_SIZE
#WIN_LEN = 160*23+400
WIN_LEN = 160*23+400+1024
FORMAT = pyaudio.paInt16
RATE = 16000
path = 'tmp/tmp.wav'
clean_path = 'tmp/tmp_clean.wav'

window = tk.Tk()
window.title("recorder")
window.geometry('300x400')

var = tk.StringVar()
l = tk.Label(window,textvariable=var)
l.pack()
on_hit = False
started = False
var.set('Sample the noise first')


p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=1, rate=RATE,
	input=True, output=True,
	frames_per_buffer=CHUNK_SIZE)
r = array('h')
frame_id = 0

def normalize(snd_data):
	"Average the volume out"
	MAXIMUM = 16384
	times = float(MAXIMUM)/max(abs(i) for i in snd_data)
	r = array('h')
	for i in snd_data:
		r.append(int(i*times))
	return r

def hit_me():
	global on_hit,started,sampled
	if on_hit == False:
		on_hit = True
		started = True
		var.set('Press the button to close recording')
		idle_task()
	else:
		on_hit = False
		end_task()
		

def idle_task():
	global on_hit,started,r,frame_id
	if not started or on_hit:
		snd_data = array('h', stream.read(CHUNK_SIZE))
		if byteorder == 'big':
			snd_data.byteswap()
		r.extend(snd_data)
		#print len(r)
		SKIP_SIZE = 20
		if len(r)>=WIN_LEN and frame_id%SKIP_SIZE==0:
			#save to file
			t1 = time.time()
			sample_width = p.get_sample_size(FORMAT)
			#r = normalize(r)
			data = r[-WIN_LEN:]
			#print 'data size:',len(data)
			data = pack('<' + ('h'*len(data)), *data)
			wf = wave.open(path, 'wb')
			wf.setnchannels(1)
			wf.setsampwidth(sample_width)
			wf.setframerate(RATE)
			wf.writeframes(data)
			wf.close()
			os.system('sox {} {} noisered tmp/noise.prof 0.25'.format(path,clean_path))
			predict()
			print time.time()-t1
		frame_id += 1
		window.after(0,idle_task)

def predict():
	(rate, sig) = wavfile.read(clean_path)
	if (sig.ndim > 1):
		sig = sig[:, 0]  # pick mono-acoustic track
	#print "sig len",len(sig)
	mean = np.mean(sig)
	rms = np.sqrt(np.sum((sig-mean)*(sig-mean))/len(sig))
	print rms
	pr = model.predict(rate,sig,group_list)
	if rms<20:
		pr = '_'
	print pr
	sendMessageToSmartBody("sb scene.getDiphoneManager().setPhonemesRealtime('foo', '{}')".format(pr))


def end_task():
	print 'finished'

def sample_noise():
	nr = array('h')
	snd_data = array('h', stream.read(RATE)) #take 1s of noise sample
	if byteorder == 'big':
		snd_data.byteswap()
	nr.extend(snd_data)
	sample_width = p.get_sample_size(FORMAT)
	#r = normalize(r)
	#save to file
	data = nr
	data = pack('<' + ('h'*len(data)), *data)
	wf = wave.open('tmp/noise.wav', 'wb')
	wf.setnchannels(1)
	wf.setsampwidth(sample_width)
	wf.setframerate(RATE)
	wf.writeframes(data)
	wf.close()
	os.system('sox tmp/noise.wav -n noiseprof tmp/noise.prof')
	var.set('Press the button to start recording')

b1 = tk.Button(window,text='sample noise',width=15,height=2,command=sample_noise)
b1.pack()
b2 = tk.Button(window,text='start',width=15,height=2,command=hit_me)
b2.pack()
#window.after(1000, task)
window.mainloop()
