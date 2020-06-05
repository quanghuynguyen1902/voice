import os, numpy as np
import pickle
import hmmlearn.hmm as hmm
from sklearn.cluster import KMeans
import librosa
from sklearn.model_selection import train_test_split
# np.random.seed(13)
import math
# import pickle, os
import numpy as np
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import messagebox
import pygame
from pydub import AudioSegment
import ffmpeg
import pyaudio
import wave
from base64 import b64decode

class_names = ['nguoi', 'toi', 'mot', 'khong']

# build data
path = 'data/'
# path = '/Users/bangdo/code/school/speech_processing/speech_processing/test/trimmed'
def get_mfcc(filename):
	y, sr = librosa.load(filename) # read .wav file
	hop_length = math.floor(sr*0.010) # 10ms hop
	win_length = math.floor(sr*0.025) # 25ms frame
	# mfcc is 13 x T matrix
	mfcc = librosa.feature.mfcc(
		y, sr, n_mfcc=12, n_fft=1024,
		hop_length=hop_length, win_length=win_length)
	# substract mean from mfcc --> normalize mfcc
	mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
	# delta feature 1st order and 2nd order
	delta1 = librosa.feature.delta(mfcc, order=1)
	delta2 = librosa.feature.delta(mfcc, order=2)
	# X is 36 x T
	X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
	# return T x 36 (transpose of X)
	return X.T # hmmlearn use T x N matrix

def read_data(path_ = path):
	X, y = {}, {}
	for idx, cln in enumerate(class_names):
		files = [os.path.join(path_, cln, f) for f in os.listdir(os.path.join(path_, cln))]
		mfcc = [get_mfcc(file) for file in files]
		label = [idx for i in range(len(mfcc))]
		X.update( {cln: mfcc} )
		y.update( {cln: label} )
	return X, y


def load_model():
	model = {}
	model = pickle.load(open('model/models.pk', 'rb'))
	return model


def test_one_file(file_):
	record_mfcc = get_mfcc(file_)
	model = load_model()
	print(model)
	scores = [model[cname].score(record_mfcc) for cname in class_names]
	pred = np.argmax(scores)
	print(class_names[pred])


def detect_leading_silence(sound, silence_threshold=-42.0, chunk_size=10):
	'''
	sound is a pydub.AudioSegment
	silence_threshold in dB
	chunk_size in ms

	iterate over chunks until you find the first one with sound
	'''
	trim_ms = 0 # ms

	assert chunk_size > 0 # to avoid infinite loop
	while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
		trim_ms += chunk_size

	return trim_ms

def record():
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 22050
	RECORD_SECONDS = 2
	WAVE_OUTPUT_FILENAME = "record.wav"

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
					channels=CHANNELS,
					rate=RATE,
					input=True,
					frames_per_buffer=CHUNK)

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

def play():    
	filename = 'record.wav'
	pygame.init()
	pygame.mixer.init()
	sounda = pygame.mixer.Sound(filename)
	sounda.play()
	#winsound.PlaySound(filename, winsound.SND_FILENAME)
	
	
def playtrimmed():    
	filename = 'trimmed.wav'
	pygame.init()
	pygame.mixer.init()
	sounda = pygame.mixer.Sound(filename)
	sounda.play()
#     winsound.PlaySound(filename, winsound.SND_FILENAME)

def trim(record_path='', name = 'trimmed.wav'):
	input_path = record_path + 'record.wav'
	output_path = record_path + name
	sound = AudioSegment.from_file(input_path, format="wav")
	start_trim = detect_leading_silence(sound)
	end_trim = detect_leading_silence(sound.reverse())
	duration = len(sound)
	trimmed_sound = sound[start_trim:duration-end_trim]    
	trimmed_sound.export(output_path, format="wav")


def predict_new():
	models = load_model()
	data = get_mfcc('trimmed.wav')
	scores = {cname : model.score(data) for cname, model in models.items()}            
	pred_name = max(scores, key=lambda key: scores[key])
	messagebox.showinfo("result", pred_name)




def gui():
	window = tk.Tk()
	window.geometry("600x500")
	window.title("Speech recognition")

	frame0 = tk.Frame(master=window)
	frame0.pack()

	frame4 = tk.Frame(master=window)
	frame4.pack()

	frame1 = tk.Frame(master=window)
	frame1.pack()

	frame2 = tk.Frame(master=window)
	frame2.pack()

	frame3 = tk.Frame(master=window)
	frame3.pack()





	label = tk.Label(master=frame0, text="Speech recognition")
	label.pack(padx=5, pady=10)

	btn_record = tk.Button(master=frame1, width=13, height=2, text="Record", command=record)
	btn_record.pack(side=tk.LEFT, padx=5, pady=5)

	btn_playback = tk.Button(master=frame2, width=13, height=2, text="Playback", command=play)
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

	btn_predict = tk.Button(master=frame3, width=13, height=2, text="Predict", command=predict_new)
	btn_predict.pack(side=tk.LEFT, padx=5, pady=5)



	lb = tk.Frame(master = window)
	lb.pack()

	btn_playback = tk.Button(master=lb, width=5, height=2, text="Người", command= lambda: retrain('nguoi'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)
	btn_playback = tk.Button(master=lb, width=5, height=2, text="Không", command= lambda: retrain('khong'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)
	btn_playback = tk.Button(master=lb, width=5, height=2, text="một", command= lambda: retrain('mot'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)
	btn_playback = tk.Button(master=lb, width=5, height=2, text="tôi", command= lambda: retrain('tôi'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)
	btn_playback = tk.Button(master=lb, width=5, height=2, text="Bệnh nhân", command= lambda: retrain('benh_nhan'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

	window.mainloop()


def main():

	model = load_model()

	gui()


def trimxxx(pathx):

	for cln in class_names:
		files = [f for f in os.listdir(os.path.join(pathx, cln))]
		print(files)
		for f in files:
			input_path = pathx + cln + '/' + f
			output_path = pathx + 'trimmed/' + cln + '/' + f
			sound = AudioSegment.from_file(input_path, format="wav")
			start_trim = detect_leading_silence(sound)
			end_trim = detect_leading_silence(sound.reverse())
			duration = len(sound)
			trimmed_sound = sound[start_trim:duration-end_trim]    
			trimmed_sound.export(output_path, format="wav")


if __name__ == '__main__':

	main()


