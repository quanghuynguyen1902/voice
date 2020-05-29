import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import hmmlearn.hmm as hmm
from pydub.playback import play
import os
import pickle
from pydub import AudioSegment, silence
import noisereduce as nr
import math
from sklearn.cluster import KMeans


class ASR():
    def __init__(self):
        self.mapping = ['mot', 'khong', 'nguoi', 'toi']
        self.hmm = pickle.load(open('model/models.pk', 'rb'))
        self.n_sample=75
    
    def clustering(X, n_clusters=10):
        kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
        kmeans.fit(X)
        print("centers", kmeans.cluster_centers_.shape)
        return kmeans  

    def softmax_stable(self, Z):
        """
        Compute softmax values for each sets of scores in Z.
        each column of Z is a set of score.    
        """
        e_Z = np.exp(Z - np.max(Z))
        A = e_Z / e_Z.sum()
        return A

    def record_sound(self, filename, duration=1, fs=44100, play=False):
        print('Recording...')
        # sd.play( np.sin( 2*np.pi*940*np.arange(fs)/fs )  , samplerate=fs, blocking=True)
        # sd.play( np.zeros( int(fs*0.2) ), samplerate=fs, blocking=True)
        data = sd.rec(frames=duration*fs, samplerate=fs, channels=1, blocking=True)
        if play:
            sd.play(data, samplerate=fs, blocking=True)
        sf.write(filename, data=data, samplerate=fs)

    def record_data(self, prefix, n=10, start=0, duration=1):
        print('Recording {} {} times'.format(prefix, n))
        for i in range(n):
            print('{}_{}.wav'.format(prefix, i+start))
            self.record_sound('train/{}/{}_{}.wav'.format(prefix, prefix, i+start), duration=duration)
            if i % 5 == 4:
                input("Press Enter to continue...")
    
    def noise_cancel(self, filename='test.wav'):
        data, fs = librosa.load(filename)
        reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=data)
        sf.write(filename, data=reduced_noise, samplerate=fs)
                
    def get_mfcc(self, filename):
        y, sr = librosa.load(filename, sr=None)
        hop_length = math.floor(sr*0.010) # 10ms hop
        win_length = math.floor(sr*0.025)
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
        kmeans = self.clustering(all_vectors, n_clusters=21)
        t = kmeans.predict(X.T).reshape(-1,1)
        return t

    def asr(self):
        # self.record_sound('test.wav')
        mfcc = self.get_mfcc('test.wav')
        score = []
        for i in range(len(self.mapping)):
            score.append(self.hmm[i].score(mfcc))
        res = self.mapping[score.index(max(score))]
        print('predict: {}'.format(res))
        return res
    def listen(self):
        while True:
            self.record_sound('record.wav', duration=5)
            myaudio = AudioSegment.from_wav('record.wav')
            audios = silence.split_on_silence(myaudio, min_silence_len=300, silence_thresh=-32, keep_silence=400)
            if audios: break
        return audios