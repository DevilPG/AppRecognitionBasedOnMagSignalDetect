# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:12:40 2020

@author: WHX
"""
import math 
import matplotlib.pyplot as plt
import numpy as np


def STFT(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, winfunc=np.hamming, NFFT=512):
    # Calculate the number of frames from the signal
    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # zero padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad signal to make sure that all frames have equal number of samples 
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z)

    # Slice the signal into frames from indices
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    frames *= winfunc(frame_length)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    complex_frames = np.fft.rfft(frames, NFFT)
    mag_frames = np.absolute(complex_frames)
    # Compute power spectrum
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
    return mag_frames

if __name__ == '__main__':
    with open("sensor.txt","r") as f:
        datalist = f.readlines()
    
    n=len(datalist)-1
    X=[]
    Y=[]
    Z=[]
    M=[]
    M_normal=[]
    sumx=0
    sumy=0
    sumz=0
    avgx=0
    avgy=0
    avgy=0
    
    for i in range(1,n+1):
        datalist[i].strip('\n')
        thislist=datalist[i].split(' ')
        X.append(float(thislist[0]))
        Y.append(float(thislist[1]))
        Z.append(float(thislist[2]))
        sumx+=float(thislist[0])
        sumy+=float(thislist[1])
        sumz+=float(thislist[2])
    
    avgx=sumx/n
    avgy=sumy/n
    avgz=sumz/n
    for i in range(0,n):
        X1=X[i]-avgx
        Y1=Y[i]-avgy
        Z1=Z[i]-avgz
        M.append(math.sqrt(X1*X1+Y1*Y1+Z1*Z1))
    M_max=max(M)
    M_min=min(M)
    for i in range(0,n):
        M_normal.append((M[i]-M_min)/(M_max-M_min))
        
    sample_rate=50
    
    mag_spec=STFT(M_normal,sample_rate,1,0.1,np.hamming,64)
    
    print(mag_spec)
    
    
    
    # plt.rcParams['figure.figsize'] = (12.0, 8.0)
    # plt.rcParams['savefig.dpi'] = 500 #图片像素
    # plt.rcParams['figure.dpi'] = 500 #分辨率
    
    # plt.plot(M,label="Aggregated Signal")
    # plt.xlabel("Time", fontsize=16, horizontalalignment="center")
    # plt.ylabel("Magnitude", fontsize=16, horizontalalignment="center")
    # plt.legend()
    # plt.savefig('M-Signal.png', dpi=200)
    # plt.show()
    
    # plt.plot(M_normal,label="Normalized Signal",color='g')
    # plt.xlabel("Time", fontsize=16, horizontalalignment="center")
    # plt.ylabel("Magnitude(normalized)", fontsize=16, horizontalalignment="center")
    # plt.legend()
    # plt.savefig('M_Normal-Signal.png', dpi=200)
    # plt.show()
    
    # plt.plot(X,ls='-',label="X-axis",color='b')
    # plt.plot(Y,ls='--',label="Y-axis",color='c')
    # plt.plot(Z,ls='-.',label="Z-axis",color='g')
    # plt.xlabel("Time", fontsize=16, horizontalalignment="center")
    # plt.ylabel("Magnitude", fontsize=16, horizontalalignment="center")
    # plt.legend()
    # plt.savefig('XYZ-Signal.png', dpi=200)
    # plt.show()
    
    plt.specgram(M_normal,NFFT=64,Fs=50,noverlap=32,cmap='Oranges')        
    plt.savefig('M-Spec1.png', dpi=200)
    
    plt.contourf(mag_spec.T,cmap='hot')
    plt.savefig('M-Spec.png', dpi=200)

        
    
# with open("M.txt","w") as f1:
#     for i in range(0,n):
#         f1.write(str(M[i])+'\n')
        
# with open("M_normal.txt","w") as f2:
#     for i in range(0,n):
#         f2.write(str(M_normal[i])+'\n')
    
# print(M)
# print([avgx,avgy,avgz])
# print([M_max,M_min])

with open("M_spec.txt","w") as f3:
        for i in range(0,len(mag_spec)):
            f3.write(str(mag_spec[i])+'\n')