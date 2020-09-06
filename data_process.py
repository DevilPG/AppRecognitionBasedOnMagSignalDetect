# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:12:40 2020

@author: WHX
"""
import math 
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)

PCA_Main_Features=[]

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
    # frames *= winfunc(frame_length)

    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    complex_frames = np.fft.fft(frames)
    
    # print(complex_frames[0])
    mag_frames = np.absolute(complex_frames)
    # Compute power spectrum
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
    return mag_frames[:,:NFFT]

def pca(X,k):
    sample_num, feature_num = X.shape
    mean = np.array([np.mean(X[:,i]) for i in range(feature_num)])
    norm_X=X-mean
    scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
    fea_val,fea_vec=np.linalg.eig(scatter_matrix)
    fea_pairs = [(np.abs(fea_val[i]),fea_vec[:,i]) for i in range(feature_num)]
    fea_pairs.sort(reverse=True)
    features=np.array([fea[1] for fea in fea_pairs[:k]])
    # PCA_Main_Features = np.transpose(features)
    data = np.dot(norm_X,np.transpose(features))
    return data,np.transpose(features)

def trained_pca(X,main_feature_matrix):
    sample_num, feature_num = X.shape
    mean = np.array([np.mean(X[:,i]) for i in range(feature_num)])
    norm_X=X-mean
    data = np.dot(norm_X,main_feature_matrix)
    return data

def dataprocess(filename):
    with open(filename,"r") as f:
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
    sample_rate=100
    # print(M_normal[0:50])
    # print(np.absolute(np.fft.fft(M_normal[0:50])))
    mag_spec=STFT(M_normal,sample_rate,0.5,0.2,np.hamming,26)
    return mag_spec
    

if __name__ == '__main__':
    # with open("rawdata/Netease.txt","r") as f:
    #     datalist = f.readlines()
    
    # n=len(datalist)-1
    # X=[]
    # Y=[]
    # Z=[]
    # M=[]
    # M_normal=[]
    # sumx=0
    # sumy=0
    # sumz=0
    # avgx=0
    # avgy=0
    # avgy=0
    
    # for i in range(1,n+1):
    #     datalist[i].strip('\n')
    #     thislist=datalist[i].split(' ')
    #     X.append(float(thislist[0]))
    #     Y.append(float(thislist[1]))
    #     Z.append(float(thislist[2]))
    #     sumx+=float(thislist[0])
    #     sumy+=float(thislist[1])
    #     sumz+=float(thislist[2])
    
    # avgx=sumx/n
    # avgy=sumy/n
    # avgz=sumz/n
    # for i in range(0,n):
    #     X1=X[i]-avgx
    #     Y1=Y[i]-avgy
    #     Z1=Z[i]-avgz
    #     M.append(math.sqrt(X1*X1+Y1*Y1+Z1*Z1))
    # M_max=max(M)
    # M_min=min(M)
    # for i in range(0,n):
    #     M_normal.append((M[i]-M_min)/(M_max-M_min))
        
    # sample_rate=10
    
    # mag_spec=STFT(M_normal,sample_rate,0.5,0.2,np.hamming,32)
    
    
            
    # with open("results/M_spec_total.txt","a") as f4:
    #     for i in range(0,len(mag_spec)):
    #         f4.write(str(mag_spec[i])+'\n')
    
    M1=dataprocess("data1/Chorme.txt")
    M2=dataprocess("data1/Email.txt")
    M3=dataprocess("data1/PPT.txt")
    # M4=dataprocess("rawdata/PPT.txt")
    # M5=dataprocess("rawdata/Netease.txt")
    
    # x1=0
    # for i in range(len(M1)):
    #     x1+=M1[i][0]
    # x1=x1/len(M1)
  
    pcadata=np.concatenate((M1,M2,M3),axis=0)
    # print(pcadata)
    # print(pcadata.shape)
    pca_mag,PCA_Main_Features = pca(np.array(pcadata), 10)
    # print(PCA_Main_Features.shape)
    
    M1 = M1[0:int((len(M1)/10))*10,:]
    M2 = M2[0:int((len(M2)/10))*10,:]
    M3 = M3[0:int((len(M3)/10))*10,:]
    M11=[]
    M22=[]
    M33=[]
    print(M1)
    
    for k in range(int(len(M1)/10)):
        M11.append(trained_pca(M1[k:k+10,:], PCA_Main_Features))
        
    for k in range(int(len(M2)/10)):
        M22.append(trained_pca(M2[k:k+10,:], PCA_Main_Features))
        
    for k in range(int(len(M3)/10)):
        M33.append(trained_pca(M3[k:k+10,:], PCA_Main_Features))
        
       
        
    # M1=trained_pca(M1, PCA_Main_Features)
    # M2=trained_pca(M2, PCA_Main_Features)
    # M3=trained_pca(M3, PCA_Main_Features)
    # M4=trained_pca(M4, PCA_Main_Features)
    # M5=trained_pca(M5, PCA_Main_Features)
    
    
    # with open("results/pca_mainfeature_matrix1.txt","w") as f3:
    #     for i in range(0,len(PCA_Main_Features)):
    #         for j in range(0,len(PCA_Main_Features[0])):
    #             f3.write(str(PCA_Main_Features[i][j])+' ')
    #         f3.write("\n")
    
    with open("data1/chorm_result.txt","w") as f3:
        for i in range(0,len(M11)):
            for j in range(0,10):
                f3.write(str(M11[i][j])+'\n')
    with open("data1/email_result.txt","w") as f3:
        for i in range(0,len(M22)):
            for j in range(0,10):
                f3.write(str(M22[i][j])+'\n')
    with open("data1/ppt_result.txt","w") as f3:
        for i in range(0,len(M33)):
            for j in range(0,10):
                f3.write(str(M33[i][j])+'\n')
    
    
   
    
    
    
    
    # plt.rcParams['figure.figsize'] = (12.0, 8.0)
    # plt.rcParams['savefig.dpi'] = 500 #图片像素
    # plt.rcParams['figure.dpi'] = 500 #分辨率
    
#     plt.plot(M,label="Aggregated Signal")
#     plt.xlabel("Time", fontsize=16, horizontalalignment="center")
#     plt.ylabel("Magnitude", fontsize=16, horizontalalignment="center")
#     plt.legend()
#     plt.savefig('chorm-M-Signal.png', dpi=200)
#     plt.show()
    
#     plt.plot(M_normal,label="Normalized Signal",color='g')
#     plt.xlabel("Time", fontsize=16, horizontalalignment="center")
#     plt.ylabel("Magnitude(normalized)", fontsize=16, horizontalalignment="center")
#     plt.legend()
#     plt.savefig('chorm-M_Normal-Signal.png', dpi=200)
#     plt.show()
    
#     plt.plot(X,ls='-',label="X-axis",color='b')
#     plt.plot(Y,ls='--',label="Y-axis",color='c')
#     plt.plot(Z,ls='-.',label="Z-axis",color='g')
#     plt.xlabel("Time", fontsize=16, horizontalalignment="center")
#     plt.ylabel("Magnitude", fontsize=16, horizontalalignment="center")
#     plt.legend()
#     plt.savefig('chorm-XYZ-Signal.png', dpi=200)
#     plt.show()
    
#     # plt.specgram(M_normal,NFFT=32,Fs=10,noverlap=8,cmap='Oranges')        
#     # plt.savefig('chorm-M-Spec1.png', dpi=200)
    
    # plt.contourf(M3.T,cmap='hot')
    # plt.show()
#     plt.savefig('chorm-M-Spec.png', dpi=200)
    
    

        
    
# with open("chorm-M.txt","w") as f1:
#     for i in range(0,n):
#         f1.write(str(M[i])+'\n')
        
# with open("chorm-M_normal.txt","w") as f2:
#     for i in range(0,n):
#         f2.write(str(M_normal[i])+'\n')
    

# with open("chorm-M_spec.txt","w") as f3:
#         for i in range(0,len(mag_spec)):
#             f3.write(str(mag_spec[i])+'\n')

# with open("chorm-M_spec_PCA.txt","w") as f3:
#         for i in range(0,len(pca_mag)):
#             f3.write(str(pca_mag[i])+'\n')