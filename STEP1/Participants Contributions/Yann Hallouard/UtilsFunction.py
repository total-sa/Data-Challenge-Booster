import os
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Input, Dense, BatchNormalization, Conv3D, MaxPooling3D, Flatten, UpSampling3D, \
    Reshape, Lambda, Add ,\
    Dropout, LeakyReLU, Activation, AveragePooling3D, PReLU, Softmax, Multiply
from keras.models import Model, Sequential
from PIL import Image
from google.colab.patches import cv2_imshow
import cv2
from sklearn.utils import shuffle
from tqdm import tqdm_notebook, tqdm
import pickle
from skimage.transform import resize, rescale
from skimage.io import imread
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema


def import_data(path='', ids=None, size=(256, 256, 3)):

    X = np.ones((len(ids), size[0], size[1], size[2]))
    count = 0
    for id in ids:
      if count%500==0:
        print('%f pourcent already charged'%(count/len(ids)*100))
      im = plt.imread(path+id)
      X[count] = np.array(im)[:size[0], :size[1], :size[2]]
      count+=1
    return X

def import_edge_data(path='', ids=None, size=(256, 256, 3)):

    X = np.ones((len(ids), size[0], size[1], size[2]))
    count = 0
    for id in ids:
      if count%500==0:
        print('%f pourcent already charged'%(count/len(ids)*100))
      img = cv2.imread(path+id)
      s = img.shape
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(s[0], s[1], 1)
      edges1 = cv2.Canny(gray, 50, 50 , apertureSize = 3).reshape(s[0], s[1], 1)
      edges2 = cv2.Canny(gray, 50, 200, apertureSize = 3).reshape(s[0], s[1], 1)
      edges3 = cv2.Canny(gray, 50, 500, apertureSize = 3).reshape(s[0], s[1], 1)

      both = np.concatenate((edges1, edges2, edges3))

      X[count] = both[:size[0], :size[1], :size[2]]
      count+=1
    return X


class DataBuilding() :
  
  def __init__(self, image_filenames, labels) :
    self.image_filenames = image_filenames
    self.labels = labels
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, size=None, name='train_%d.pickle', path='./Challenge_Data/TRAIN_Resized/', image_shape=(64, 64, 3)):
    x = self.image_filenames#[:size]
    y = self.labels#[:size]

    new_x = np.concatenate((x[np.where(y==0)][:int(size/2)], 
                           x[np.where(y==1)][:int(size/2)]), 
                           axis=0)
    new_y = np.concatenate((y[np.where(y==0)][:int(size/2)], 
                           y[np.where(y==1)][:int(size/2)]), 
                           axis=0)
    
    x, y = shuffle(new_x, new_y, random_state=0)
    
    
    X_train = np.array([
               resize(imread(path + str(file_name)), image_shape)
               for file_name in tqdm(x)])

    with open(name%(size), 'wb') as f:
      pickle.dump([X_train, y], f, protocol=4)

    return [X_train, y]

  class features_extraction():
    
    def __init__(self, X=None):
        self.X = X.reshape(X.shape[0], X.shape[1]*X.shape[2], X.shape[3])
        self.N = X.shape[0]

        self.extracted_features = None
        
        
        
        
    def calcul_features(self, features=['global_level',
                                        'grey_level', 
                                        'R_mean', 
                                        'G_mean', 
                                        'B_mean', 
                                        'R_quantille25',
                                        'G_quantille25', 
                                        'B_quantille25', 
                                        'R_quantille75', 
                                        'G_quantille75', 
                                        'B_quantille75', 
                                        'R_std',
                                        'G_std', 
                                        'B_std']):
        
        
        dico = {'global_level': self.Global_level,
                'grey_level': self.Grey_level, 
                'R_mean': self.R_mean, 
                'G_mean': self.G_mean, 
                'B_mean': self.B_mean, 
                'R_quantille25': self.R_quant25,
                'G_quantille25': self.G_quant25, 
                'B_quantille25': self.B_quant75, 
                'R_quantille75': self.R_quant75, 
                'G_quantille75': self.G_quant75, 
                'B_quantille75': self.B_quant75, 
                'R_std': self.R_std,
                'G_std': self.G_std, 
                'B_std': self.B_std}
        
        result = pd.DataFrame(np.ones((self.N,len(features))))
        result.columns = features
        
        for feature in tqdm(features):
            result[feature] = dico.get(feature, self.fonctAutre)()
             
        self.extracted_features = result
        
        return(result)
    
    
        
    def fonctAutre(self):
        print("Cette feature n'existe pas")
    
    def Global_level(self):
      grey_level = []
      for i in range(self.N):
        img = self.X[i]

        grey_level.append(np.mean(img))
          
      return np.array(grey_level)
    
    def Grey_level(self):
      grey_level = []
      for i in range(self.X.shape[0]):
        img = self.X[i]
        grey = np.ones(img.shape)/np.sqrt(3)

        dist = np.ones((img.shape[0])) - (np.sum(img*grey, axis=1)/\
                                          np.sqrt(np.sum((img)**2, axis=1)))**2

        grey_share = len(np.where((dist<2.09335693e-02)*\
                                  (np.mean(img, axis=1)>0.1)*\
                                  (np.mean(img, axis=1)<0.9))[0])/img.shape[0]

        grey_level.append(grey_share)
          
      return np.array(grey_level)
    
    def R_mean(self):
      r_mean = []
      for i in range(self.N):
        img = self.X[i,:,0]
        r_mean.append(np.mean(img))
        
      return np.array(r_mean)
        
    def G_mean(self):
      g_mean = []
      for i in range(self.N):
        img = self.X[i,:,1]
        g_mean.append(np.mean(img))
            
      return np.array(g_mean)

    def B_mean(self):
      b_mean = []
      for i in range(self.N):
            img = self.X[i,:,2]
            b_mean.append(np.mean(img))
            
      return np.array(b_mean)

    def R_quant25(self):
      r_quant25 = []
      for i in range(self.N):
            img = np.sort(self.X[i,:,0])
            r_quant25.append(img[int(len(img)*0.25)])
            
      return np.array(r_quant25)

    def G_quant25(self):
      g_quant25 = []
      for i in range(self.N):
            img = np.sort(self.X[i,:,1])
            g_quant25.append(img[int(len(img)*0.25)])
            
      return np.array(g_quant25)

    def B_quant25(self):
      b_quant25 = []
      for i in range(self.N):
            img = np.sort(self.X[i,:,2])
            b_quant25.append(img[int(len(img)*0.25)])
            
      return np.array(b_quant25)

    def R_quant75(self):
      r_quant75 = []
      for i in range(self.N):
            img = np.sort(self.X[i,:,0])
            r_quant75.append(img[int(len(img)*0.75)])
            
      return np.array(r_quant75)

    def G_quant75(self):
      g_quant75 = []
      for i in range(self.N):
            img = np.sort(self.X[i,:,1])
            g_quant75.append(img[int(len(img)*0.75)])
            
      return np.array(g_quant75)

    def B_quant75(self):
      b_quant75 = []
      for i in range(self.N):
            img = np.sort(self.X[i,:,2])
            b_quant75.append(img[int(len(img)*0.75)])
            
      return np.array(b_quant75)

    def R_std(self):
      r_std = []
      for i in range(self.N):
            img = self.X[i,:,0]
            r_std.append(np.std(img))
            
      return np.array(r_std)

    def G_std(self):
      g_std = []
      for i in range(self.N):
            img = self.X[i,:,1]
            g_std.append(np.std(img))
            
      return np.array(g_std)

    def B_std(self):
      b_std = []
      for i in range(self.N):
            img = self.X[i,:,2]
            b_std.append(np.std(img))
            
      return np.array(b_std)