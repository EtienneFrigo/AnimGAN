#Loading Libraries

from __future__ import print_function
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import math
import random
from keras.models import Sequential
from keras.layers import Dense, Reshape, Dropout, LSTM, Embedding, TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D 
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import mnist
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from keras.utils.generic_utils import Progbar

#Parameters

RUN_ID = "1"
NUM_EPOCHS = 200
BATCH_SIZE = 128
TENSORBOARD_DIR = './summaries' + RUN_ID

bvh_duration = 4
framerate = 1.0
nb_frame = int(bvh_duration / framerate)

NAMES = ['Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head', 'RightCollar', 'RightShoulder', 'RightElbow',
         'RightWrist', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightHip', 'RightKnee', 'RightAnkle',
         'RightToe', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe']

# use specific GPU
GPU = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

def mirror_rotations(rotations):
    """give the mirror image of a rotation (np array)"""
    F = rotations.shape[0]
    J = rotations.shape[1]
    
    modif = np.array([1,1,-1,-1])
    for f in range(F):
        for j in range(0,7):
            rotations[f][j] *= modif
        
        temp = rotations[f][7].copy()
        rotations[f][7]=rotations[f][11].copy()*modif
        rotations[f][11]=temp* modif
        
        temp = rotations[f][8].copy()
        rotations[f][8]=rotations[f][12].copy()* modif
        rotations[f][12]=temp* modif
        
        temp = rotations[f][9].copy()
        rotations[f][9]=rotations[f][13].copy()* modif
        rotations[f][13]=temp* modif
        
        temp = rotations[f][10].copy()
        rotations[f][10]=rotations[f][14].copy()*modif
        rotations[f][14]=temp* modif
        
        temp = rotations[f][15].copy()
        rotations[f][15]=rotations[f][19].copy()* modif
        rotations[f][19]=temp* modif
        
        temp = rotations[f][16].copy()
        rotations[f][16]=rotations[f][20].copy()* modif
        rotations[f][20]=temp* modif
        
        temp = rotations[f][17].copy()
        rotations[f][17]=rotations[f][21].copy()* modif
        rotations[f][21]=temp* modif
        
        temp = rotations[f][18].copy()
        rotations[f][18]=rotations[f][22].copy()* modif
        rotations[f][22]=temp* modif
        
    return rotations
	
def prepare_data(directory):
    """
    prepare the data to be feed in the network. Extract every frame of all the files.
    directory : name of the folder containing the motion_data. directory should be in the same folder as the algorithm.string
    """
    #Getting all the paths to the bvh files of the directory
    print("loading bvh file ...", end="\r")
    anim_paths = []
    
    current_dir = os.getcwd()
    motion_dir = os.path.join(current_dir, directory)
    for each in os.listdir(motion_dir):
        anim_paths.append(os.path.join(motion_dir, each))
    
    #Loading the bvh files and save them as file of a smaller duration
    motion_data = []
    
    for a in anim_paths :

        if not(a=='/home/dl-box/lab/GANs/Movement_GAN/motionDataSet_bvhFormat/.ipynb_checkpoints'): #some issues with an unexistaniming file
            try:
                new_data = BVH.load(filename=a)
                motion_data.append(BVH.load(filename=a))
            except ValueError :
                print ("on line",a)
                
    print("loading bvh files : DONE",end="\r")
    
    motion_data = np.array(motion_data)
    
    #extracting all the rotations of all frames from the dataset
    postures_data = []
    
    for m in motion_data :      
        for f in range(m[0].rotations.shape[0]):
            insert = np.array(m[0].rotations[f])
            if insert.shape==(23,4):
                postures_data.append(insert)
    
    postures_data = np.array(postures_data)
    
    print("extracting postures rotations : DONE", end="\r")
    
    return postures_data
	
def add_noise(rots, level):
    """adds random noise to a rotation matrix."""
    
    noised_rots = [[np.random.uniform(-level,level,4)]*23]*rots.shape[0]
    noised_rots = Quaternions(np.array(noised_rots))
    
    return rots+noised_rots
	
#PREPARE DATA
postures_data = prepare_data('/home/dl-box/lab/GANs/Movement_GAN/motionDataSet_bvhFormat')

print("input data shape : ",postures_data.shape)

postures_data = postures_data.reshape((postures_data.shape[0], 1) + postures_data.shape[1:])
print(postures_data.shape)

def generator_Dense1():
    
    net = Sequential()
    
    net.add(Dense(input_dim=4, units=4))
    net.add(Activation('tanh'))
    
    net.add(Dense(12))
    net.add(Activation('tanh'))
    
    net.add(Dense(36))
    net.add(Activation('tanh'))
    
    net.add(Dense(92))
    net.add(Activation('tanh'))
    
    net.add(Reshape((1,23,4)))
    
    return net
	
def discriminator_Dense1():
    
    net = Sequential()
    input_shape = (1,23,4)
    
    net.add(Flatten(input_shape=input_shape))
    net.add(Dense(92))
    net.add(Activation(LeakyReLU()))
    net.add(Activation('relu'))
    
    net.add(Dense(36))
    net.add(Activation(LeakyReLU()))
    net.add(Activation('relu'))
    
    net.add(Dense(1))
    net.add(Activation('sigmoid'))
    
    return net
	
def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model
	
def posture_chain(postures_list):
    nb_frames = len(postures_list)
    positions = np.array([[[0.0]*3]*23]*nb_frames)
    orients = Quaternions(np.array([1.0, 0.0, 0.0, 0.0]*23))
    offsets = np.array([[ 0.00000,  0.00000,  0.00000],
                [ 0.00000,  10.3548,  0.00650000],
                 [ 0.00000,  10.04358, -0.0132000],
                 [ 0.00000,  10.03937,  0.00000],
                 [ 0.00000,  10.03831,  0.00000],
                 [ 0.00000,  10.56940,  0.00000],
                 [ 0.00000 , 9.71100, -0.0280000],
                 [-3.15720,  10.03675,  0.00000],
                 [-10.44825,  0.00000,  0.00000],
                 [ 0.00000, -30.17078,  0.00000],
                 [ 0.00000, -20.56593,  0.00000],
                 [ 3.15720,  10.03675,  0.00000],
                 [ 10.44825,  0.00000,  0.00000],
                 [ 0.00000, -30.17078,  0.00000],
                 [ 0.00000, -20.56593,  0.00000],
                 [-8.36560, -0.0321000, -0.00320000],
                 [ 0.00000, -40.39800,  0.00310000],
                 [ 0.00000, -40.28828,  0.00520000],
                 [ 0.00000, -4.91550,  10.08263],
                 [ 8.36560, -0.0321000, -0.00320000],
                 [ 0.00000, -40.39800,  0.00310000],
                 [ 0.00000, -40.28828,  0.00520000],
                 [ 0.00000, -4.91550,  10.08263]])
    parents = np.array([-1,  0,  1,  2,  3,  4,  5 , 4,  7,  8 , 9, 4, 11, 12, 13, 0, 15, 16, 17,  0, 19, 20, 21])
    generated_anim = Animation.Animation(Quaternions(postures_list),
                                         positions,
                                         orients,
                                         offsets,
                                         parents
                                        )
    return generated_anim
	
def postures_preprocess(posture_batch, epsilon):
    """take an np.array of posture and replace the zeros by a very low value.
    This is suppose to avoid errors when feeding the generated postures to the generator"""
    
    S = posture_batch.shape[0]
    J = posture_batch.shape[2]
    
    result = posture_batch
    
    for s in range(S):
        for j in range(J):
            for i in range(4):
                if abs(result[s][0][j][i])<=epsilon :
                    result[s][0][j][i]=epsilon
    return result
	
def posture_postprocess(posture_list):
    """put all the generated posture (np array) in the same orientation before displaying them"""
    
    result = posture_list.copy()
    
    for i in range(posture_list.shape[0]):
        result[i][0][0] = np.array([0.9987335, -0.05031297, 0.00000001, 0.00000001])
        
    return result    
	
def make_fake(nb_frames):
    """build a random rotation element"""
    
    rots = np.array([[[0.0]*4]*23]*nb_frames)
    
    for f in range(nb_frames):
        for j in range(23):
            for k in range (4):
                rots[f][j][k] = np.random.uniform(-1,1)
                
    return rots

def make_odd(rots_data, oddity):
    """make rotations from an np.array of postures but adds some unrealistic features to it.
    These features can be parametrized. Oddity allows to choose in which manner they will be change:
    - random : add noise an arm, a leg or the spine
    - inverse : reverse the rotations
    - static : set to default rotations an arm
    """
    S = rots_data.shape[0]
    J = rots_data.shape[2]
    result = rots_data.copy()
                
    if oddity == 'random':
        for s in range(S):
            R = random.randint(0,4)
            if R == 0:#spines
                modif = [0,1,2,3,4,5]
            if R == 1:#right arm
                modif = [6,8,9,10]
            if R == 2:#left arm
                modif = [11,12,13,14]
            if R == 3:#right lieg
                modif = [15,16,17,18]
            if R == 4:#left leg
                modif = [19,20,21,22]
            for j in modif:
                for q in range(4):
                    result[s][0][j][q]+=np.random.uniform(-0.5,0.5)
                    
    if oddity == 'inverse':
        for s in range(S):
            R = random.randint(0,4)
            if R == 0:#spines
                modif = [0,1,2,3,4,5]
            if R == 1:#right arm
                modif = [6,8,9,10]
            if R == 2:#left arm
                modif = [11,12,13,14]
            if R == 3:#right lieg
                modif = [15,16,17,18]
            if R == 4:#left leg
                modif = [19,20,21,22]        
            for j in modif:
                for q in range(1,4):
                    result[s][0][j][q]=-result[s][0][j][q]
                    
    if oddity == 'static':
        for s in range(S):
            R = random.randint(0,4)
            if R == 0:#spines
                modif = [0,1,2,3,4,5]
            if R == 1:#right arm
                modif = [6,8,9,10]
            if R == 2:#left arm
                modif = [11,12,13,14]
            if R == 3:#right lieg
                modif = [15,16,17,18]
            if R == 4:#left leg
                modif = [19,20,21,22]
            for j in modif:
                for q in range(4):
                    result[s][0][j][q]=0.0000001
        
    return result
	
def test_GAN(G_model, D_model, data_set, num_epoch, batch_size):
    
    X_train = np.random.permutation(data_set)
    X_train_reals = X_train[0:100000]
    print('X_train_reals shape: ', X_train_reals.shape)
    
    discriminator = D_model # building discriminator
    generator = G_model #building generator
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator) # generator + discriminator
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=False)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=False)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((batch_size, 4))
    
    progress_bar = Progbar(target=num_epoch)
    DG_losses = []
    D_losses = []
    samples_list = []
    
    for epoch in range(num_epoch):
        
        if len(D_losses) > 0:
            progress_bar.update(
                epoch,
                values=[
                    ('D_losses_mean', np.mean(D_losses[-5:], axis=0)),
                    ('DG_losses_mean', np.mean(DG_losses[-5:], axis=0))
                ]        
            )
        
        else:
            progress_bar.update(epoch)
            
        for index in range(int(X_train_reals.shape[0]/batch_size)):
            
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1,1,4)
            
            reals_batch = X_train_reals[index*batch_size:(index+1)*batch_size]
            fakes_batch = generator.predict(noise, verbose=0)
            
            
            fakes_batch = postures_preprocess(fakes_batch, 0.000001)
                  
            X_train = np.concatenate((reals_batch, fakes_batch))
            Y_train = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X_train, Y_train)
            
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1,1,4)
                
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)
            discriminator.trainable = True
            
            if index == 0:
                sample = fakes_batch[0]
                samples_list.append(sample)
                DG_losses.append(g_loss)
                D_losses.append(d_loss)
            
            if epoch % 10 == 0:
                generator.save_weights('generator_weights', True)
                discriminator.save_weights('discriminator_weights', True) 
    
    samples_list = np.array(samples_list)
    
    X_pred_noise = []
    for i in range(1000):
        X_pred_noise.append(np.random.uniform(-1,1,4))
    X_pred_noise = np.array(X_pred_noise)
    
    prediction_G_end = generator.predict(x=X_pred_noise, batch_size=batch_size, verbose=1, steps=None)
    prediction_D_end = discriminator.predict(x=prediction_G_end, batch_size=batch_size, verbose=1, steps=None)
    prediction_D_samples = discriminator.predict(x=samples_list, batch_size=batch_size, verbose=1, steps=None)
    
    show_samples = posture_postprocess(samples_list)
    show_samples = show_samples.reshape(show_samples.shape[0],23,4)
    show_samples = posture_chain(show_samples)
    BVH.save(filename = '/home/dl-box/lab/GANs/Posture_GAN/postures_dir'+RUN_ID+'/samples2.bvh',
             anim = show_samples,
             names = NAMES,
             frametime = 1/1)
    
    return [discriminator, generator, D_losses, DG_losses, samples_list, prediction_G_end, prediction_D_end, prediction_D_samples]
	
if not os.path.exists("postures_dir"+RUN_ID):
        os.makedirs("postures_dir"+RUN_ID)
		
[ADV_D, ADV_DG, D_losses, DG_losses, adv_samples_list, adv_prediction_G_end, adv_prediction_D_end, adv_prediction_D_samples] = test_GAN(generator_Dense1(), discriminator_Dense1(), postures_data, 1000, 128)

plt.plot(DG_losses)
plt.ylabel('DG_losses')
plt.show()

plt.plot(D_losses)
plt.ylabel('D_losses')
plt.show()

plt.plot(adv_prediction_D_end)
plt.ylabel('adv_prediction_D_end')
plt.show()

plt.plot(adv_prediction_D_samples)
plt.ylabel('adv_prediction_D_samples')
plt.show()

def generate(batch_size, pretty=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator_weights')
    if pretty:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator_weights')
        noise = np.zeros((batch_size*20, 100))
        for i in range(batch_size*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size*20)
        index.resize((batch_size*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        pretty_images = np.zeros((batch_size, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(batch_size)):
            idx = int(pre_with_index[i][1])
            pretty_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(pretty_images)
    else:
        noise = np.zeros((batch_size, 100))
        for i in range(batch_size):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save("images/generated_image.png")
	
#GENERATION
generate(batch_size = BATCH_SIZE, pretty = False)

