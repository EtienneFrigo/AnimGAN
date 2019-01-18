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
from keras.layers import Dense, Reshape, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional
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
DIR = '/home/dl-box/lab/GANs/Movement_GAN/' #working directory
NUM_EPOCHS = 1000
BATCH_SIZE = 64
DURATION = 3 #seconds
FRAMERATE = 1/12 #should be the fraction of a multiple of 12 (otherwise, may get some problem during the pre-processing)
NB_FRAMES = int(DURATION / FRAMERATE)
NOISE_SHAPE = (10,10) #shape of the noise use as input for the generator
NAMES = ['Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head', 'RightCollar', 'RightShoulder', 'RightElbow',
         'RightWrist', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightHip', 'RightKnee', 'RightAnkle',
         'RightToe', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'] #names of the 3Dmodel joints
NB_JOINTS = len(NAMES) #number of joints

# use specific GPU
GPU = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

#create the folder to the low resolution data
if not os.path.exists("lowFPS_"+str(RUN_ID)+"_"+str(DURATION)+"s_"+str(1/FRAMERATE)+"fps"):
        os.makedirs("lowFPS_"+str(RUN_ID)+"_"+str(DURATION)+"s_"+str(1/FRAMERATE)+"fps")

# create the folder to save the samples
if not os.path.exists("motion_samples_dir"+RUN_ID):
        os.makedirs("motion_samples_dir"+RUN_ID)
        
# create the folder to save the predictions
if not os.path.exists("motion_pred_dir"+RUN_ID):
        os.makedirs("motion_pred_dir"+RUN_ID)

def mirror_rotations(rotations):
    """give the mirror image of a rotation matrix (np array)"""
    
    F = rotations.shape[0] #number of frames
    J = rotations.shape[1] #number of joints
    result = rotations.copy()

    modif = np.array([1,1,-1,-1]) #to get the mirror image of Quaternions, you can reverse the last parameters (empirical)
    
    for f in range(F):
        
        for j in range(0,7):# mirror the spine and head
            rotations[f][j] *= modif
            
        #mirror Collars
        temp = rotations[f][7]
        result[f][7]=rotations[f][11]*modif
        result[f][11]=temp*modif
        #mirror Shoulders
        temp = rotations[f][8]
        result[f][8]=rotations[f][12]*modif
        result[f][12]=temp*modif
        #mirror Elbow
        temp = rotations[f][9]
        result[f][9]=rotations[f][13]*modif
        result[f][13]=temp*modif
        #mirror Wrists
        temp = rotations[f][10]
        result[f][10]=rotations[f][14]*modif
        result[f][14]=temp*modif
        #mirror Hips
        temp = rotations[f][15]
        result[f][15]=rotations[f][19]*modif
        result[f][19]=temp*modif
        #mirror Knees
        temp = rotations[f][16]
        result[f][16]=rotations[f][20]*modif
        result[f][20]=temp*modif
        #mirror Ankles
        temp = rotations[f][17]
        result[f][17]=rotations[f][21]*modif
        result[f][21]=temp*modif
        #mirror Toes
        temp = rotations[f][18]
        result[f][18]=rotations[f][22]*modif
        result[f][22]=temp*modif

    return result

def add_noise(rotations, level):
    """adds random noise to a rotation matrix np array. The noise will have a value in the range [-level ; level]"""
    
    S = rotations.shape[0] #size of the array
    F = rotations.shape[1] #number of frames
    J = rotations.shape[2] #number of joints
    result = rotations.copy()
    
    for s in range(S):
        for f in range(F):
            for j in range(J):
                for q in range(4):
                    result[s][f][j][q]+=np.random.uniform(-level,level)
    
    return result

def make_animation(generated_rots, original_anim):
    """make a proper animation object from rotations by using a model from the real data for static parameters (orients, offsets ...) """
    
    generated_anim = (Animation.Animation(Quaternions(generated_rots),
                                         original_anim.positions,
                                         original_anim.orients,
                                         original_anim.offsets,
                                         original_anim.parents
                                        ))
    return generated_anim

def make_fake_list(nb_motions, nb_frames, nb_joints):
    """build a totally random motion"""
    
    rots = np.array([[[[0.0]*4]*nb_joints]*nb_frames]*nb_motions)
    for s in range (nb_motions):
        for f in range(nb_frames):
            for j in range(nb_joints):
                for q in range (4):
                    rots[s][f][j][q] = np.random.uniform(-1,1)
                
    return rots

def make_odd(rots_data, oddity):
    """adds some unrealistic features to a rotations matrix data by modifying a random body part (spine, left arm ...).
    These features can be parametrized. Oddity allows to choose in which manner they will be change:
    - random : add noise an arm, a leg or the spine
    - inverse : reverse the rotations
    - static : set to default rotations"""
    
    S = rots_data.shape[0] # number of elements in the data set
    F = rots_data.shape[1] # number of frame per element
    J = rots_data.shape[2] # number of joint per frame
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
            for f in range(F):
                for j in modif:
                    for q in range(4):
                        result[s][f][j][q]+=np.random.uniform(-0.3,0.3)
                    
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
            for f in range(F):
                for j in modif:
                    for q in range(1,4):
                        result[s][f][j][q]=-result[s][f][j][q]
                    
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
            for f in range(F):
                for j in modif:
                    for q in range(4):
                        result[s][f][j][q]=0.0000001
        
    return result

def rotations_preprocess(rotations_batch, epsilon):
    """take an np.array of rotations and replace the zeros (that can cause problems with Quaternions) by a very low value.
    This is suppose to avoid errors when feeding the generated postures to the generator"""
    
    S = rotations_batch.shape[0] # number of elements in the data set
    F = rotations_batch.shape[1] # number of frame per element
    J = rotations_batch.shape[2] # number of joint per frame
    
    result = rotations_batch
    
    for s in range(S):
        for f in range(F):
            for j in range(J):
                for i in range(4):
                    if abs(result[s][f][j][i]) <= epsilon :
                        result[s][0][j][i] = epsilon
    return result

def motion_postprocess(motion):
    """put a the generated gesture in the same orientation before displaying it"""
    
    F = motion.shape[0] # number of elements in the data set
    J = motion.shape[1] # number of frame per element
    result = motion.copy()
    
    for f in range(0,F):
        result[f][0] = np.array([0.9987335, -0.05031297, 0.00000001, 0.00000001])
        
    return result

def prepare_data(directory, duration, framerate, shift):
    """prepare the data to be feed in the network
    directory : name of the folder containing the motion_data. directory should be in the same folder as the algorithm.string
    duration : duration (seconds) we want the data elements to be. float
    framerate : fps we want the data to be. float
    shift : when getting multiple samples from a file, indicate the time in between (if shift < duration : overlaping) 
    Every elements in the data is given a same duration and framerate"""

    #Getting all the paths to the bvh files of the directory
    print("loading bvh file ...", end="\r")
    bvh_paths = []
    current_dir = os.getcwd()
    motion_dir = os.path.join(current_dir, directory)
    
    for each in os.listdir(motion_dir):
        bvh_paths.append(os.path.join(motion_dir, each))
    
    #Loading the bvh files and save them as file of a smaller duration
    motion_data = []
    
    for i in bvh_paths :
        if not(i==DIR+'motionDataSet_bvhFormat/.ipynb_checkpoints'): #some issues with a non existing file
            try:
                new_data = BVH.load(filename=i)
                motion_data.append(BVH.load(filename=i))
            except ValueError :
                print ("on line",i)
                
    print("loading bvh files : DONE",end="\r")
    
    #Changing the animations' framerate by sampling the rotations and positions of the files
    lowFPS_data = []
    
    for m in motion_data :
        file_duration = m[0].rotations.shape[0]*m[2] #duration of the file (s)
        frame_skip = int(framerate/m[2]) #number of frame to skip to get the wanted duration
        end_frame = int(duration/m[2]) #frame to end one sample
        #we need to count how many samples we can extract from a single file to prceed the multi sampling
        nb_samples = 0
        r = 0
        while r + duration < file_duration:
            nb_samples += 1
            r += shift
        
        if(nb_samples > 0):  
            for sample in range(nb_samples):
                rots = []
                poss = []
            
                for k in range(sample*(shift*int(1/m[2])), sample*(shift*int(1/m[2]))+end_frame, frame_skip) :
                    rots.append(m[0].rotations[k])
                    poss.append(m[0].positions[k])
            
                new_rotations = Quaternions(np.array(rots))
                new_positions = np.array(poss)
                new_positions = np.array([[[0]*3]*23]*36)

                if new_rotations.shape == (36, 23):
                    new_anim = Animation.Animation(new_rotations, new_positions, m[0].orients, m[0].offsets, m[0].parents)
                    lowFPS_tuple = (new_anim, m[1], framerate)
                    lowFPS_data.append(lowFPS_tuple)
    
    print("lowering framerate : DONE", end="\r")
    
    return np.array(lowFPS_data)
	

#preparing data of 3 seconds long animations at 12 fps ; this step can take a few minutes depending on the datasetlow size

lowFPS_data = prepare_data('motionDataSet_bvhFormat', DURATION, FRAMERATE, 1)
print(f"lowFPS_{RUN_ID}_{DURATION}s_{1/FRAMERATE}fps")
print("preparing low_fps_data : DONE",end="\r")
 
#saving lowDPS-data in directory
for i in range (len(lowFPS_data)) :
     BVH.save(filename=DIR+"lowFPS_"+str(RUN_ID)+"_"+str(DURATION)+"s_"+str(1/FRAMERATE)+"fps/data_LFPS_"+str(i)+".bvh",
                anim=lowFPS_data[i][0],
                names=lowFPS_data[i][1],
                frametime=lowFPS_data[i][2]
            )
        
#extracting the roations from the lowFPS_data
print("extracting rotations ...",end="\r")
rots = []
for i in lowFPS_data :
    insert = np.array(i[0].rotations)
    rots.append(insert)
    rots.append(mirror_rotations(insert)) # add the mirrored rotations to get more data
rots_data = np.array(rots)
print("extracting rotations : DONE", end="\r")
print("input data shape : ",rots_data.shape)
	
def generator_model():
    """build the generator"""
     
    model = Sequential()
    #default activation layer for LSTM in Keras is 'tanh'
    model.add(LSTM(512,input_shape=NOISE_SHAPE,return_sequences=False, kernel_initializer='he_normal'))
    model.add(Dense(NB_FRAMES*NB_JOINTS*4))
    model.add(Activation('tanh'))
    model.add(Reshape(target_shape=(NB_FRAMES,NB_JOINTS,4)))
    
    return model

def discriminator_model():
    """build the discriminator"""
    
    model = Sequential()
    model.add(Reshape(input_shape=(NB_FRAMES,NB_JOINTS,4), target_shape=(NB_FRAMES,NB_JOINTS*4)))
    model.add(Bidirectional(LSTM(512,input_shape=(NB_FRAMES, NB_JOINTS*4),return_sequences=False)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

def generator_containing_discriminator(generator, discriminator):
    """Build the GAN model by putting togehter the generator and discriminator"""
    
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model
	
	
def test_discriminator(model, data_set, num_epoch, batch_size):
    """test the discriminator alone. We evaluate its capacity of identifying real from odd data. 
    Return the trained model and informations about the training"""
    
    shuffled_data = np.random.permutation(data_set)
    
    #build training set
    X_train_reals = shuffled_data[0:2000]
    print('X_train_reals shape: ', X_train_reals.shape)
    X_train_fakes = add_noise(shuffled_data[2000:4000],0.3)
    X_train = np.concatenate((X_train_reals,X_train_fakes))
    Y_train = [1]*X_train_reals.shape[0] + [0]*X_train_fakes.shape[0]
    
    #building evaluation set
    X_eval_reals = shuffled_data[4000:5000]
    print('X_eval_reals shape: ', X_eval_reals.shape)
    X_eval_fakes = add_noise(shuffled_data[5000:6000],0.3)
    X_eval = np.concatenate((X_eval_reals,X_eval_fakes))
    Y_eval = [1]*X_eval_reals.shape[0] + [0]*X_eval_fakes.shape[0]
    
    #buildingg prediction set
    X_pred_reals = shuffled_data[6000:6123]
    print('X_pred_reals shape: ', X_pred_reals.shape)
    X_pred_fakes = add_noise(shuffled_data[6163:6326],0.3)
    X_pred = np.concatenate((X_pred_reals,X_pred_fakes))
    
    discriminator = model # building discriminator
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=False) # D optimizer
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['acc'])
    
    fitting = discriminator.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=num_epoch) #training
    score, acc = discriminator.evaluate(x=X_eval, y=Y_eval, batch_size=batch_size) #evaluation

    prediction = discriminator.predict(x=X_pred, batch_size=batch_size, verbose=1, steps=None)
    #showing the results
    print('score : ', score)
    print('accuracy : ', acc)
    plt.plot(fitting.history['acc'])
    plt.show()
    print('prediction : ', prediction)
    
    return [discriminator, fitting, prediction]
# TEST D alone
[DISCRIMINATOR, D_fitting, D_prediction] = test_discriminator(discriminator_model(), rots_data, 100, 64)


def test_generator(model, data_set, num_epoch, batch_size):
    """test the generator alone. we evaluate its capacity to generate data close to the reals ones.
    Return the trained model and informations about the training"""
    
    shuffled_data = np.random.permutation(data_set)
    
    Y_train_reals = np.array(shuffled_data[0:2000])
    print('Y_train_reals shape: ', Y_train_reals.shape)
    
    Y_eval_reals = np.array(shuffled_data[2000:4000])
    print('Y_eval_reals shape: ', Y_eval_reals.shape)
    
    #building network
    generator = model
    g_optim = SGD(lr=0.05, momentum=0.9, nesterov=False)
    generator.trainable = True
    generator.compile(loss='mean_squared_error', optimizer=g_optim, metrics=['acc'])
    noise = np.zeros((batch_size,NOISE_SHAPE[0], NOISE_SHAPE[1]))
    
    #training
    for epoch in range(num_epoch):
               
        for index in range(int(noise.shape[0]/batch_size)):
            
            for i in range(batch_size):
                noise[i, :] = np.random.rand(NOISE_SHAPE[0], NOISE_SHAPE[1])
                
            reals_batch = Y_train_reals[index*batch_size:(index+1)*batch_size]
            generator.train_on_batch(noise, reals_batch, sample_weight=None, class_weight=None)
            
    X_eval_noise = []
    for i in range(2000):
        X_eval_noise.append(np.random.rand(NOISE_SHAPE[0], NOISE_SHAPE[1]))
    X_eval_noise = np.array(X_eval_noise)

    score, acc = generator.evaluate(x=X_eval_noise, y=Y_train_reals, batch_size=batch_size) #evaluation
    
    X_pred_noise = []
    for i in range(100):
        X_pred_noise.append(np.random.rand(NOISE_SHAPE[0], NOISE_SHAPE[1]))
    X_pred_noise = np.array(X_pred_noise)
    
    prediction = generator.predict(x=X_pred_noise, batch_size=batch_size, verbose=1, steps=None)
    
    print('score : ', score)
    print('accuracy : ', acc)
    
    return [generator, prediction]
# TEST G alone
[GENERATOR, G_prediction] = test_generator(generator_model(), rots_data, 200, 64)	
#saving the generated animations when G is trained alone
BVH.save(filename = '/home/dl-box/lab/GANs/Movement_GAN/Test_Folder_GAN11/'+RUN_ID+'predictionsGAN12_generatorAlone6.bvh',
             anim = make_animation(motion_postprocess(G_prediction[0]), lowFPS_data[0][0]),
             names = NAMES,
             frametime = FRAMERATE)

			 
def test_GAN(G_model, D_model, data_set, num_epoch, batch_size):
    
    X_train = np.random.permutation(data_set)
    X_train_reals = X_train[0:2000]
    print('X_train_reals shape: ', X_train_reals.shape)
    
    discriminator = D_model # building discriminator
    generator = G_model #building generator
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator) # generator + discriminator
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=False)
    g_optim = SGD(lr=0.005, momentum=0.9, nesterov=False)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((batch_size, NOISE_SHAPE[0], NOISE_SHAPE[1]))
    
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
                noise[i, :] = np.random.rand(NOISE_SHAPE[0], NOISE_SHAPE[1])
            
            reals_batch = X_train_reals[index*batch_size:(index+1)*batch_size]
            fakes_batch = generator.predict(noise, verbose=0)    
            fakes_batch = rotations_preprocess(fakes_batch, 0.000001)
            
            X_train = np.concatenate((reals_batch, fakes_batch))
            Y_train = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X_train, Y_train)
            
            for i in range(batch_size):
                noise[i, :] = np.random.rand(NOISE_SHAPE[0], NOISE_SHAPE[1])
                
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)
            discriminator.trainable = True
            
            if index == 0:
                sample = fakes_batch[0]
                samples_list.append(sample)
                DG_losses.append(g_loss)
                D_losses.append(d_loss)
                
            #saving training samples
            if (epoch % 20 == 0)and(index==0):
                motion = motion_postprocess(fakes_batch[0])
                motion = make_animation(motion, lowFPS_data[0][0])
                motion_bvh = BVH.save(filename="/home/dl-box/lab/GANs/Movement_GAN/motion_samples_dir"+RUN_ID+"/"
                                      +"motion"+str(epoch)+".bvh",
                                      anim=motion,
                                      names=NAMES,
                                      frametime=FRAMERATE)
                  
            if epoch % 10 == 0:
                generator.save_weights('generator_weights', True)
                discriminator.save_weights('discriminator_weights', True)
    
    X_pred_noise = []
    for i in range(100):
        X_pred_noise.append(np.random.rand(NOISE_SHAPE[0], NOISE_SHAPE[1]))
    X_pred_noise = np.array(X_pred_noise)

    samples_list = np.array(samples_list)
    prediction_G_end = generator.predict(x=X_pred_noise, batch_size=batch_size, verbose=1, steps=None)
    
    for i in range(100):
        motion = motion_postprocess(prediction_G_end[i])
        motion = make_animation(motion, lowFPS_data[0][0])
        motion_bvh = BVH.save(filename=DIR+"motion_pred_dir"+RUN_ID+"/"
                                      +"pred"+str(i)+".bvh",
                                      anim=motion,
                                      names=NAMES,
                                      frametime=FRAMERATE)
    
    prediction_D_end = discriminator.predict(x=prediction_G_end, batch_size=batch_size, verbose=1, steps=None)
    prediction_D_samples = discriminator.predict(x=samples_list, batch_size=batch_size, verbose=1, steps=None)
       
    return [discriminator, generator, D_losses, DG_losses, samples_list, prediction_G_end, prediction_D_end, prediction_D_samples]
# Testing the Adversarial network
[ADV_D, ADV_DG, D_losses, DG_losses, adv_samples_list, adv_prediction_G_end, adv_prediction_D_end, adv_prediction_D_samples] = test_GAN(generator_model(), discriminator_model(), rots_data, NUM_EPOCHS, BATCH_SIZE)
#ploting statics about the training
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

print(adv_prediction_D_end)