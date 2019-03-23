# In[1]:
from __future__ import print_function
import numpy as np
import math
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import itertools
import matplotlib.pyplot as plt
import sys
import random
import os
import tarfile
import time
import scipy
import skimage
import itertools
import pandas as pd
plt.switch_backend('agg')
from datetime import datetime
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from helper import get_class_names, get_train_data, get_test_data, plot_images, plot_model
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt


##########################################################
# Check if a folder exist and create one if not
# Input -- foldername: Name of the folder
def create_folder(folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)

#########################################################

# Data_index manages the operation performed  on the index of a dataset.
# REPRESENTATION INVARIANT: 0 < batch_size <= size
class Data_index:

    def __init__(self, size, batch_size):

        self.size = size
        self.ids = np.arange(self.size)
        np.random.shuffle(self.ids)

        self.batch_size = batch_size
        self.start = 0
        self.start_limit = self.size - batch_size

    # Returns the data size
    def get_data_size (self):
        return self.size

    # Returns the batch size
    def get_batch_size(self):
        return self.batch_size

    # Reset the batch size
    # Input -- new_batch_size : The new batch size. (0 < new_batch_size <= self.size)
    def set_batch_size(self, new_batch_size):
        class Error(Exception):
            #Base class for other exceptions
            pass
        class BatchSizeTooLargeError(Error):
            #Raised when the input value is too small
            pass

        try:
            if (new_batch_size <= self.size):
                self.batch_size = new_batch_size
            else:
                raise BatchSizeTooLargeError
        except BatchSizeTooLargeError:
            print("The new batch size is bigger than the size of data. Pick a smaller batch size !!!")

    # Returns the indices for the next batch
    def get_next_batch(self):
        batch_ids = np.arange(self.start, self.start + self.batch_size).tolist()

        self.start = self.start + self.batch_size
        if (self.start > self.start_limit):
            self.start = 0
            np.random.shuffle(self.ids)

        return self.ids[batch_ids]

    def get_ids(self):
        return self.ids

    # Shuffles the indices of the dataset
    def shuffle_ids(self):
    	np.random.shuffle(self.ids)


# In[3]:

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    lab = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    df = pd.DataFrame(cm)
    df.index = lab
    df.columns = lab
    print(df)
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[16]:

def encode_labels(label):
    n = np.shape(label)[0]
    n_classes = np.shape(np.unique(label))[0]

    enc = np.zeros([n, n_classes])
    #label_ = np.zeros([50000,10])
    for i in range(n):
        enc[i,label[i]] = 1.0

    return enc


matplotlib.style.use('ggplot')
class_names = get_class_names()
print(class_names)

# Hight and width of the images
IMAGE_SIZE = 32
CHANNELS = 3
num_classes = len(class_names)
print(num_classes)


images_train, labels_train, class_train = get_train_data()
images_test, labels_test, class_test = get_test_data()
print("Training set size:\t",len(images_train))
print("Testing set size:\t",len(images_test))


patch_size_1 = 3
patch_size_2 = 3
patch_size_3 = 3

pool_stride_1 = 1
pool_stride_2 = 1
pool_stride_3 = 1

num_filters_1 = 32
num_filters_2 = 32
num_filters_3 = 64

num_hidden = 128

sample_size = 3
image_size = 32
num_labels = 10
num_channels = 3

X = tf.placeholder(tf.float32, shape =(None, image_size, image_size, num_channels))
y = tf.placeholder(tf.float32, shape =(None, num_labels))
step = tf.placeholder(tf.float32, name = 'step')

b = 0.0086

layer1_weights = tf.Variable(tf.random_uniform(shape = [3, 3, 3, 32], minval = -b, maxval = b))
layer1_biases = tf.Variable(tf.random_uniform(shape = [32], minval = -b, maxval = b))

layer2_weights = tf.Variable(tf.random_uniform(shape = [3, 3, 32, 64], minval = -b, maxval = b))
layer2_biases = tf.Variable(tf.random_uniform(shape = [64], minval = -b, maxval = b))

layer3_weights = tf.Variable(tf.random_uniform(shape = [3, 3, 64, 128], minval = -b, maxval = b))
layer3_biases = tf.Variable(tf.random_uniform(shape = [128], minval = -b, maxval = b))

layer4_weights = tf.Variable(tf.random_uniform(shape = [3, 3, 128, 128], minval = -b, maxval = b))
layer4_biases = tf.Variable(tf.random_uniform(shape = [128], minval = -b, maxval = b))

layer5_weights = tf.Variable(tf.random_uniform(shape = [4*4*128, 1024], minval = -b, maxval = b))
layer5_biases = tf.Variable(tf.random_uniform(shape = [1024], minval = -b, maxval = b))

layer6_weights = tf.Variable(tf.random_uniform([1024, num_classes], minval = -b, maxval = b))
layer6_biases = tf.Variable(tf.random_uniform([num_classes], minval = -b, maxval = b))

noise_mode_list=['nem','blind']
ann_factor_list=[1.0,2.0]
STDDEV_list=[0.03 , 0.1, 0.3, 1.0]
for a in noise_mode_list:
    for b in ann_factor_list:
        for c in STDDEV_list:
            # Change the learning rate, maximum number of epoch, and noise mode
            # For the noise mode, you can only pick from {'blind', 'none', 'nem'}
            opt_param = dict(learning_rate = 0.00002, batch_size = 128, max_epoch = 100, noise_mode = a)

            # Define the noise parameters
            # Try [1.0, 2.0]
            ann_factor = b

            # Try [0.03 , 0.1, 0.3, 1.0]
            STDDEV = c 
            noise_var = STDDEV


            # Add noise to the network
            def add_noise(ay, epoch, mode):

                nv = noise_var / math.pow(epoch, ann_factor)
                n_classes = 10
                batch_size = opt_param['batch_size']
                if (mode == "nem"):
                    noise = nv*(np.random.uniform(-0.3,0.3,[batch_size, n_classes])) 
                    crit = (noise * np.log(np.maximum(ay, 1e-3))).sum(axis = 1)
                    index = (crit >= 0).astype(float)
                    noise_index = np.reshape(np.repeat(index, n_classes), [batch_size, n_classes])
                    noise_ = noise_index * noise
                elif (mode == "blind"):
                    noise_ = nv*(np.random.uniform(-0.3,0.3,[batch_size, n_classes])) 
                else:
                    noise_ = np.zeros((batch_size, n_classes))
                return noise_



            #  CHANGE THE ACTIVATION ON THIS LINE TO ANY OF THIS {relu, tanh, sigmoid}
            def model(data):
                conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer1_biases)

                conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer2_biases)

                pool = tf.nn.avg_pool(hidden, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
                pool = tf.nn.dropout(pool, 0.7)

                conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer3_biases)

                pool = tf.nn.avg_pool(hidden, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

                conv = tf.nn.conv2d(pool, layer4_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer4_biases)

                pool = tf.nn.avg_pool(hidden, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

                pool = tf.nn.dropout(pool, 0.7)
                shape = pool.get_shape().as_list()

                hidden = tf.reshape(pool, [-1, shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(hidden, layer5_weights) + layer5_biases)

                hidden = tf.nn.dropout(hidden, 0.7)
                return tf.matmul(hidden, layer6_weights) + layer6_biases



            # IF YOU HAVE GPU ON YOUR SYSTEM, UNCOMMENT THE NEXT LINE
            with tf.device('/device:GPU:0'):
                logit = model(X)
                prob = tf.nn.softmax(logit, name = 'prob')
                loss1 = -1 * tf.reduce_mean(tf.reduce_sum(y * tf.log(tf.maximum(prob, 1e-3)), axis = 1))
                correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100.0
                tf.summary.scalar('accuracy', accuracy)
                pred_label = tf.argmax(logit, 1)
                with tf.name_scope('loss'):
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))
                tf.summary.scalar('loss', loss)
                optimizer = tf.train.AdamOptimizer(opt_param['learning_rate']).minimize(loss1)

            idx = Data_index(np.shape(images_train)[0], opt_param['batch_size'])
            images_test = images_test - 0.5
            images_train = images_train - 0.5

            labels_train_ = encode_labels(labels_train)
            labels_test_ = encode_labels(labels_test)
            
            # Create a folder
            time_stamp = datetime.now().strftime("%m_%d_%H_%M")
            folder_name = "Result/" + time_stamp + "_CNN_NOISE:_" + opt_param['noise_mode']+ann_factor+STDDEV
            create_folder(folder_name)
            
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(folder_name+'/train', sess.graph)
            test_writer = tf.summary.FileWriter(folder_name+'/test')

            sess.run(tf.global_variables_initializer())
            loss_ = np.empty([0])
            acc_  = np.empty([0])

            # Training cycle
            for epoch in range(opt_param['max_epoch'] + 1):

                if epoch > 0:
                    for i in range(int(50000 / opt_param['batch_size'])):
                        ids = idx.get_next_batch()
                        batch_x = images_train[ids,:,:,:]
                        batch_y = labels_train_[ids, :]
                        ay = sess.run(prob, feed_dict={X: batch_x, y: batch_y})
                        noise_1 = add_noise(ay, epoch, opt_param['noise_mode'])
                        batch_y = batch_y + noise_1
                        summary,_=sess.run([merged,optimizer], feed_dict={X: batch_x, y: batch_y})
                    train_writer.add_summary(summary, epoch)
                print("Epoch:", '%03d' % (epoch))
                #a, l = sess.run([accuracy,loss], {X: images_test, y: labels_test_})
                summary,a1, l1 = sess.run([merged,accuracy,loss], {X: images_test[0:5000,:,:,:], y: labels_test_[0:5000,:]})
                summary,a2, l2 = sess.run([merged,accuracy,loss], {X: images_test[5000:,:,:,:], y: labels_test_[5000:,:]})
                a = 0.5 * (a1+ a2)
                l = 0.5 * (l1+ l2)
                print("Accuracy:", a)
                print("Loss_Fxn:", l)
                test_writer.add_summary(summary, epoch)
                acc_ = np.append(acc_, a)
                loss_ = np.append(loss_, l)
            print("Optimization Finished!")
            train_writer.close()
            test_writer.close()

            

            np.savetxt(folder_name + "/accuracy.csv", acc_, delimiter = ',', fmt = '%.4f')
            np.savetxt(folder_name + "/Loss_fxn.csv", loss_, delimiter = ',', fmt = '%.4f')


            #pred_test_label = sess.run(pred_label,feed_dict= {X: images_test})
            pred_test_label0 = sess.run(pred_label,feed_dict= {X: images_test[0:5000,:,:,:]})
            pred_test_label1 = sess.run(pred_label,feed_dict= {X: images_test[5000:,:,:,:]})
            pred_test_label = np.append(pred_test_label0, pred_test_label1, axis = 0)
            cnf_matrix = confusion_matrix(labels_test, pred_test_label)
            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
            plt.savefig(folder_name + "/CNN_confusion_matrix")
            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')

            plt.savefig(folder_name + "/CNN_normalized_confusion_matrix")
            plt.show()

