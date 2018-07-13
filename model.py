import tensorflow as tf
import utils
import random
import cv2
import numpy as np
from backalla_utils import tensorflow as tfu
from backalla_utils.misc import rprint

class Model:
    def __init__(self, batch_size = 32, data_path = "./data/omniglot/", learning_rate=1e-3):
        self.dataset = utils.get_dataset(data_path=data_path)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        print(len(self.dataset))
        self.train_data, self.test_data = self.prepare_dataset()
        self.image_height,self.image_width,self.image_channels = 105,105,3
        self.model = tfu.Model(name = "test-model")
        self.model_graph = self.model.init().as_default()
        self.Xa = tf.placeholder(tf.float32,shape=[None,self.image_height,self.image_width,self.image_channels],name="anchor_image_input")
        self.Xp = tf.placeholder(tf.float32,shape=[None,self.image_height,self.image_width,self.image_channels],name="positive_image_input")
        self.Xn = tf.placeholder(tf.float32,shape=[None,self.image_height,self.image_width,self.image_channels],name="negative_image_input")
        self.a = tf.placeholder_with_default(0.2,shape=(),name="alpha")
        self.Xa_embed, self.Xp_embed, self.Xn_embed = self.build_model(self.Xa,False),self.build_model(self.Xp,True),self.build_model(self.Xn,True)
        self.train_ops = self.build_train_ops()
        # self.train()

    
    def prepare_dataset(self):

        labels_list = list(self.dataset.keys())
        train_dataset_list = []
        test_dataset_list = []
        num_batches = 200
        for _ in range(self.batch_size*num_batches):
            label_positive,label_negative = random.sample(labels_list,2)
            assert label_positive != label_negative, "Positive and negative labels cannot be same."
            [anchor_path,positive_path] = random.sample(self.dataset[label_positive]["train"],2)
            [negative_path] = random.sample(self.dataset[label_negative]["train"],1)
            train_dataset_list.append([cv2.imread(anchor_path)/255.0,cv2.imread(positive_path)/255.0,cv2.imread(negative_path)/255.0])
        
        for _ in range(self.batch_size*num_batches):
            label_positive,label_negative = random.sample(labels_list,2)
            assert label_positive != label_negative, "Positive and negative labels cannot be same."
            [anchor_path,positive_path] = random.sample(self.dataset[label_positive]["valid"],2)
            [negative_path] = random.sample(self.dataset[label_negative]["valid"],1)
            test_dataset_list.append([cv2.imread(anchor_path)/255.0,cv2.imread(positive_path)/255.0,cv2.imread(negative_path)/255.0])
        

        return train_dataset_list,test_dataset_list
    
    def build_model(self,input,reuse=False):
        with tf.name_scope("model"):
            with tf.variable_scope("conv1") as scope:
                net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv2") as scope:
                net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv3") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv4") as scope:
                net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            net = tf.contrib.layers.flatten(net)
        
        return net
    
    def triplet_loss(self, anchor, positive, negative, alpha):
        """Calculate the triplet loss according to the FaceNet paper
        
        Args:
        anchor: the embeddings for the anchor images.
        positive: the embeddings for the positive images.
        negative: the embeddings for the negative images.
    
        Returns:
        the triplet loss according to the FaceNet paper as a float tensor.
        """
        with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
            
            basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
            loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        
        return loss
    
    def build_train_ops(self):
        loss = self.triplet_loss(self.Xa_embed, self.Xp_embed, self.Xn_embed, self.a)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return [optimizer,loss]
    
    def train(self,num_epochs = 20,restore_model=False):
        with self.model.session() as sess:
            sess.run(tf.global_variables_initializer())
            if restore_model:
                self.model.restore_weights()
            num_train_samples = len(self.train_data)
            for epoch in range(num_epochs):
                print("Epoch:",epoch)
                for batch_index in range(0,num_train_samples,self.batch_size):
                    train_batch_xa = np.array(self.train_data[batch_index:batch_index+self.batch_size])[:,0,:]
                    train_batch_xp = np.array(self.train_data[batch_index:batch_index+self.batch_size])[:,1,:]
                    train_batch_xn = np.array(self.train_data[batch_index:batch_index+self.batch_size])[:,2,:]
                    _, loss_val = sess.run(self.train_ops,feed_dict = {self.Xa:train_batch_xa,self.Xp: train_batch_xp, self.Xn: train_batch_xn, self.a: 2.0})
                    rprint("Batch: {}  Loss: {}".format(batch_index,loss_val))
                if epoch%5==0:
                    self.model.save_weights(weight_file_prefix="test")



        

            



