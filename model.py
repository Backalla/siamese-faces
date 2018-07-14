import tensorflow as tf
import utils
import random
import cv2
import numpy as np
from backalla_utils import tensorflow as tfu
from backalla_utils.misc import rprint
import time
import pickle

class Model:
    def __init__(self, model_name, batch_size = 32, data_path = "./data/omniglot/", learning_rate=1e-3):
        self.dataset = utils.get_dataset(data_path=data_path)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        print(len(self.dataset))
        self.train_data, self.test_data = self.prepare_dataset()
        self.image_height,self.image_width,self.image_channels = 105,105,3
        self.model = tfu.Model(name = model_name)
        self.model_graph = self.model.init().as_default()
        self.Xa = tf.placeholder(tf.float32,shape=[None,self.image_height,self.image_width,self.image_channels],name="anchor_image_input")
        self.Xp = tf.placeholder(tf.float32,shape=[None,self.image_height,self.image_width,self.image_channels],name="positive_image_input")
        self.Xn = tf.placeholder(tf.float32,shape=[None,self.image_height,self.image_width,self.image_channels],name="negative_image_input")
        self.X_combined = tf.placeholder(tf.float32,shape=[None,None,None,None],name="combined_image_input")
        self.a = tf.placeholder_with_default(0.2,shape=(),name="alpha")
        self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)
        self.combined_image_summary = tf.summary.image("imput_images",self.X_combined)
        self.Xa_embed, self.Xp_embed, self.Xn_embed = self.build_model(self.Xa,False),self.build_model(self.Xp,True),self.build_model(self.Xn,True)
        self.train_data_init, self.train_data_next = self.get_dataset_iterator(self.train_data,self.batch_size)
        self.test_data_init, self.test_data_next = self.get_dataset_iterator(self.test_data,self.batch_size)
        self.train_ops = self.build_train_ops()
        self.model.visualise()
        # self.train()

    
    def prepare_dataset(self):

        labels_list = list(self.dataset.keys())
        train_dataset_list = []
        test_dataset_list = []
        num_batches = 400
        for _ in range(self.batch_size*num_batches):
            label_positive,label_negative = random.sample(labels_list,2)
            assert label_positive != label_negative, "Positive and negative labels cannot be same."
            [anchor_path,positive_path] = random.sample(self.dataset[label_positive]["train"],2)
            [negative_path] = random.sample(self.dataset[label_negative]["train"],1)
            train_dataset_list.append([anchor_path,positive_path,negative_path])
        
        for _ in range(self.batch_size*num_batches):
            label_positive,label_negative = random.sample(labels_list,2)
            assert label_positive != label_negative, "Positive and negative labels cannot be same."
            [anchor_path,positive_path] = random.sample(self.dataset[label_positive]["valid"],2)
            [negative_path] = random.sample(self.dataset[label_negative]["valid"],1)
            test_dataset_list.append([anchor_path,positive_path,negative_path])
        

        

        return train_dataset_list, test_dataset_list

    def parse_function(self, filename):
        anchor_image_string = tf.read_file(filename[0])
        positive_image_string = tf.read_file(filename[1])
        negative_image_string = tf.read_file(filename[2])


        anchor_image = tf.image.decode_image(anchor_image_string, channels=self.image_channels)
        positive_image = tf.image.decode_image(positive_image_string, channels=self.image_channels)
        negative_image = tf.image.decode_image(negative_image_string, channels=self.image_channels)

        # This will convert to float values in [0, 1]
        anchor_image = tf.image.convert_image_dtype(anchor_image, tf.float32)
        positive_image = tf.image.convert_image_dtype(positive_image, tf.float32)
        negative_image = tf.image.convert_image_dtype(negative_image, tf.float32)

        anchor_image = tf.image.resize_image_with_crop_or_pad(anchor_image, self.image_height, self.image_width)
        positive_image = tf.image.resize_image_with_crop_or_pad(positive_image, self.image_height, self.image_width)
        negative_image = tf.image.resize_image_with_crop_or_pad(negative_image, self.image_height, self.image_width)

        return anchor_image, positive_image, negative_image

    def get_dataset_iterator(self, filenames, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(self.parse_function, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        init_op = iterator.initializer

        return init_op, next_element   
    def build_model(self,input,reuse=False):
        if not reuse:
            print("input:",input.get_shape())
        with tf.name_scope("model"):
            with tf.variable_scope("conv1") as scope:
                net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                if not reuse:        
                    print("conv1:",net.get_shape())
                
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
                if not reuse:
                    print("maxpool1:",net.get_shape())


            with tf.variable_scope("conv2") as scope:
                net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                if not reuse:                
                    print("conv2:",net.get_shape())
                
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
                if not reuse:
                    print("maxpool2:",net.get_shape())

            with tf.variable_scope("conv3") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                if not reuse:
                    print("conv3:",net.get_shape())
                
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
                if not reuse:
                    print("maxpool3:",net.get_shape())


            with tf.variable_scope("conv4") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                if not reuse:
                    print("conv4:",net.get_shape())
                
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
                if not reuse:
                    print("maxpool4:",net.get_shape())


            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                if not reuse:
                    print("conv5:",net.get_shape())
                
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
                if not reuse:
                    print("maxpool5:",net.get_shape())


            net = tf.contrib.layers.flatten(net)
            if not reuse:
                print("flatten:",net.get_shape())

        
        return net


    def triplet_loss(self, anchor, positive, negative, alpha):
        with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
            basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
            loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        
        return loss,pos_dist,neg_dist
    
    def build_train_ops(self):
        loss,pos_dist,neg_dist = self.triplet_loss(self.Xa_embed, self.Xp_embed, self.Xn_embed, self.a)
        loss_summary = tf.summary.scalar("loss",loss)
        test_loss_summary = tf.summary.scalar("test_loss",loss)
        
        pos_dist_mean = tf.reduce_mean(pos_dist)
        neg_dist_mean = tf.reduce_mean(neg_dist)
        
        pos_dist_summary = tf.summary.scalar("pos_dist",pos_dist_mean)
        test_pos_dist_summary = tf.summary.scalar("test_pos_dist",pos_dist_mean)
        
        neg_dist_summary = tf.summary.scalar("neg_dist",neg_dist_mean)
        test_neg_dist_summary = tf.summary.scalar("test_neg_dist",neg_dist_mean)
        
        difference_summary = tf.summary.scalar("neg-pos", neg_dist_mean-pos_dist_mean)
        test_difference_summary = tf.summary.scalar("test_neg-pos", neg_dist_mean-pos_dist_mean)

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return {
            "train":[optimizer,loss, loss_summary,pos_dist_summary,neg_dist_summary,difference_summary],
            "test": [loss,test_loss_summary,test_pos_dist_summary,test_neg_dist_summary,test_difference_summary]
        }


    
    def build_known_entities(self, labels_json, db_name = "embeddings_db"):
        embeddings = []
        labels = []
        images_list = []
        classes = set(labels_json.values())
        label2int = {label:index for index,label in enumerate(classes)}
        int2label = {label2int[label]:label for label in label2int}

        for path in labels_json:
            image_obj = cv2.imread(path)/255.0
            images_list.append(image_obj)
            labels.append(label2int[labels_json[path]])
        
        with self.model.session() as sess:
            self.model.restore_weights()
            for batch_index in range(0,len(images_list),self.batch_size):
                batch_images = images_list[batch_index:batch_index+self.batch_size]
                batch_embeddings = sess.run(self.Xa_embed,feed_dict={self.Xa:batch_images})
                embeddings.extend(batch_embeddings)
        
        pickle.dump((labels,embeddings),open(f"{db_name}.p","wb"))
    
    def create_validation_db(self):
        train_labels = {}
        test_labels = {}
        for label in self.dataset:
            validation_paths = self.dataset[label]['valid']
            if len(validation_paths)<4:
                continue
            validation_paths = random.sample(validation_paths,4)
            train_labels[validation_paths[0]] = label
            train_labels[validation_paths[1]] = label
            train_labels[validation_paths[2]] = label
            test_labels[validation_paths[3]] = label
        
        self.build_known_entities(train_labels,"train_embeddings")
        self.build_known_entities(test_labels,"test_embeddings")
            
        


    def train(self,num_epochs = 1000,restore_model=False):
        with self.model.session() as sess:
            sess.run(tf.global_variables_initializer())
            if restore_model:
                self.model.restore_weights()
            sess.run(self.train_data_init)
            sess.run(self.test_data_init)
            for epoch in range(num_epochs):
                print("\nEpoch:",epoch)
                train_batch_index = 0
                test_batch_index = 0 
                while True:
                    try:
                        start_time = time.time()
                        train_batch_xa, train_batch_xp, train_batch_xn = sess.run(self.train_data_next)
                        test_batch_xa, test_batch_xp, test_batch_xn = sess.run(self.test_data_next)
                        combined_image = np.concatenate((train_batch_xa, train_batch_xp, train_batch_xn),axis=1)
                        if train_batch_index % 20==0:
                            self.model.summary_writer.add_summary(sess.run(self.combined_image_summary,feed_dict={ self.X_combined:combined_image}),global_step=sess.run(self.global_step))

                        _, loss_val, loss_summary_val,pos_dist_summary_val,neg_dist_summary_val, difference_summary_val = sess.run(self.train_ops["train"],feed_dict = {self.Xa:train_batch_xa,self.Xp: train_batch_xp, self.Xn: train_batch_xn, self.a: 5.0})
                        self.model.summary_writer.add_summary(loss_summary_val,global_step=sess.run(self.global_step))
                        self.model.summary_writer.add_summary(pos_dist_summary_val,global_step=sess.run(self.global_step))
                        self.model.summary_writer.add_summary(neg_dist_summary_val,global_step=sess.run(self.global_step))
                        self.model.summary_writer.add_summary(difference_summary_val,global_step=sess.run(self.global_step))


                        loss_val, loss_summary_val,pos_dist_summary_val,neg_dist_summary_val, difference_summary_val = sess.run(self.train_ops["test"],feed_dict = {self.Xa:test_batch_xa,self.Xp: test_batch_xp, self.Xn: test_batch_xn, self.a: 5.0})
                        self.model.summary_writer.add_summary(loss_summary_val,global_step=sess.run(self.global_step))
                        self.model.summary_writer.add_summary(pos_dist_summary_val,global_step=sess.run(self.global_step))
                        self.model.summary_writer.add_summary(neg_dist_summary_val,global_step=sess.run(self.global_step))
                        self.model.summary_writer.add_summary(difference_summary_val,global_step=sess.run(self.global_step))

                        sess.run(self.increment_global_step_op)
                        rprint(f"Batch: {train_batch_index}-{test_batch_index}, Loss: {loss_val:.4f}, Time: {(time.time()-start_time):.4f}")
                        train_batch_index+=1
                        test_batch_index+=1
                    
                    except tf.errors.OutOfRangeError:
                        print("\nBatch ended")
                        train_batch_index=0
                        test_batch_index=0
                        sess.run(self.train_data_init)
                        sess.run(self.test_data_init)
                    except KeyboardInterrupt:
                        print("Training paused..")

                        print("Press [q] to quit without saving")

                        print("Press [s] to save and continue")

                        print("Anything else to resume training..")

                        option = input("> ")
                        if option.lower() == "q":
                            print("Stopping training without saving..")

                            print("Exiting by keyboard interrupt..")
                            return
                        elif option.lower() == 's':
                            print("Saving model..")
                            self.model.save_weights(weight_file_prefix="test")

                        else:
                            print("Resuming training..")

                            continue
                    except Exception as e:
                        print("Exception aaya..\n",str(e))
                        raise(e)
                        # continue




    

        



