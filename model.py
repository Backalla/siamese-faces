import tensorflow as tf
import utils
import random
import cv2
import numpy as np
from backalla_utils import tensorflow as tfu
from backalla_utils.misc import rprint
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier



class Model:
    def __init__(self, model_name, batch_size = 32, data_path = "./data/omniglot/", learning_rate=1e-3):
        self.dataset = utils.get_dataset(data_path=data_path)
        print(f"Num of classes: {len(self.dataset)}")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.train_data, self.valid_data = self.prepare_dataset()
        self.image_height,self.image_width,self.image_channels = 64,64,1
        self.model = tfu.Model(name = self.model_name)
        self.model_summaries = []
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
        self.valid_data_init, self.valid_data_next = self.get_dataset_iterator(self.valid_data,self.batch_size)
        self.train_ops = self.build_train_ops()
        self.model.visualise()
        # self.train()

    
    def prepare_dataset(self):

        labels_list = list(self.dataset.keys())
        train_dataset_list = []
        valid_dataset_list = []
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
            valid_dataset_list.append([anchor_path,positive_path,negative_path])
        

        

        return train_dataset_list, valid_dataset_list

    def parse_function(self, filename):
        anchor_image_string = tf.read_file(filename[0])
        positive_image_string = tf.read_file(filename[1])
        negative_image_string = tf.read_file(filename[2])


        anchor_image = tf.image.decode_png(anchor_image_string, channels=self.image_channels)
        positive_image = tf.image.decode_png(positive_image_string, channels=self.image_channels)
        negative_image = tf.image.decode_png(negative_image_string, channels=self.image_channels)

        # This will convert to float values in [0, 1]
        anchor_image = tf.image.convert_image_dtype(anchor_image, tf.float32)
        positive_image = tf.image.convert_image_dtype(positive_image, tf.float32)
        negative_image = tf.image.convert_image_dtype(negative_image, tf.float32)

        anchor_image = tf.image.resize_images(anchor_image, [self.image_height, self.image_width])
        positive_image = tf.image.resize_images(positive_image, [self.image_height, self.image_width])
        negative_image = tf.image.resize_images(negative_image, [self.image_height, self.image_width])

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

    def _conv_layer(self, input, num_filters, kernel_size, activation, padding, scope, reuse, batch_norm=False):
        with tf.variable_scope(scope) as scope_obj:
            cnn_output = tf.contrib.layers.conv2d(input, num_filters, kernel_size, activation_fn=activation, padding=padding,
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope_obj,reuse=reuse)
            if not reuse:        
                print(f"{scope}:",cnn_output.get_shape())
                self.model_summaries.append(tf.contrib.layers.summarize_tensor(cnn_output,scope))
            
            if batch_norm:
                cnn_output = tf.layers.batch_normalization(cnn_output, axis=3, training=True,reuse=reuse)
        return cnn_output
    
    def _maxpool_layer(self,input, kernel_size, padding, scope, reuse):
        with tf.variable_scope(scope) as scope_obj:
            maxpool_output = tf.contrib.layers.max_pool2d(input, kernel_size=kernel_size, padding=padding)
            if not reuse:
                print(f"{scope}:",maxpool_output.get_shape())
        return maxpool_output



    def build_model(self,input,reuse=False):
        if not reuse:
            print("input:",input.get_shape())
        with tf.name_scope("model"):
            
                
            net = self._conv_layer(input=input,num_filters=16, kernel_size=[7,7], activation=tf.nn.relu, padding="SAME", scope="conv1",reuse=reuse)
            net = self._maxpool_layer(input=net,kernel_size=2,padding="SAME", scope="maxpool1", reuse=reuse)
            
            net = self._conv_layer(input=net,num_filters=32, kernel_size=[5,5], activation=tf.nn.relu, padding="SAME", scope="conv2",reuse=reuse,batch_norm=True)
            net = self._maxpool_layer(input=net,kernel_size=2,padding="SAME", scope="maxpool2", reuse=reuse)

            net = self._conv_layer(input=net,num_filters=64, kernel_size=[3,3], activation=tf.nn.relu, padding="SAME", scope="conv3",reuse=reuse)
            net = self._maxpool_layer(input=net,kernel_size=2,padding="SAME", scope="maxpool3", reuse=reuse)

            # net = self._conv_layer(input=net,num_filters=128, kernel_size=[1,1], activation=tf.nn.relu, padding="SAME", scope="conv4",reuse=reuse,batch_norm=True)
            # net = self._maxpool_layer(input=net,kernel_size=2,padding="SAME", scope="maxpool4", reuse=reuse)

            net = self._conv_layer(input=net,num_filters=8, kernel_size=[1,1], activation=None, padding="SAME", scope="conv5",reuse=reuse)
            net = self._maxpool_layer(input=net,kernel_size=2,padding="SAME", scope="maxpool5", reuse=reuse)

            net = tf.contrib.layers.flatten(net)
            if not reuse:
                print("flatten:",net.get_shape())
            
            # net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            # if not reuse:
            #     print("fully-connected:",net.get_shape())

        
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
        valid_loss_summary = tf.summary.scalar("valid_loss",loss)
        
        pos_dist_mean = tf.reduce_mean(pos_dist)
        neg_dist_mean = tf.reduce_mean(neg_dist)
        
        pos_dist_summary = tf.summary.scalar("pos_dist",pos_dist_mean)
        valid_pos_dist_summary = tf.summary.scalar("valid_pos_dist",pos_dist_mean)
        
        neg_dist_summary = tf.summary.scalar("neg_dist",neg_dist_mean)
        valid_neg_dist_summary = tf.summary.scalar("valid_neg_dist",neg_dist_mean)
        
        difference_summary = tf.summary.scalar("neg-pos", neg_dist_mean-pos_dist_mean)
        valid_difference_summary = tf.summary.scalar("valid_neg-pos", neg_dist_mean-pos_dist_mean)

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return {
            "train":[optimizer,loss, loss_summary,pos_dist_summary,neg_dist_summary,difference_summary],
            "valid": [loss,valid_loss_summary,valid_pos_dist_summary,valid_neg_dist_summary,valid_difference_summary]
        }


    
    def build_known_entities(self, labels_json, db_name = "embeddings_db"):
        embeddings = []
        labels = []
        images_list = []
        classes = set(labels_json.values())
        label2int = {label:index for index,label in enumerate(classes)}
        int2label = {label2int[label]:label for label in label2int}

        for path in labels_json:
            image_obj = cv2.imread(path,0)/255.0
            image_obj = cv2.resize(image_obj,(self.image_height,self.image_width))
            image_obj = np.expand_dims(image_obj,axis=3)

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
        valid_labels = {}
        for label in self.dataset:
            validation_paths = self.dataset[label]['valid']
            if len(validation_paths)<4:
                continue
            validation_paths = random.sample(validation_paths,4)
            train_labels[validation_paths[0]] = label
            train_labels[validation_paths[1]] = label
            train_labels[validation_paths[2]] = label
            valid_labels[validation_paths[3]] = label
        
        self.build_known_entities(train_labels,"train_embeddings")
        self.build_known_entities(valid_labels,"valid_embeddings")         
    
    def calculate_accuracy(self):
        classifier = KNeighborsClassifier(n_neighbors=2)
        (y,x) = pickle.load(open("train_embeddings.p","rb"))
        classifier.fit(x,y)
        (y_test,x_test) = pickle.load(open("valid_embeddings.p","rb"))
        predictions = classifier.predict(x_test)
        correct = predictions==y_test
        accuracy = np.sum(correct)/len(predictions)
        print(f"Accuracy: {(accuracy*100):.3f}%")
    
    def _write_summaries(self, summaries, step):
        for summary in summaries:
            self.model.summary_writer.add_summary(summary,global_step=step)



    def train(self,num_epochs = 1000,restore_model=False):
        with self.model.session() as sess:
            sess.run(tf.global_variables_initializer())
            if restore_model:
                self.model.restore_weights()
            sess.run(self.train_data_init)
            sess.run(self.valid_data_init)
            for epoch in range(num_epochs):
                print("\nEpoch:",epoch)
                train_batch_index = 0
                valid_batch_index = 0
                epoch_start_time = time.time() 
                while True:
                    try:
                        start_time = time.time()
                        train_batch_xa, train_batch_xp, train_batch_xn = sess.run(self.train_data_next)
                        valid_batch_xa, valid_batch_xp, valid_batch_xn = sess.run(self.valid_data_next)
                        combined_image = np.concatenate((train_batch_xa, train_batch_xp, train_batch_xn),axis=1)
                        if train_batch_index % 20==0:
                            self.model.summary_writer.add_summary(sess.run(self.combined_image_summary,feed_dict={ self.X_combined:combined_image}),global_step=sess.run(self.global_step))

                        train_outputs = sess.run(self.train_ops["train"]+self.model_summaries,feed_dict = {self.Xa:train_batch_xa,self.Xp: train_batch_xp, self.Xn: train_batch_xn, self.a: 5.0})
                        loss_val = train_outputs[1]
                        train_summaries_vals = train_outputs[2:]

                        valid_outputs = sess.run(self.train_ops["valid"],feed_dict = {self.Xa:valid_batch_xa,self.Xp: valid_batch_xp, self.Xn: valid_batch_xn, self.a: 5.0})
                        valid_summaries_vals = valid_outputs[1:]

                        # summaries_list = [loss_summary_val,pos_dist_summary_val,neg_dist_summary_val, difference_summary_val,valid_loss_summary_val,valid_pos_dist_summary_val,valid_neg_dist_summary_val, valid_difference_summary_val]
                        self._write_summaries(train_summaries_vals+valid_summaries_vals,sess.run(self.global_step))
                        
                        sess.run(self.increment_global_step_op)
                        rprint(f"Batch: {train_batch_index}-{valid_batch_index}, Loss: {loss_val:.4f}, Time: {(time.time()-epoch_start_time):.4f}, Time/batch: {(time.time()-start_time):.4f}")
                        train_batch_index+=1
                        valid_batch_index+=1
                    
                    except tf.errors.OutOfRangeError:
                        sess.run(self.train_data_init)
                        sess.run(self.valid_data_init)
                        break
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
                            self.model.save_weights(weight_file_prefix=self.model_name)

                        else:
                            print("Resuming training..")

                            continue
                    except Exception as e:
                        print("Exception aaya..\n",str(e))
                        raise(e)
                        # continue




    

        



