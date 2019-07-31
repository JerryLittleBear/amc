# -*- coding: utf-8 -*-





import tensorflow as tf
import numpy as np
import os



from cifar10_resolver import CifarData
from ops import pruned_weights, trainable_variables, conv_layers, scale_arrays

model_path = '../res18_model/res18.ckpt'
class prune_env:
    def __init__(self):



        self.sess = tf.Session()
        self._define_net()
        self.layer_num = 18
        self.step_counter = 1
        self.flops_of_each_layer = np.array([19267584.,
                                            18874368., 18874368., 18874368., 18874368.,
                                            9437184.,18874368., 18874368., 18874368., 
                                            9437184.,18874368., 18874368., 18874368., 
                                            9437184.,18874368., 18874368., 18874368.,
                                            10240.])
        self.weights_of_layer = np.array([2352.,
                                        2304., 2304., 2304., 2304.,
                                        4608, 9216., 9216., 9216.,
                                        18432., 36864., 36864., 36864.,
                                        73728., 147456., 147456., 147456.,
                                        1280.])
        self.state_of_layers = np.array([[1., 64., 3., 32., 32., 1., 7., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[1:]), 1.],
                                         [2., 64., 64., 16., 16., 2., 3., self.flops_of_each_layer[1], 0., np.sum(self.flops_of_each_layer[2:]), 2.],
                                         [3., 64., 64., 16., 16., 1., 3., self.flops_of_each_layer[2], 0., np.sum(self.flops_of_each_layer[3:]), 3.],
                                         [4., 64., 64., 16., 16., 1., 3., self.flops_of_each_layer[3], 0., np.sum(self.flops_of_each_layer[4:]), 4.],
                                         [5., 64., 64., 16., 16., 2., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[5:]), 5.],
                                         [6., 128., 64., 8., 8., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[6:]), 6.],
                                         [7., 128., 128., 8., 8., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[7:]), 7.],
                                         [8., 128., 128., 8., 8., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[8:]), 8.],
                                         [9., 128., 128., 8., 8., 2., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[9:]), 9.],
                                         [10., 256., 128., 4., 4., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[10:]), 10.],
                                         [11., 256., 256., 4., 4., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[11:]), 11.],
                                         [12., 256., 256., 4., 4., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[12:]), 12.],
                                         [13., 256., 256., 4., 4., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[13:]), 13.],
                                         [14., 512., 256., 2., 2., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[14:]), 14.],
                                         [15., 512., 512., 2., 2., 2., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[15:]), 15.],
                                         [16., 512., 512., 2., 2., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[16:]), 16.],
                                         [17., 512., 512., 2., 2., 1., 3., self.flops_of_each_layer[0], 0., np.sum(self.flops_of_each_layer[17:]), 17.],
                                         [18., 0., 512., 0., 0., 0., 0., self.flops_of_each_layer[3], 1., np.sum(self.flops_of_each_layer[18:]), 18.]])





        self.state_of_layers = scale_arrays(self.state_of_layers)
        
        print(self.state_of_layers)
        self.state_of_layers_in_current_episode = self.state_of_layers
        self.prune_rate_in_current_episode = np.ones((18, 1)) * 0.01
        
        self.current_state = np.zeros([11])
        
        dataset_path = '../cifar-10-batches-py'
        #train_filename = [os.path.join(dataset_path, 'data_batch_%d' % i) for i in range(1, 6)]
        test_filename = [os.path.join(dataset_path, 'test_batch')]
        dataset = CifarData(test_filename, False)
        self.images, self.labels = dataset.next_batch(1500)
        
        
    def _define_net(self):
        self.prune_rate = tf.placeholder(dtype='float', shape=[18, 1], name='prune_rate')
        
        self.x = tf.placeholder('float', shape=[None, 3072])
        self.y = tf.placeholder(tf.int64, shape=[None])
        
        
        x_image = tf.reshape( self.x, [-1, 3, 32, 32] )
        
        x_image = tf.transpose( x_image, perm = [0, 2, 3, 1])
        
        
        
        
        
        
        
        #------------------------ Net Begin ------------------------#
        
        w0 = trainable_variables(shape=[7,7,3,16], name='w0')
        b0 = trainable_variables(shape=[16], name='b0')
        conv0_out = conv_layers(inpt = x_image, kernel = pruned_weights(w0, self.prune_rate[0]), bias = b0, strides = [1,1,1,1])
        
        pooling0 = tf.layers.max_pooling2d(conv0_out,
                                           (2,2),
                                           (2,2),
                                           padding = 'same',
                                           name = 'pooling0')
        
        w1_1 = trainable_variables(shape=[3,3,16,16], name='w1_1')
        b1_1 = trainable_variables(shape=[16], name='b1_1')
        conv1_1 = conv_layers(pooling0, pruned_weights(w1_1, self.prune_rate[1]), bias = b1_1, strides = [1,1,1,1])
        
        w1_2 = trainable_variables(shape=[3,3,16,16], name='w1_2')
        b1_2 = trainable_variables(shape=[16], name='b1_2')
        conv1_2 = conv_layers(conv1_1, pruned_weights(w1_2, self.prune_rate[2]), bias = b1_2, strides = [1,1,1,1])
        
        res1_1 = conv1_2 + pooling0
        
        w1_3 = trainable_variables(shape=[3,3,16,16], name='w1_3')
        b1_3 = trainable_variables(shape=[16], name='b1_3')
        conv1_3 = conv_layers(conv1_2, pruned_weights(w1_3, self.prune_rate[3]), bias = b1_3, strides = [1,1,1,1])
        
        w1_4 = trainable_variables(shape=[3,3,16,16], name='w1_4')
        b1_4 = trainable_variables(shape=[16], name='b1_4')
        conv1_4 = conv_layers(conv1_3, pruned_weights(w1_4, self.prune_rate[4]), bias = b1_4, strides = [1,1,1,1])
        
        res1_2 = res1_1 + conv1_4
        
        pooling1 = tf.layers.max_pooling2d(res1_2,
                                           (2,2),
                                           (2,2),
                                           padding = 'same',
                                           name = 'pooling1')
        
        w2_1 = trainable_variables(shape=[3,3,16,32], name='w2_1')
        b2_1 = trainable_variables(shape=[32], name='b2_1')
        conv2_1 = conv_layers(pooling1, pruned_weights(w2_1, self.prune_rate[5]), bias = b2_1, strides = [1,1,1,1])
        
        w2_2 = trainable_variables(shape=[3,3,32,32], name='w2_2')
        b2_2 = trainable_variables(shape=[32], name='b2_2')
        conv2_2 = conv_layers(conv2_1, pruned_weights(w2_2, self.prune_rate[6]), bias = b2_2, strides = [1,1,1,1])
        
        res2_1 = tf.pad(pooling1, [[0,0], [0,0], [0,0], [8,8]]) + conv2_2
        
        w2_3 = trainable_variables(shape=[3,3,32,32], name='w2_3')
        b2_3 = trainable_variables(shape=[32], name='b2_3')
        conv2_3 = conv_layers(conv2_2, pruned_weights(w2_3, self.prune_rate[7]), bias = b2_3, strides = [1,1,1,1])
        
        
        w2_4 = trainable_variables(shape=[3,3,32,32], name='w2_4')
        b2_4 = trainable_variables(shape=[32], name='b2_4')
        conv2_4 = conv_layers(conv2_3, pruned_weights(w2_4, self.prune_rate[8]), bias = b2_4, strides = [1,1,1,1])
        
        
        res2_2 = res2_1 + conv2_4
        
        
        pooling2 = tf.layers.max_pooling2d(res2_2,
                                           (2,2),
                                           (2,2),
                                           padding = 'same',
                                           name = 'pooling2')
        
        
        w3_1 = trainable_variables(shape=[3,3,32,64], name='w3_1')
        b3_1 = trainable_variables(shape=[64], name='b3_1')
        conv3_1 = conv_layers(pooling2, pruned_weights(w3_1, self.prune_rate[9]), bias = b3_1, strides = [1,1,1,1])
        
        
        w3_2 = trainable_variables(shape=[3,3,64,64], name='w3_2')
        b3_2 = trainable_variables(shape=[64], name='b3_2')
        conv3_2 = conv_layers(conv3_1, pruned_weights(w3_2, self.prune_rate[10]), bias = b3_2, strides = [1,1,1,1])
        
        
        res3_1 = tf.pad(pooling2,  [[0,0], [0,0], [0,0], [16,16]]) + conv3_2
        
        
        w3_3 = trainable_variables(shape=[3,3,64,64], name='w3_3')
        b3_3 = trainable_variables(shape=[64], name='b3_3')
        conv3_3 = conv_layers(conv3_2, pruned_weights(w3_3, self.prune_rate[11]), bias = b3_3, strides = [1,1,1,1])
        
        
        w3_4 = trainable_variables(shape=[3,3,64,64], name='w3_4')
        b3_4 = trainable_variables(shape=[64], name='b3_4')
        conv3_4 = conv_layers(conv3_3, pruned_weights(w3_4, self.prune_rate[12]), bias = b3_4, strides = [1,1,1,1])
        
        
        res3_2 = res3_1 + conv3_4
        
        
        pooling3 = tf.layers.max_pooling2d(res3_2,
                                           (2,2),
                                           (2,2),
                                           padding = 'same',
                                           name = 'pooling3')



        w4_1 = trainable_variables(shape=[3,3,64,128], name='w4_1')
        b4_1 = trainable_variables(shape=[128], name='b4_1')
        conv4_1 = conv_layers(pooling3, pruned_weights(w4_1, self.prune_rate[13]), bias = b4_1, strides = [1,1,1,1])
        

        w4_2 = trainable_variables(shape=[3,3,128,128], name='w4_2')
        b4_2 = trainable_variables(shape=[128], name='b4_2')
        conv4_2 = conv_layers(conv4_1, pruned_weights(w4_2, self.prune_rate[14]), bias = b4_2, strides = [1,1,1,1])
        
        
        res4_1 = tf.pad(pooling3, [[0,0], [0,0], [0,0], [32,32]]) + conv4_2
        
        
        w4_3 = trainable_variables(shape=[3,3,128,128], name='w4_3')
        b4_3 = trainable_variables(shape=[128], name='b4_3')
        conv4_3 = conv_layers(conv4_2, pruned_weights(w4_3, self.prune_rate[15]), bias = b4_3, strides = [1,1,1,1])
        
        
        w4_4 = trainable_variables(shape=[3,3,128,128], name='w4_4')
        b4_4 = trainable_variables(shape=[128], name='b4_4')
        conv4_4 = conv_layers(conv4_3, pruned_weights(w4_4, self.prune_rate[16]), bias = b4_4, strides = [1,1,1,1])
        
        
        res4_2 = res4_1 + conv4_4
        
        global_average_pooling = tf.reduce_mean(res4_2, axis = [1, 2])
        
        w_fc = trainable_variables(shape = [128, 10], name = 'w_fc')
        b_fc = trainable_variables(shape = [10], name = 'b_fc')
        y_ = tf.matmul(global_average_pooling, pruned_weights(w_fc, self.prune_rate[17])) + b_fc
        #------------------------ Net End ------------------------#
        
        
        
        predict = tf.argmax( y_, 1 )
        
        correct_prediction = tf.equal( predict, self.y )
        
        
        accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float64) )
        
        saver = tf.train.Saver({'w0':w0, 'b0':b0,
                                'w1_1':w1_1, 'b1_1':b1_1, 'w1_2':w1_2, 'b1_2':b1_2, 'w1_3':w1_3, 'b1_3':b1_3, 'w1_4':w1_4, 'b1_4':b1_4,
                                'w2_1':w2_1, 'b2_1':b2_1, 'w2_2':w2_2, 'b2_2':b2_2, 'w2_3':w2_3, 'b2_3':b2_3, 'w2_4':w2_4, 'b2_4':b2_4,
                                'w3_1':w3_1, 'b3_1':b3_1, 'w3_2':w3_2, 'b3_2':b3_2, 'w3_3':w3_3, 'b3_3':b3_3, 'w3_4':w3_4, 'b3_4':b3_4,
                                'w4_1':w4_1, 'b4_1':b4_1, 'w4_2':w4_2, 'b4_2':b4_2, 'w4_3':w4_3, 'b4_3':b4_3, 'w4_4':w4_4, 'b4_4':b4_4,
                                'w_fc':w_fc, 'b_fc':b_fc})
        saver.restore(self.sess, model_path)
        
        self.acc = accuracy
        
        
    def step(self, action):
        
        self._change_prune_rate_in_current_episode(self.step_counter, action)
        
        
        if(self.step_counter == self.layer_num):
            s_ = np.zeros([11])
            acc = self.sess.run(self.acc, feed_dict = {self.x : self.images, self.y : self.labels, self.prune_rate : self.prune_rate_in_current_episode})



            parameters_remained = (self.weights_of_layer[0] * (1 - self.prune_rate_in_current_episode[0])
                                + self.weights_of_layer[1] * (1 - self.prune_rate_in_current_episode[1])
                                + self.weights_of_layer[2] * (1 - self.prune_rate_in_current_episode[2])
                                + self.weights_of_layer[3] * (1 - self.prune_rate_in_current_episode[3])
                                + self.weights_of_layer[4] * (1 - self.prune_rate_in_current_episode[4])
                                + self.weights_of_layer[5] * (1 - self.prune_rate_in_current_episode[5])
                                + self.weights_of_layer[6] * (1 - self.prune_rate_in_current_episode[6])
                                + self.weights_of_layer[7] * (1 - self.prune_rate_in_current_episode[7])
                                + self.weights_of_layer[8] * (1 - self.prune_rate_in_current_episode[8])
                                + self.weights_of_layer[9] * (1 - self.prune_rate_in_current_episode[9])
                                + self.weights_of_layer[10] * (1 - self.prune_rate_in_current_episode[10])
                                + self.weights_of_layer[11] * (1 - self.prune_rate_in_current_episode[11])
                                + self.weights_of_layer[12] * (1 - self.prune_rate_in_current_episode[12])
                                + self.weights_of_layer[13] * (1 - self.prune_rate_in_current_episode[13])
                                + self.weights_of_layer[14] * (1 - self.prune_rate_in_current_episode[14])
                                + self.weights_of_layer[15] * (1 - self.prune_rate_in_current_episode[15])
                                + self.weights_of_layer[16] * (1 - self.prune_rate_in_current_episode[16])
                                + self.weights_of_layer[17] * (1 - self.prune_rate_in_current_episode[17]))
            
            
            
            
            total_pr = 1 - parameters_remained / np.sum(self.weights_of_layer)
            
            
            reward = 10 * np.log2(acc + 1) * np.log2(total_pr[0] + 1)
            
            done_or_not = True
            self.step_counter = 1
            
        else:
            s_ = self.state_of_layers_in_current_episode[self.step_counter]
            acc = 0.
            reward = 0.
            done_or_not = False
            self.step_counter += 1
            
        return s_, reward, done_or_not
    
    def _change_prune_rate_in_current_episode(self, step_counter, action):
        self.prune_rate_in_current_episode[step_counter - 1] = action
        
    def _reset_prune_rate_in_current_episode(self):
        self.prune_rate_in_current_episode = np.array(np.ones([18, 1]) * 0.01)
        
    def _change_states_in_current_episode(self, step_counter, action):
        if step_counter != (self.state_of_layers.shape[0] - 1):
            self.state_of_layers_in_current_episode[step_counter][10] = action
            
            
            
            
            
            
            
            
    def _reset_states_in_current_episode(self):
        self.state_of_layers_in_current_episode = self.state_of_layers
        
    def reset(self):
        self.step_counter = 1
        self._reset_states_in_current_episode()
        self._reset_prune_rate_in_current_episode()
        self.current_state = self.state_of_layers_in_current_episode[0]
        return self.current_state
    
if __name__ == '__main__':
    from DDPG_for_prune import DDPG
    ddpg = DDPG(a_dim = 1, s_dim = 11)
    
    env = prune_env()
    

    for i in range(10):
        a = i / 10. + 0.08
        re_all = 0.
        s = env.reset()
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        s_, reward, done = env.step(action = np.array(a))
        re_all += reward
        print('re_all=' + str(re_all))
    '''    
    re_all = 0
    s = env.reset()
    s_, reward, done = env.step(action = np.array(0.6))
    re_all += reward
    s_, reward, done = env.step(action = np.array(0.7))
    re_all += reward
    s_, reward, done = env.step(action = np.array(0.65))
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    s_, reward, done = env.step(action = np.array[a])
    re_all += reward
    print('re_all=' + str(re_all))
    '''
