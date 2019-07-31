# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from env_res18_all_layers import prune_env
from DDPG_for_prune import DDPG

#EXPLORE_MODE = 'action' # or 'parameter'
EXPLORE_MODE = 'parameter'
def explore_a(a, var, a_down_limit, a_up_limit):
    a_ = np.clip(np.random.normal(a, var), a_down_limit, a_up_limit)
    while a_ == a_down_limit or a_ == a_up_limit :
        a_ = np.clip(np.random.normal(a, var), a_down_limit, a_up_limit)
    return a_


class s_n_a_buffer:
    def __init__(self, buffer_size):
        self.s_buf = []
        self.a_buf = []
        self.s_next_buf = []
        self.buffer_size = buffer_size
        
    def clean(self):
        self.s_buf = []
        self.a_buf = []
        self.s_next_buf = []
    
    def storeInBuffer(self, s, a, s_, stored_times):
        if stored_times > self.buffer_size:
            exit('Error:s_n_a_buffer should be cleaned')
        self.s_buf.append(s)
        self.a_buf.append(a)
        self.s_next_buf.append(s_)
    
env = prune_env()
rl_agent = DDPG(a_dim = 1, s_dim = 11)

var = np.array(0.99, dtype=np.float32)
mem_counter = 0
var_decy_rat = (1e-5 / 0.99) ** (1.0 / 400.)
for ep in range(4500):
    
    s = env.reset()
    ep_reward = 0
    rewards = np.zeros(env.layer_num)

    if ep % 5 == 0 and rl_agent.memory_full() == True: # test
        print('variance = ' + str('%.4f'%var))
        s = env.reset()
        print('episode:' + str(ep))
        for t in range(env.layer_num):
            a_test = rl_agent.choose_action(s)
            print('a' + str(t) + '=' + str('%.2f'%a_test))
            s__test, r_test, done_test = env.step(a_test)
            s = s__test
            ep_reward += r_test
            if done_test :
                break
        print('reward:' + str('%.4f'%ep_reward) + '\n')
    else: # train
        if rl_agent.memory_full() == True:
            var *= var_decy_rat
        s = env.reset()
        buf = s_n_a_buffer(env.layer_num)
        
        for layer_num in range(env.layer_num):
            if EXPLORE_MODE == 'parameter':
                a = rl_agent.choose_noisy_action(s, var)
            elif EXPLORE_MODE == 'action':
                a = rl_agent.choose_action(s)
                a = explore_a(a, var, 0.01, 0.99)
            s_, reward, done = env.step(a)
            rewards[layer_num] = reward
            buf.storeInBuffer(s, a, s_, layer_num + 1)
            if layer_num == env.layer_num - 1 :
                for j in range(env.layer_num) :
                    rl_agent.store_transition(buf.s_buf[j], buf.a_buf[j], reward, buf.s_next_buf[j])
                buf.clean()
            if rl_agent.memory_full() == True:  
                rl_agent.learn()
            s = s_
            ep_reward += reward
            mem_counter += 1
            if done :
                break
        rl_agent.finalize_rlout(rewards)
        print('ep' + str(ep) + ':' + str('%.4f' % ep_reward))

rl_agent.sess.close()
tf.reset_default_graph()
    
