import tensorflow as tf
import numpy as np
from ops import scale_arrays

LR_A = 5e-5
LR_C = 0.001
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 1800
BATCH_SIZE = 64
RENDER  = False
ENV_NAME = 'Pendulum-v0'
enbl_bsln_func = False

def get_target_init_ops(model, model_tr):
  """Get operations related to the target model.

  Args:
  * model: original model
  * model_tr: target model

  Returns:
  * init_op: initialization operation for the target model
  * updt_op: update operation for the target model
  """
  init_ops = []
  for var, var_tr in zip(model, model_tr):
    init_ops.append(tf.assign(var_tr, var))

  return tf.group(*init_ops)

class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.reward_ema = None
        self.actions = tf.placeholder(tf.float32, [None, a_dim], 'actions_for_critic_update')
        self.ops = {}
        self.noisy_variance = tf.placeholder(tf.float32, [], 'noisy_variance')
        self.memory_capacity = MEMORY_CAPACITY
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype = np.float32)
        self.stored_times = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope = 'eval', trainable = True)
            a_ = self._build_a(self.S_, scope = 'target', trainable = True)
            self.a_noisy = self._build_a(self.S, scope = 'eval_noise', trainable = True)

        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.actions, scope = 'eval', trainable = True)
            q_ = self._build_c(self.S_, a_, scope = 'target', trainable = True)
            q_for_a_loss = self._build_c(self.S, self.a, scope = 'eval', trainable = True, reuse = True)#可以尝试把trainable=False，因为这里是为了更新Action网络的参数，Critic网络参数可以设为不可训练。不过，由于这里的scope=‘eval’，reuse=True，所以这个critic网络是重用了Online评价网络的参数，这些参数创建时是可训练的，可能无法使其不可训练。
        
        q_target = self.R + GAMMA * q_
        
        self.ae_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Actor/target')
        self.an_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/eval_noise')
        self.ce_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Critic/target')
        #var_noise = tf.random_normal(tf.shape(var_clean), mean=0., stddev=param_noise_std)
        self.ops['param_explore'] = self.get_perturb_op(self.ae_params, self.an_params, self.noisy_variance)

        self.ops['a_target_init'] = get_target_init_ops(self.ae_params, self.at_params)
        self.ops['c_target_init'] = get_target_init_ops(self.ce_params, self.ct_params)

        self.ops['a_target_soft_replace'] = [tf.assign(ta, (1 - TAU) * ta + TAU * ea) for ta, ea in zip(self.at_params, self.ae_params)]
        self.ops['c_target_soft_replace'] = [tf.assign(tc, (1 - TAU) * tc + TAU * ec) for tc, ec in zip(self.ct_params, self.ce_params) ]
        print(len(self.ops['a_target_soft_replace']))
        print(len(self.ops['c_target_soft_replace']))
        
        td_error = tf.losses.mean_squared_error(labels = q_target, predictions = q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list = self.ce_params)
        
        a_loss = - tf.reduce_mean(q_for_a_loss)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list = self.ae_params)
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.ops['a_target_init'])
        self.sess.run(self.ops['c_target_init'])


    def finalize_rlout(self, rewards):
        """Finalize the current roll-out (to update the baseline function).

        Args:
        * rewards: reward scalars (one per roll-out tick)
        """

        # early return if baseline function is disabled
        if not enbl_bsln_func:
            return
        bsln_decy_rate = 0.95
        # update the baseline function
        if self.reward_ema is None:
            self.reward_ema = np.mean(rewards)
        else:
            self.reward_ema = bsln_decy_rate * self.reward_ema \
                + (1.0 - bsln_decy_rate) * np.mean(rewards)

    def get_perturb_op(self, model, model_noisy, param_noise_std):
        """Get operations for pertubing the model's parameters.

        Args:
        * model: original model
        * model_noisy: perturbed model
        * param_noise_std: standard deviation of the parameter noise

        Returns:
        * perturb_op: perturbation operation for the noisy model
        """

        perturb_ops = []
        for var_clean, var_noisy in zip(model, model_noisy):
            var_noise = tf.random_normal(tf.shape(var_clean), mean=0., stddev=param_noise_std)
            perturb_ops.append(tf.assign(var_noisy, var_clean + var_noise))

        return tf.group(*perturb_ops)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S : s[np.newaxis, :]})[0]

    def choose_noisy_action(self, s, variance):
        self.sess.run(self.ops['param_explore'], feed_dict = {self.noisy_variance : variance})
        a_noisy = self.sess.run(self.a_noisy, feed_dict = {self.S : s[np.newaxis, :]})[0]
        return a_noisy
        
    def learn(self):
        
        indices = np.random.choice(self.memory_capacity, size = BATCH_SIZE)
        bt = self.memory[indices, :]
        #bt = scale_arrays(bt)
        
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim : self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1 : -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        if enbl_bsln_func :
            br -= self.reward_ema
        self.sess.run(self.atrain, {self.S : bs})
        self.sess.run(self.ctrain, {self.S : bs, self.actions : ba, self.R : br, self.S_ : bs_})
        self.sess.run(self.ops['a_target_soft_replace'])
        self.sess.run(self.ops['c_target_soft_replace'])

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.stored_times % self.memory_capacity
        self.memory[index, :] = transition
        self.stored_times += 1
    def memory_full(self):
        if self.stored_times > self.memory_capacity : 
            return True
        else :
            return False
        
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            h1 = tf.layers.dense(s, 64, activation=None, name='h1', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.16), trainable = trainable)
            h1 = tf.contrib.layers.layer_norm(h1)
            h1 = tf.nn.relu(h1)
            h2 = tf.layers.dense(h1, 64, activation=None, name='h2', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.16), trainable = trainable)
            h2 = tf.contrib.layers.layer_norm(h1)
            h2 = tf.nn.relu(h2)
            a = tf.layers.dense(h2, self.a_dim, activation = tf.nn.sigmoid, name = 'a', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.16), trainable = trainable)
            return tf.add(tf.multiply(a, 0.98), 0.01, name = 'scaled_a')
            
    def _build_c(self, s, a, scope, trainable, reuse = False):
        with tf.variable_scope(scope) as scope:
            if reuse == True:
                scope.reuse_variables()
            #inputs = tf.layers.dense(s, 10, activation = tf.nn.relu, kernel_initializer = tf.truncated_normal_initializer(stddev = 0.19), trainable = trainable)
            inputs = tf.concat([s, a], axis = 1)
            h1 = tf.layers.dense(inputs, 64, activation = None, kernel_initializer = tf.truncated_normal_initializer(stddev = 0.16), trainable = trainable)
            h1 = tf.contrib.layers.layer_norm(h1)
            h1 = tf.nn.relu(h1)
            h2 = tf.layers.dense(h1, 64, activation = None, kernel_initializer = tf.truncated_normal_initializer(stddev = 0.16), trainable = trainable)
            h2 = tf.contrib.layers.layer_norm(h2)
            h2 = tf.nn.relu(h2)
            q = tf.layers.dense(h2, 1, activation = None, kernel_initializer = tf.truncated_normal_initializer(stddev = 0.16), trainable = trainable)
            return q
if __name__ == '__main__' :
    ddpg = DDPG(a_dim = 1, s_dim = 3)
    
