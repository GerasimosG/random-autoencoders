import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import math

class AE(object):
    def __init__(self, NN_architecture):        
        self.NN = NN_architecture
        self.dropOut = NN_architecture['dropOut']
        self.transfer_func_enc = NN_architecture['transfer_func_encoder']
        self.transfer_func_dec = NN_architecture['transfer_func_decoder']
        self.transfer_func_out = NN_architecture['transfer_func_output']
        self.dims = NN_architecture['dims']
        self.layersNum = len(NN_architecture['dims']) - 1
        self.W_init_var = NN_architecture['W_init_var']
        self.b_init_var = NN_architecture['b_init_var']
        self.v_init_var = NN_architecture['v_init_var']
        self.loss_choice = NN_architecture['loss_choice']
        self.is_batchnorm = NN_architecture['is_use_batchnorm']
        self.is_update_v = NN_architecture['is_train_decoder_bias']        
        if NN_architecture['transfer_func_mid']!='default':
            self.transfer_func_mid = NN_architecture['transfer_func_mid']
        else:
            self.transfer_func_mid = NN_architecture['transfer_func_encoder']
        GPU_memory_fraction = NN_architecture['GPU_memory_fraction']
        GPU_which = NN_architecture['GPU_which']
        Tensorflow_randomSeed = NN_architecture['Tensorflow_randomSeed']
        
        # reset graph     
        tf.reset_default_graph()     
        if Tensorflow_randomSeed is not None:
            tf.set_random_seed(Tensorflow_randomSeed)
        
        # create graph
        if GPU_which is not None:
            with tf.device('/device:GPU:' + str(GPU_which)):
                self._create_graph()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_memory_fraction)    
            config = tf.ConfigProto(gpu_options = gpu_options,\
                                    allow_soft_placement=True, log_device_placement=True)
        else:
            self._create_graph()            
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, \
                                device_count = {'GPU': 0})
        
        # initialize tf variables & launch the session
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        
        # to save model
        self.saver = tf.train.Saver()
        
    def _create_graph(self):        
        # create input to graph
        self.x = tf.placeholder(tf.float32, [None, self.dims[0]])
        self.learning_rate = tf.placeholder(tf.float32)
        self.l2reg = tf.placeholder(tf.float32)    
        self.is_training = tf.placeholder(tf.bool)

        # create graph with AE architecture
        self._create_AE()

        # create loss & optimizer
        self._create_loss_optimizer()

    def _create_AE(self):
        self.enc, self.dec, self.W, self.b, self.v, self.latent_preact, self.enc_pre, self.dec_pre = self._autoencoder()
        self.output = self.dec[str(1)]
        self.latent = self.enc[str(self.layersNum)]        
          
    def _create_loss_optimizer(self):              
        if self.loss_choice=='MSE':
            self.reconstr_loss = self._create_MSE()
        elif self.loss_choice=='NMSE':
            self.reconstr_loss = self._create_NMSE()
        elif self.loss_choice=='cross entropy':            
            self.reconstr_loss = self._create_crossEnt()
        self.reg_loss = 0
        for layers in range(1,self.layersNum+1):            
            self.reg_loss += self.l2reg * tf.reduce_mean(tf.square(self.W[str(layers)]))
        self.cost = self.reconstr_loss + self.reg_loss 
        
        # optimizer
        if self.is_batchnorm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=\
                                                           self.learning_rate).minimize(self.cost)    
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=\
                                                           self.learning_rate).minimize(self.cost)   
    
    
    def fit(self, X, learning_rate, l2reg):
        optimizer, cost = self.sess.run((self.optimizer, self.cost), 
                                        feed_dict={self.x: X, self.learning_rate: learning_rate, \
                                                   self.l2reg: l2reg, self.is_training: True})
        return cost
    
    def retrieve_cost(self, X, batch_size=1000):
        ind = 0
        cost = 0.0
        while ind < X.shape[0]:
            x = X[ind:(ind+batch_size)]
            loss_batch = self.sess.run(self.cost, feed_dict={self.x: x, self.is_training: False})
            cost += loss_batch*x.shape[0]
            ind = ind + x.shape[0]
        return cost/X.shape[0]
        
    def retrieve_reconstr_loss(self, X, batch_size=1000):
        ind = 0
        cost = 0.0
        while ind < X.shape[0]:
            x = X[ind:(ind+batch_size)]
            loss_batch = self.sess.run(self.reconstr_loss, feed_dict={self.x: x, self.is_training: False})
            cost += loss_batch*x.shape[0]
            ind = ind + x.shape[0]
        return cost/X.shape[0]
    
    def reconstruct(self, X, batch_size=1000):
        ind = 0
        pred = np.empty_like(X)
        while ind < X.shape[0]:
            x = X[ind:(ind+batch_size)]
            pred[ind:(ind+batch_size)] = self.sess.run(self.output, \
                                                       feed_dict={self.x: x, self.is_training: False})
            ind = ind + x.shape[0]
        return pred    
    
    def get_latent(self, X, batch_size=1000):  # get the latent's values for multiple inputs X
        ind = 0
        latent = np.empty([X.shape[0], self.dims[-1]])
        latent_preact = np.empty([X.shape[0], self.dims[-1]])
        while ind < X.shape[0]:
            x = X[ind:(ind+batch_size)]
            latent[ind:(ind+batch_size)], latent_preact[ind:(ind+batch_size)] = \
                self.sess.run((self.latent, self.latent_preact),\
                              feed_dict={self.x: x, self.is_training: False})
            ind = ind + x.shape[0]
        return latent, latent_preact
    
    def get_weight(self):  # retrieve weights' values
        return self.sess.run((self.W, self.b, self.v),feed_dict={})
    
    def get_neurons(self, x):  # get neurons' post-activation values and pre-activation values, for a single input x
        return self.sess.run((self.enc, self.enc_pre, self.dec, self.dec_pre), \
                             feed_dict={self.x: x, self.is_training: False})
    
    def save_model(self, path_to_saved_model):        
        self.saver.save(self.sess, path_to_saved_model)
        
    def load_model(self, path_to_saved_model):
        self.saver.restore(self.sess, path_to_saved_model)
            
        
    #"----------------------------------------------------------------------------------------------"
    #"------------------------------ Vanila autoencoder --------------------------------------------"
    #"----------------------------------------------------------------------------------------------"
    
    def _autoencoder(self):
        x = self.x
        
        # initialization
        W = dict()
        b = dict()
        v = dict()
        for layers in range(self.layersNum):
            W[str(layers+1)] = tf.Variable(self._weight_init(self.NN['dims'][layers], \
                                                       self.NN['dims'][layers+1], self.W_init_var))
            b[str(layers+1)] = tf.Variable(self._bias_init(self.NN['dims'][layers+1], self.b_init_var))
            if self.is_update_v:
                v[str(layers+1)] = tf.Variable(self._bias_init(self.NN['dims'][layers], self.v_init_var))
            else:
                v[str(layers+1)] = tf.zeros(self.NN['dims'][layers])
        
        # autoencoder with weight tying
        ## encoder:
        enc = dict()
        enc_pre = dict()
        for layers in range(1,self.layersNum+1):
            if layers==1:
                in_ = x
            else:
                in_ = enc[str(layers-1)]
            if layers==self.layersNum:
                enc[str(layers)], enc_pre[str(layers)] = \
                    self._fully_connected_layer(in_, W[str(layers)], b[str(layers)], \
                                                self.dropOut, self.transfer_func_mid)
            else:                
                enc[str(layers)], enc_pre[str(layers)] = \
                    self._fully_connected_layer(in_, W[str(layers)], b[str(layers)],\
                                                self.dropOut, self.transfer_func_enc)
                latent_preact = enc_pre[str(layers)]
                
        ## decoder:
        dec = dict()
        dec_pre = dict()
        for layers in range(self.layersNum, 0, -1):
            if layers==self.layersNum:
                in_ = enc[str(layers)]
            else:
                in_ = dec[str(layers+1)]
            if layers==1:                
                dec[str(layers)], dec_pre[str(layers)] = \
                    self._fully_connected_layer(in_, tf.transpose(W[str(layers)]), v[str(layers)], \
                                                None, self.transfer_func_out)
            else:
                dec[str(layers)], dec_pre[str(layers)] = \
                    self._fully_connected_layer(in_, tf.transpose(W[str(layers)]), v[str(layers)], \
                                                self.dropOut, self.transfer_func_dec)
        return enc, dec, W, b, v, latent_preact, enc_pre, dec_pre
        
    def _fully_connected_layer(self, input_, weight, bias=None, dropOut=None, \
                               transfer_func=None):            
        if bias is not None:
            preact = tf.matmul(input_, weight) + bias
        else:
            preact = tf.matmul(input_, weight)
        
        if transfer_func is not None:
            h = transfer_func(preact)
        else:
            h = preact
                    
        if dropOut is not None:
            h = tf.layers.dropout(h, dropOut, training=self.is_training)
                
        if self.is_batchnorm:
            h = tf.layers.batch_normalization(h, training=self.is_training)
        
        return h, preact
        
    def _weight_init(self, size_in, size_out, variance):
        return tf.random_normal([size_in, size_out], dtype=tf.float32)*np.sqrt(variance/size_in)
    
    def _bias_init(self, size_out, variance):
        return tf.random_normal([1, size_out], dtype=tf.float32)*np.sqrt(variance)
    
    def _create_MSE(self):
        MSE = tf.reduce_mean(tf.square(self.output - self.x))
        return MSE
            
    def _create_NMSE(self):
        NMSE = tf.reduce_mean(tf.divide(tf.reduce_sum(tf.square(self.output - self.x),axis=1),\
                                            tf.reduce_sum(tf.square(self.x),axis=1)))
        return NMSE
    
    def _create_crossEnt(self, epsilon=1e-8):
        y = tf.clip_by_value(self.output, epsilon, 1.0-epsilon)
        crossEnt = - tf.reduce_mean(self.x*tf.log(y) + (1.0-self.x)*tf.log(1.0-y))
        return crossEnt
        
        
def AEbuild(NN_architecture):
    return AE(NN_architecture)


def AEtrain(nn, X_train, X_val=None, X_test=None, learning_rate=1e-3, lr_decay_rate=1.0, l2reg=0.0, 
            maxIter=100000, batch_size=1, preIter=10, prebatch_size=1000,
            display_iters=[10, 100, 1000, 10000, 100000],
            is_verbose=True, is_save_model=False, model_path=None, model_key=None,
            save_model_iters = [10, 100, 1000, 10000, 100000, 1000000, 10000000]):
    
    # error if numTrain is not a multiple of batch_size
    numTrain = X_train.shape[0]
    if numTrain%batch_size > 0:
        print('Batch_size has to divide training size!!!')
        raise SystemExit(0)
    
    # flag if there is validate set
    if X_val is not None:
        flag_there_is_val = True
    else:
        flag_there_is_val = False
            
    # flag if there is test set
    if X_test is not None:
        flag_there_is_test = True
    else:
        flag_there_is_test = False
    
    if is_verbose:
        print('Iter | Time (min) | Learning rate | Train Rec Loss | Val Rec Loss | Test Rec Loss')
    
    # Now train
    time = timer()    
    train_loss_saved = np.array([])    
    val_loss_saved = np.array([])      
    test_loss_saved = np.array([])
    for iters in range(maxIter):
        # get training data
        # For the first preIter iterations, we use a batch of size prebatch_size, instead of batch_size        
        start = (iters*batch_size)
        end = ((iters+1)*batch_size)        
        if ((start//numTrain)>(((iters-1)*batch_size)//numTrain)) or (iters==0):
            train_drawIndex = np.random.permutation(numTrain)
        ind = train_drawIndex[np.arange(start,end)%numTrain]
        if iters<preIter:
            temp1 = np.array([train_drawIndex[r] for r in np.arange(numTrain) if \
                              train_drawIndex[r] not in ind], dtype=np.int32)
            temp2 = np.random.choice(temp1, prebatch_size-batch_size)            
            ind = np.append(ind, temp2)
        x = X_train[ind]        
            
        # train 
        nn.fit(x, learning_rate, l2reg)           
        
        # get the loss & display & save
        if (iters+1) in display_iters:
            train_loss = nn.retrieve_reconstr_loss(X_train, batch_size=2500)
            if flag_there_is_val:
                val_loss = nn.retrieve_reconstr_loss(X_val, batch_size=2500)
            else:
                val_loss = np.nan
            if flag_there_is_test:
                test_loss = nn.retrieve_reconstr_loss(X_test, batch_size=2500)
            else:
                test_loss = np.nan
                    
            time = timer() - time  
            
            if is_verbose:
                print('%08d' % (iters+1), 
                          "|",  '%.2f' % (time/60),
                          "|", '%.2e' % (learning_rate),
                          "|", '%.3e' % (train_loss),
                          "|", '%.3e' % (val_loss),
                          "|", '%.3e' % (test_loss),
                         )
                
            time = timer()
            
            # save
            train_loss_saved = np.append(train_loss_saved, train_loss)
            val_loss_saved = np.append(val_loss_saved, val_loss)
            test_loss_saved = np.append(test_loss_saved, test_loss)
            
        # save model to file
        if is_save_model and (iters+1) in save_model_iters:
            if model_key is None:
                nn.save_model(model_path + 'ae_iters' + str(iters+1))
            else:
                nn.save_model(model_path + 'ae_iters' + str(iters+1) + '_' + model_key)
            
        # Decay learning rate
        if (iters*batch_size)%numTrain == 0:
            learning_rate *= lr_decay_rate
                
    return nn, train_loss_saved, val_loss_saved, test_loss_saved
    