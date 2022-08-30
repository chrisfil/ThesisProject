from keras.layers import Input, Lambda, Dense, Flatten, Multiply, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, AveragePooling2D
from keras.layers import LeakyReLU, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l1
import keras.backend as K

from dcase_models.model.container import KerasModelContainer

from .losses import prototype_loss, dummy_loss
from .layers import PrototypeReconstruction


class AttProtos(KerasModelContainer):
    def __init__(self, model=None, model_path=None, metrics=['sed'],
                 n_classes=10, n_frames_cnn=64, n_freq_cnn=128,
                 filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                 n_prototypes=50, logits_activation='softmax',
                 dilation_rate=(1,1), distance='euclidean',
                 use_weighted_sum=True, N_filters = [32,32,32],
                 **kwargs):
 
        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn 
        self.n_freq_cnn = n_freq_cnn
        self.filter_size_cnn = filter_size_cnn 
        self.pool_size_cnn = pool_size_cnn
        self.n_prototypes = n_prototypes 
        self.logits_activation = logits_activation
        self.dilation_rate = dilation_rate
        self.distance = distance 
        self.use_weighted_sum = use_weighted_sum
        self.N_filters = N_filters      

        self.prototypes = None
        self.data_instances = None

        super().__init__(model=model, model_path=model_path,
                        model_name='AttProtos', metrics=metrics, **kwargs)


    def build(self):
        self.model_encoder = self.create_encoder()

        decoder_input_shape = self.model_encoder.get_layer('conv3').output_shape[1:]
        mask1_shape = self.model_encoder.get_layer('mask1').input_shape[1:]
        mask2_shape = self.model_encoder.get_layer('mask2').input_shape[1:]

        #self.model_decoder_mask = self.create_decoder_mask(
        #    decoder_input_shape, name='decoder_mask',
        #    final_activation='softmax', N_filters_out=self.n_prototypes)

        self.model_encoder_mask = self.create_encoder_mask(
            N_filters_out=self.n_prototypes, name='encoder_mask',
        )
        #    decoder_input_shape, name='decoder_mask',
        #    final_activation='softmax', N_filters_out=self.n_prototypes)

        self.model_decoder = self.create_decoder(decoder_input_shape, mask1_shape, mask2_shape)

        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')

        h, mask1, mask2 = self.model_encoder(x)
        alpha = self.model_encoder_mask(x)

        h_hat, distance = PrototypeReconstruction(self.n_prototypes, name='lc')([h, alpha])
        x_hat = self.model_decoder([h, mask1, mask2])

        alpha = Lambda(lambda x: K.sum(x, axis=1), name='sum')(alpha)
        alpha = Flatten()(alpha)
        dense = Dense(self.n_classes, use_bias=False, name='dense', kernel_regularizer=l1(0.00001)) #
        logits = dense(alpha)

        out = Activation(activation=self.logits_activation, name='out')(logits)

        # mse = Lambda(lambda x: K.mean(K.pow(x[0] - x[1], 2), axis=(1,2,3)), name='mse')([h, h_hat])
        # x_hat2 = self.model_decoder([h_hat, mask1, mask2])
        # mse2 = Lambda(lambda x: K.mean(K.pow(x[0] - x[1], 2), axis=(1,2)), name='mse2')([x_hat, x_hat2])
        # mse = Lambda(lambda x: x[0] + x[1], name='mse3')([mse, mse2])

        mse = Lambda(lambda x: K.mean(K.sum(K.pow(x[0] - x[1], 2), axis=(2,3))), name='mse')([h, h_hat])
        x_hat2 = self.model_decoder([h_hat, mask1, mask2])
        mse2 = Lambda(lambda x: K.mean(K.sum(K.pow(x[0] - x[1], 2), axis=2)), name='mse2')([x_hat, x_hat2])
        mse = Lambda(lambda x: x[0] + x[1], name='mse3')([mse, mse2])

        self.model = Model(inputs=x, outputs=[out, x_hat, distance, mse])

    def model_h_hat(self):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        h, mask1, mask2 = self.model_encoder(x)
        #alpha = self.model_decoder_mask(h)
        alpha = self.model_encoder_mask(x)
        h_hat, distance = self.model.get_layer('lc')([h, alpha])
        model = Model(x, [h_hat, distance])
        return model


    def train(self, data_train, data_val, weights_path='./',
              optimizer='Adam', learning_rate=0.001, early_stopping=100,
              considered_improvement=0.01,
              loss_weights=[10,5,5,5], sequence_time_sec=0.5,
              metric_resolution_sec=1.0, label_list=[],
              shuffle=True, init_last_layer=False, loss_classification='categorical_crossentropy',
              **kwargs_keras_fit):

        """
        Specific training function for APNet model
        """
        n_classes = len(label_list)
        # define optimizer and compile
        losses = [
            loss_classification,
            'mean_squared_error',
            prototype_loss,
            dummy_loss
        ]

        super().train(
            data_train, data_val,
            weights_path=weights_path, optimizer=optimizer,
            learning_rate=learning_rate, early_stopping=early_stopping,
            considered_improvement=considered_improvement,
            losses=losses, loss_weights=loss_weights,
            sequence_time_sec=sequence_time_sec,
            metric_resolution_sec=metric_resolution_sec,
            label_list=label_list, shuffle=shuffle,
            **kwargs_keras_fit
        )

    def create_encoder(self, activation = 'linear', name='encoder', use_batch_norm=False):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        y = Lambda(lambda x: K.expand_dims(x,-1), name='expand_dims')(x)
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='conv1', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = LeakyReLU(name='leaky_relu1')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling1')(y)
        bool_mask1 = Lambda(lambda t: K.greater_equal(t[0], t[1]), name='bool_mask1')([orig, y_up])
        mask1 = Lambda(lambda t: K.cast(t, dtype='float32'), name='mask1')(bool_mask1)

        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='conv2', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = LeakyReLU(name='leaky_relu2')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling2')(y)
        bool_mask2 = Lambda(lambda t: K.greater_equal(t[0], t[1]), name='bool_mask2')([orig, y_up])
        mask2 = Lambda(lambda t: K.cast(t, dtype='float32'), name='mask2')(bool_mask2)    
        
        y = Conv2D(self.N_filters[2], self.filter_size_cnn, padding='same', activation=activation, name='conv3')(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = LeakyReLU(name='leaky_relu3')(y)

        model = Model(inputs=x, outputs=[y, mask1, mask2], name=name)
        return model

    def create_decoder(self, input_shape, mask1_shape, mask2_shape,
                       activation = 'linear', final_activation='tanh',
                       name='decoder', use_batch_norm=False, N_filters_out=1):
    
        x = Input(shape=input_shape, dtype='float32', name='input')
        mask1 = Input(shape=mask1_shape, dtype='float32', name= 'input_mask1')
        mask2 = Input(shape=mask2_shape, dtype='float32', name= 'input_mask2')

        deconv = Conv2DTranspose(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='deconv1')(x) 
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling2_1')(deconv)
        deconv = Multiply(name='multiply2')([mask2, deconv]) 
        deconv = LeakyReLU(name='leaky_relu4')(deconv)

        deconv = Conv2DTranspose(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='deconv2')(deconv)     
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling3_1')(deconv)
        deconv = Multiply(name='multiply3')([mask1, deconv])    
        deconv = LeakyReLU(name='leaky_relu5')(deconv)

        deconv = Conv2DTranspose(N_filters_out, self.filter_size_cnn, padding='same', activation='linear', name='deconv3')(deconv)

        if N_filters_out == 1:
            deconv = Lambda(lambda x: K.squeeze(x, axis=-1), name='input_reconstructed')(deconv)

        if use_batch_norm:
            deconv = BatchNormalization()(deconv) 
        deconv = Activation(final_activation, name='final_activation')(deconv)

        model = Model(inputs=[x, mask1, mask2], outputs=deconv, name=name)        
        return model

    def create_decoder_mask(self, input_shape,activation = 'linear', final_activation='tanh',
                       name='decoder', use_batch_norm=False, N_filters_out=1):
    
        x = Input(shape=input_shape, dtype='float32', name='input')

        deconv = Conv2DTranspose(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='deconv1')(x) 
        deconv = LeakyReLU(name='leaky_relu4')(deconv)

        deconv = Conv2DTranspose(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='deconv2')(deconv)      
        deconv = LeakyReLU(name='leaky_relu5')(deconv)

        deconv = Conv2DTranspose(N_filters_out, self.filter_size_cnn, padding='same', activation='linear', name='deconv3', activity_regularizer=l1(0.000001))(deconv)

        if N_filters_out == 1:
            deconv = Lambda(lambda x: K.squeeze(x, axis=-1), name='input_reconstructed')(deconv)

        if use_batch_norm:
            deconv = BatchNormalization()(deconv) 

        deconv = Activation(final_activation, name='final_activation')(deconv)

        model = Model(inputs=x, outputs=deconv, name=name)        
        return model

    def create_encoder_mask(self, activation = 'linear', name='encoder', use_batch_norm=False, N_filters_out=1):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        y = Lambda(lambda x: K.expand_dims(x,-1), name='expand_dims')(x)
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='conv1', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        #y = LeakyReLU(name='leaky_relu1')(y)
        y = ReLU(name='leaky_relu1')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)

        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='conv2', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        #y = LeakyReLU(name='leaky_relu2')(y)
        y = ReLU(name='leaky_relu2')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)
        
        y = Conv2D(N_filters_out, self.filter_size_cnn, padding='same', activation=activation, name='conv3', activity_regularizer=l1(0.000001))(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        #y = LeakyReLU(name='leaky_relu3')(y)
        y = ReLU(name='leaky_relu3')(y)

        #y = Softmax()(y)

        model = Model(inputs=x, outputs=y, name=name)
        return model


class AttProtos2(KerasModelContainer):
    def __init__(self, model=None, model_path=None, metrics=['sed'],
                 n_classes=10, n_frames_cnn=64, n_freq_cnn=128,
                 filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                 n_prototypes=50, logits_activation='softmax',
                 dilation_rate=(1,1), distance='euclidean', N_filters_out=1,
                 use_weighted_sum=True, N_filters = [32,32,32], training=True,
                 **kwargs):
 
        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn 
        self.n_freq_cnn = n_freq_cnn
        self.filter_size_cnn = filter_size_cnn 
        self.pool_size_cnn = pool_size_cnn
        self.n_prototypes = n_prototypes 
        self.logits_activation = logits_activation
        self.dilation_rate = dilation_rate
        self.distance = distance 
        self.use_weighted_sum = use_weighted_sum
        self.N_filters = N_filters  
        self.N_filters_out = N_filters_out
        self.training = training
        #self.use_batch_norm = use_batch_norm
        
        self.prototypes = None
        self.data_instances = None

        super().__init__(model=model, model_path=model_path,
                        model_name='AttProtos2', metrics=metrics, **kwargs)


    def build(self):
        self.model_encoder = self.create_encoder()

        decoder_input_shape = self.model_encoder.get_layer('conv3').output_shape[1:]
        mask1_shape = self.model_encoder.get_layer('mask1').input_shape[1:]
        mask2_shape = self.model_encoder.get_layer('mask2').input_shape[1:]

        #self.model_decoder_mask = self.create_decoder_mask(
        #    decoder_input_shape, name='decoder_mask',
        #    final_activation='softmax', N_filters_out=self.n_prototypes)

        #self.model_encoder_mask = self.create_encoder_mask(
        #    N_filters_out=self.n_prototypes, name='encoder_mask',
        #)
        #decoder_input_shape, name='decoder_mask',
        #final_activation='softmax', N_filters_out=self.n_prototypes)

        
        
        
        self.model_decoder = self.create_decoder(decoder_input_shape, mask1_shape, mask2_shape)
        
        training = self.training
        
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        #print('x encoder mask', x.shape)
        
        #####################################
        #MODEL ENCODER MASK
        #y = Reshape((self.n_frames_cnn, self.n_freq_cnn, 1))(x)
        y = Reshape((128, 256, 1))(x)
        print('y encoder mask', y.shape)
        #y = Lambda(lambda x: K.expand_dims(x,-1), name='expand_dims')(x)
        
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation='linear', name='conv1', dilation_rate=self.dilation_rate)(y)
        #if use_batch_norm:
        #    y = BatchNormalization()(y) 
        y = ReLU(name='relu1')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)

        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation='linear', name='conv2', dilation_rate=self.dilation_rate)(y)
        #if use_batch_norm:
        #    y = BatchNormalization()(y) 
        y = ReLU(name='relu2')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)

        y = Conv2D(self.n_prototypes, self.filter_size_cnn, padding='same', activation='linear', name='conv3', activity_regularizer=l1(0.000001))(y)
        #if use_batch_norm:
        #    y = BatchNormalization()(y) 
        y = ReLU(name='relu3')(y)
            
        
        #alpha = self.model_encoder_mask(x)
        alpha = y
        #####################################
        h, mask1, mask2 = self.model_encoder(x)
        h_hat, distance = PrototypeReconstruction(self.n_prototypes, name='lc')([h, alpha])
        x_hat = self.model_decoder([h, mask1, mask2])
        print('alpha 1:', alpha.shape)
        #alpha = Lambda(lambda x: K.sum(x, axis=1), name='sum')(alpha)
        
        alpha = AveragePooling2D(pool_size=(int(alpha.shape[1]), 1), name="mean")(alpha)
        #print(alpha.shape)
        #print('pooling shape:', alpha.shape)
        
        #alpha = alpha[:,-1,:,:]
        #print(alpha)
        #alpha = Flatten(name="flatten")(alpha)
        #alpha = Reshape((1,1))(alpha)
        print('alpha 2:', alpha.shape)
        alpha = Flatten()(alpha)
        print('alpha 3:', alpha.shape)
        dense = Dense(self.n_classes, use_bias=False, name='dense', kernel_regularizer=l1(0.00001)) #
        logits = dense(alpha)

        out = Activation(activation=self.logits_activation, name='out')(logits)

        # mse = Lambda(lambda x: K.mean(K.pow(x[0] - x[1], 2), axis=(1,2,3)), name='mse')([h, h_hat])
        # x_hat2 = self.model_decoder([h_hat, mask1, mask2])
        # mse2 = Lambda(lambda x: K.mean(K.pow(x[0] - x[1], 2), axis=(1,2)), name='mse2')([x_hat, x_hat2])
        # mse = Lambda(lambda x: x[0] + x[1], name='mse3')([mse, mse2])

        mse = Lambda(lambda x: K.mean(K.sum(K.pow(x[0] - x[1], 2), axis=(2,3))), name='mse')([h, h_hat])
        x_hat2 = self.model_decoder([h_hat, mask1, mask2])
        mse2 = Lambda(lambda x: K.mean(K.sum(K.pow(x[0] - x[1], 2), axis=2)), name='mse2')([x_hat, x_hat2])
        mse = Lambda(lambda x: x[0] + x[1], name='mse3')([mse, mse2])
        #output = tf.concat([out, x_hat, distance, mse], 0)
        if training==True:
            self.model = Model(inputs=x, outputs=[out, x_hat, distance, mse])
            print(self.model.outputs)
        else:
            self.model = Model(inputs=x, outputs=out)
            #print(model.outputs)
        

    def model_h_hat(self):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        h, mask1, mask2 = self.model_encoder(x)
        #alpha = self.model_decoder_mask(h)
        ###################################
        #y = Reshape((self.n_frames_cnn, self.n_freq_cnn, 1))(x)
        y = Reshape((128, 256, 1))(x)
        print('y encoder mask', y.shape)
        
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation='linear', name='conv1', dilation_rate=self.dilation_rate)(y)
        y = ReLU(name='relu1')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)

        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation='linear', name='conv2', dilation_rate=self.dilation_rate)(y) 
        y = ReLU(name='relu2')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)

        y = Conv2D(self.n_prototypes, self.filter_size_cnn, padding='same', activation='linear', name='conv3', activity_regularizer=l1(0.000001))(y)
        y = ReLU(name='relu3')(y)
        ###################################    
        alpha = y
        #alpha = self.model_encoder_mask(x)
        h_hat, distance = self.model.get_layer('lc')([h, alpha])
        model = Model(x, [h_hat, distance])
        return model


    def train(self, data_train, data_val, weights_path='./',
              optimizer='Adam', learning_rate=0.001, early_stopping=100,
              considered_improvement=0.01,
              loss_weights=[10,5,5,5], sequence_time_sec=0.5,
              metric_resolution_sec=1.0, label_list=[],
              shuffle=True, init_last_layer=False, loss_classification='categorical_crossentropy',
              **kwargs_keras_fit):

        """
        Specific training function for APNet model
        """
        n_classes = len(label_list)
        # define optimizer and compile
        losses = [
            loss_classification,
            'mean_squared_error',
            prototype_loss,
            dummy_loss
        ]

        super().train(
            data_train, data_val,
            weights_path=weights_path, optimizer=optimizer,
            learning_rate=learning_rate, early_stopping=early_stopping,
            considered_improvement=considered_improvement,
            losses=losses, loss_weights=loss_weights,
            sequence_time_sec=sequence_time_sec,
            metric_resolution_sec=metric_resolution_sec,
            label_list=label_list, shuffle=shuffle,
            **kwargs_keras_fit
        )

    def create_encoder(self, activation = 'linear', name='encoder', use_batch_norm=False):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        print('encoder x', x.shape)
        #y = Reshape((self.n_frames_cnn, self.n_freq_cnn, 1))(x)
        y = Reshape((128, 256, 1))(x)
        #y = Lambda(lambda x: K.expand_dims(x,-1), name='expand_dims')(x)
        print('encoder y', y.shape)
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='conv1', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = ReLU(name='leaky_relu1')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling1')(y)
        bool_mask1 = Lambda(lambda t: K.greater_equal(t[0], t[1]), name='bool_mask1')([orig, y_up])
        #print('bool_mask1', bool_mask1)
        print('bool_mask1 shape', bool_mask1.shape)
        mask1 = Lambda(lambda t: K.cast(t, dtype='float32'), name='mask1')(bool_mask1)
        #print('mask1', mask1)
        print('mask1 shape', mask1.shape)
        
        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='conv2', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = ReLU(name='leaky_relu2')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling2')(y)
        bool_mask2 = Lambda(lambda t: K.greater_equal(t[0], t[1]), name='bool_mask2')([orig, y_up])
        mask2 = Lambda(lambda t: K.cast(t, dtype='float32'), name='mask2')(bool_mask2)    
        
        y = Conv2D(self.N_filters[2], self.filter_size_cnn, padding='same', activation=activation, name='conv3')(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = ReLU(name='leaky_relu3')(y)

        model = Model(inputs=x, outputs=[y, mask1, mask2], name=name)
        return model

    def create_decoder(self, input_shape, mask1_shape, mask2_shape,
                       activation = 'linear', final_activation='tanh',
                       name='decoder', use_batch_norm=False, N_filters_out=1):
    
        x = Input(shape=input_shape, dtype='float32', name='input')
        mask1 = Input(shape=mask1_shape, dtype='float32', name= 'input_mask1')
        mask2 = Input(shape=mask2_shape, dtype='float32', name= 'input_mask2')

        deconv = Conv2DTranspose(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='deconv1')(x) 
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling2_1')(deconv)
        deconv = Multiply(name='multiply2')([mask2, deconv]) 
        deconv = ReLU(name='leaky_relu4')(deconv)

        deconv = Conv2DTranspose(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='deconv2')(deconv)     
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling3_1')(deconv)
        deconv = Multiply(name='multiply3')([mask1, deconv])    
        deconv = ReLU(name='leaky_relu5')(deconv)

        deconv = Conv2DTranspose(N_filters_out, self.filter_size_cnn, padding='same', activation='linear', name='deconv3')(deconv)
        print('1:deconv', deconv.shape)
        

        
        if N_filters_out == 1:
            #deconv = Lambda(lambda x: K.squeeze(x, axis=-1), name='input_reconstructed')(deconv)
            #shape = (self.n_frames_cnn, self.n_freq_cnn)
            deconv = Reshape((self.n_frames_cnn, self.n_freq_cnn), name='input_reconstructed')(deconv)
            print('2:deconv', deconv.shape)
        if use_batch_norm:
            deconv = BatchNormalization()(deconv) 
        deconv = Activation(final_activation, name='final_activation')(deconv)

        model = Model(inputs=[x, mask1, mask2], outputs=deconv, name=name)        
        return model

    def create_decoder_mask(self, input_shape,activation = 'linear', final_activation='tanh',
                       name='decoder', use_batch_norm=False, N_filters_out=1):
    
        x = Input(shape=input_shape, dtype='float32', name='input')

        deconv = Conv2DTranspose(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='deconv1')(x) 
        deconv = ReLU(name='leaky_relu4')(deconv)

        deconv = Conv2DTranspose(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='deconv2')(deconv)      
        deconv = ReLU(name='leaky_relu5')(deconv)

        deconv = Conv2DTranspose(N_filters_out, self.filter_size_cnn, padding='same', activation='linear', name='deconv3', activity_regularizer=l1(0.000001))(deconv)
        print('3:deconv', deconv.shape)
        if N_filters_out == 1:
            #deconv = Lambda(lambda x: K.squeeze(x, axis=-1), name='input_reconstructed')(deconv)
            #deconv = tf.squeeze(deconv, [-1], name='input_reconstructed')
            #deconv = Reshape((-1, 1))(deconv)
            deconv = Reshape((self.n_frames_cnn, self.n_freq_cnn), name='input_reconstructed')(deconv)
            print('4:deconv', deconv.shape)
        if use_batch_norm:
            deconv = BatchNormalization()(deconv) 

        deconv = Activation(final_activation, name='final_activation')(deconv)

        model = Model(inputs=x, outputs=deconv, name=name)        
        return model

    
class AttProtos3(KerasModelContainer):
    def __init__(self, model=None, model_path=None, metrics=['classification'],
                 n_classes=10, n_frames_cnn=64, n_freq_cnn=128,
                 filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                 n_prototypes=50, logits_activation='sigmoid',
                 dilation_rate=(1,1), distance='euclidean',
                 use_weighted_sum=True, N_filters = [32,32,32],
                 **kwargs):
 
        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn 
        self.n_freq_cnn = n_freq_cnn
        self.filter_size_cnn = filter_size_cnn 
        self.pool_size_cnn = pool_size_cnn
        self.n_prototypes = n_prototypes 
        self.logits_activation = logits_activation
        self.dilation_rate = dilation_rate
        self.distance = distance 
        self.use_weighted_sum = use_weighted_sum
        self.N_filters = N_filters      

        self.prototypes = None
        self.data_instances = None

        super().__init__(model=model, model_path=model_path,
                        model_name='AttProtos3', metrics=metrics, **kwargs)


    def build(self):
        self.model_encoder = self.create_encoder()

        decoder_input_shape = self.model_encoder.get_layer('conv3').output_shape[1:]
        mask1_shape = self.model_encoder.get_layer('mask1').input_shape[1:]
        mask2_shape = self.model_encoder.get_layer('mask2').input_shape[1:]

        #self.model_decoder_mask = self.create_decoder_mask(
        #    decoder_input_shape, name='decoder_mask',
        #    final_activation='softmax', N_filters_out=self.n_prototypes)

        self.model_encoder_mask = self.create_encoder_mask(
            N_filters_out=self.n_prototypes, name='encoder_mask',
        )
        #    decoder_input_shape, name='decoder_mask',
        #    final_activation='softmax', N_filters_out=self.n_prototypes)

        self.model_decoder = self.create_decoder(decoder_input_shape, mask1_shape, mask2_shape)

        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')

        h, mask1, mask2 = self.model_encoder(x)
        alpha = self.model_encoder_mask(x)

        h_hat, distance = PrototypeReconstruction(self.n_prototypes, name='lc')([h, alpha])
        x_hat = self.model_decoder([h, mask1, mask2])

        alpha = Lambda(lambda x: K.sum(x, axis=1), name='sum')(alpha)
        alpha = Flatten()(alpha)
        dense = Dense(self.n_classes, use_bias=False, name='dense', kernel_regularizer=l1(0.00001)) #
        logits = dense(alpha)

        out = Activation(activation=self.logits_activation, name='out')(logits)

        # mse = Lambda(lambda x: K.mean(K.pow(x[0] - x[1], 2), axis=(1,2,3)), name='mse')([h, h_hat])
        # x_hat2 = self.model_decoder([h_hat, mask1, mask2])
        # mse2 = Lambda(lambda x: K.mean(K.pow(x[0] - x[1], 2), axis=(1,2)), name='mse2')([x_hat, x_hat2])
        # mse = Lambda(lambda x: x[0] + x[1], name='mse3')([mse, mse2])

        mse = Lambda(lambda x: K.mean(K.sum(K.pow(x[0] - x[1], 2), axis=(2,3))), name='mse')([h, h_hat])
        x_hat2 = self.model_decoder([h_hat, mask1, mask2])
        mse2 = Lambda(lambda x: K.mean(K.sum(K.pow(x[0] - x[1], 2), axis=2)), name='mse2')([x_hat, x_hat2])
        mse = Lambda(lambda x: x[0] + x[1], name='mse3')([mse, mse2])

        self.model = Model(inputs=x, outputs=[out, x_hat, distance, mse])

    def model_h_hat(self):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        h, mask1, mask2 = self.model_encoder(x)
        #alpha = self.model_decoder_mask(h)
        alpha = self.model_encoder_mask(x)
        h_hat, distance = self.model.get_layer('lc')([h, alpha])
        model = Model(x, [h_hat, distance])
        return model


    def train(self, data_train, data_val, weights_path='./',
              optimizer='Adam', learning_rate=0.001, early_stopping=100,
              considered_improvement=0.01,
              loss_weights=[10,5,5,5], sequence_time_sec=0.5,
              metric_resolution_sec=1.0, label_list=[],
              shuffle=True, init_last_layer=False, loss_classification='categorical_crossentropy',
              **kwargs_keras_fit):

        """
        Specific training function for APNet model
        """
        n_classes = len(label_list)
        # define optimizer and compile
        losses = [
            loss_classification,
            'mean_squared_error',
            prototype_loss,
            dummy_loss
        ]

        super().train(
            data_train, data_val,
            weights_path=weights_path, optimizer=optimizer,
            learning_rate=learning_rate, early_stopping=early_stopping,
            considered_improvement=considered_improvement,
            losses=losses, loss_weights=loss_weights,
            sequence_time_sec=sequence_time_sec,
            metric_resolution_sec=metric_resolution_sec,
            label_list=label_list, shuffle=shuffle,
            **kwargs_keras_fit
        )

    def create_encoder(self, activation = 'linear', name='encoder', use_batch_norm=False):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        y = Lambda(lambda x: K.expand_dims(x,-1), name='expand_dims')(x)
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='conv1', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = LeakyReLU(name='leaky_relu1')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling1')(y)
        bool_mask1 = Lambda(lambda t: K.greater_equal(t[0], t[1]), name='bool_mask1')([orig, y_up])
        mask1 = Lambda(lambda t: K.cast(t, dtype='float32'), name='mask1')(bool_mask1)

        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='conv2', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = LeakyReLU(name='leaky_relu2')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling2')(y)
        bool_mask2 = Lambda(lambda t: K.greater_equal(t[0], t[1]), name='bool_mask2')([orig, y_up])
        mask2 = Lambda(lambda t: K.cast(t, dtype='float32'), name='mask2')(bool_mask2)    
        
        y = Conv2D(self.N_filters[2], self.filter_size_cnn, padding='same', activation=activation, name='conv3')(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = LeakyReLU(name='leaky_relu3')(y)

        model = Model(inputs=x, outputs=[y, mask1, mask2], name=name)
        return model

    def create_decoder(self, input_shape, mask1_shape, mask2_shape,
                       activation = 'linear', final_activation='tanh',
                       name='decoder', use_batch_norm=False, N_filters_out=1):
    
        x = Input(shape=input_shape, dtype='float32', name='input')
        mask1 = Input(shape=mask1_shape, dtype='float32', name= 'input_mask1')
        mask2 = Input(shape=mask2_shape, dtype='float32', name= 'input_mask2')

        deconv = Conv2DTranspose(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='deconv1')(x) 
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling2_1')(deconv)
        deconv = Multiply(name='multiply2')([mask2, deconv]) 
        deconv = LeakyReLU(name='leaky_relu4')(deconv)

        deconv = Conv2DTranspose(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='deconv2')(deconv)     
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling3_1')(deconv)
        deconv = Multiply(name='multiply3')([mask1, deconv])    
        deconv = LeakyReLU(name='leaky_relu5')(deconv)

        deconv = Conv2DTranspose(N_filters_out, self.filter_size_cnn, padding='same', activation='linear', name='deconv3')(deconv)

        if N_filters_out == 1:
            deconv = Lambda(lambda x: K.squeeze(x, axis=-1), name='input_reconstructed')(deconv)

        if use_batch_norm:
            deconv = BatchNormalization()(deconv) 
        deconv = Activation(final_activation, name='final_activation')(deconv)

        model = Model(inputs=[x, mask1, mask2], outputs=deconv, name=name)        
        return model

    def create_decoder_mask(self, input_shape,activation = 'linear', final_activation='tanh',
                       name='decoder', use_batch_norm=False, N_filters_out=1):
    
        x = Input(shape=input_shape, dtype='float32', name='input')

        deconv = Conv2DTranspose(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='deconv1')(x) 
        deconv = LeakyReLU(name='leaky_relu4')(deconv)

        deconv = Conv2DTranspose(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='deconv2')(deconv)      
        deconv = LeakyReLU(name='leaky_relu5')(deconv)

        deconv = Conv2DTranspose(N_filters_out, self.filter_size_cnn, padding='same', activation='linear', name='deconv3', activity_regularizer=l1(0.000001))(deconv)

        if N_filters_out == 1:
            deconv = Lambda(lambda x: K.squeeze(x, axis=-1), name='input_reconstructed')(deconv)

        if use_batch_norm:
            deconv = BatchNormalization()(deconv) 

        deconv = Activation(final_activation, name='final_activation')(deconv)

        model = Model(inputs=x, outputs=deconv, name=name)        
        return model

    def create_encoder_mask(self, activation = 'linear', name='encoder', use_batch_norm=False, N_filters_out=1):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        y = Lambda(lambda x: K.expand_dims(x,-1), name='expand_dims')(x)
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='conv1', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        #y = LeakyReLU(name='leaky_relu1')(y)
        y = ReLU(name='leaky_relu1')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)

        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='conv2', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        #y = LeakyReLU(name='leaky_relu2')(y)
        y = ReLU(name='leaky_relu2')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)
        
        y = Conv2D(N_filters_out, self.filter_size_cnn, padding='same', activation=activation, name='conv3', activity_regularizer=l1(0.000001))(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        #y = LeakyReLU(name='leaky_relu3')(y)
        y = ReLU(name='leaky_relu3')(y)

        #y = Softmax()(y)

        model = Model(inputs=x, outputs=y, name=name)
        return model



class AttProtos4(KerasModelContainer):
    def __init__(self, model=None, model_path=None, metrics=['classification'],
                 n_classes=10, n_frames_cnn=64, n_freq_cnn=128,
                 filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                 n_prototypes=50, logits_activation='sigmoid',
                 dilation_rate=(1,1), distance='euclidean', N_filters_out=1,
                 use_weighted_sum=True, N_filters = [32,32,32], training=True,
                 **kwargs):
 
        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn 
        self.n_freq_cnn = n_freq_cnn
        self.filter_size_cnn = filter_size_cnn 
        self.pool_size_cnn = pool_size_cnn
        self.n_prototypes = n_prototypes 
        self.logits_activation = logits_activation
        self.dilation_rate = dilation_rate
        self.distance = distance 
        self.use_weighted_sum = use_weighted_sum
        self.N_filters = N_filters  
        self.N_filters_out = N_filters_out
        self.training = training
        #self.use_batch_norm = use_batch_norm
        
        self.prototypes = None
        self.data_instances = None

        super().__init__(model=model, model_path=model_path,
                        model_name='AttProtos4', metrics=metrics, **kwargs)


    def build(self):
        self.model_encoder = self.create_encoder()

        decoder_input_shape = self.model_encoder.get_layer('conv3').output_shape[1:]
        mask1_shape = self.model_encoder.get_layer('mask1').input_shape[1:]
        mask2_shape = self.model_encoder.get_layer('mask2').input_shape[1:]

        #self.model_decoder_mask = self.create_decoder_mask(
        #    decoder_input_shape, name='decoder_mask',
        #    final_activation='softmax', N_filters_out=self.n_prototypes)

        #self.model_encoder_mask = self.create_encoder_mask(
        #    N_filters_out=self.n_prototypes, name='encoder_mask',
        #)
        #decoder_input_shape, name='decoder_mask',
        #final_activation='softmax', N_filters_out=self.n_prototypes)

        
        
        
        self.model_decoder = self.create_decoder(decoder_input_shape, mask1_shape, mask2_shape)
        
        training = self.training
        
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        #print('x encoder mask', x.shape)
        
        #####################################
        #MODEL ENCODER MASK
        #y = Reshape((self.n_frames_cnn, self.n_freq_cnn, 1))(x)
        y = Reshape((128, 256, 1))(x)
        print('y encoder mask', y.shape)
        #y = Lambda(lambda x: K.expand_dims(x,-1), name='expand_dims')(x)
        
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation='linear', name='conv1', dilation_rate=self.dilation_rate)(y)
        #if use_batch_norm:
        #    y = BatchNormalization()(y) 
        y = ReLU(name='relu1')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)

        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation='linear', name='conv2', dilation_rate=self.dilation_rate)(y)
        #if use_batch_norm:
        #    y = BatchNormalization()(y) 
        y = ReLU(name='relu2')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)

        y = Conv2D(self.n_prototypes, self.filter_size_cnn, padding='same', activation='linear', name='conv3', activity_regularizer=l1(0.000001))(y)
        #if use_batch_norm:
        #    y = BatchNormalization()(y) 
        y = ReLU(name='relu3')(y)
            
        
        #alpha = self.model_encoder_mask(x)
        alpha = y
        #####################################
        h, mask1, mask2 = self.model_encoder(x)
        h_hat, distance = PrototypeReconstruction(self.n_prototypes, name='lc')([h, alpha])
        x_hat = self.model_decoder([h, mask1, mask2])
        print('alpha 1:', alpha.shape)
        #alpha = Lambda(lambda x: K.sum(x, axis=1), name='sum')(alpha)
        
        alpha = AveragePooling2D(pool_size=(int(alpha.shape[1]), 1), name="mean")(alpha)
        #print(alpha.shape)
        #print('pooling shape:', alpha.shape)
        
        #alpha = alpha[:,-1,:,:]
        #print(alpha)
        #alpha = Flatten(name="flatten")(alpha)
        #alpha = Reshape((1,1))(alpha)
        print('alpha 2:', alpha.shape)
        alpha = Flatten()(alpha)
        print('alpha 3:', alpha.shape)
        dense = Dense(self.n_classes, use_bias=False, name='dense', kernel_regularizer=l1(0.00001)) #
        logits = dense(alpha)

        out = Activation(activation=self.logits_activation, name='out')(logits)

        # mse = Lambda(lambda x: K.mean(K.pow(x[0] - x[1], 2), axis=(1,2,3)), name='mse')([h, h_hat])
        # x_hat2 = self.model_decoder([h_hat, mask1, mask2])
        # mse2 = Lambda(lambda x: K.mean(K.pow(x[0] - x[1], 2), axis=(1,2)), name='mse2')([x_hat, x_hat2])
        # mse = Lambda(lambda x: x[0] + x[1], name='mse3')([mse, mse2])

        mse = Lambda(lambda x: K.mean(K.sum(K.pow(x[0] - x[1], 2), axis=(2,3))), name='mse')([h, h_hat])
        x_hat2 = self.model_decoder([h_hat, mask1, mask2])
        mse2 = Lambda(lambda x: K.mean(K.sum(K.pow(x[0] - x[1], 2), axis=2)), name='mse2')([x_hat, x_hat2])
        mse = Lambda(lambda x: x[0] + x[1], name='mse3')([mse, mse2])
        #output = tf.concat([out, x_hat, distance, mse], 0)
        if training==True:
            self.model = Model(inputs=x, outputs=[out, x_hat, distance, mse])
            print(self.model.outputs)
        else:
            self.model = Model(inputs=x, outputs=out)
            #print(model.outputs)
        

    def model_h_hat(self):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        h, mask1, mask2 = self.model_encoder(x)
        #alpha = self.model_decoder_mask(h)
        ###################################
        #y = Reshape((self.n_frames_cnn, self.n_freq_cnn, 1))(x)
        y = Reshape((128, 256, 1))(x)
        print('y encoder mask', y.shape)
        
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation='linear', name='conv1', dilation_rate=self.dilation_rate)(y)
        y = ReLU(name='relu1')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)

        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation='linear', name='conv2', dilation_rate=self.dilation_rate)(y) 
        y = ReLU(name='relu2')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)

        y = Conv2D(self.n_prototypes, self.filter_size_cnn, padding='same', activation='linear', name='conv3', activity_regularizer=l1(0.000001))(y)
        y = ReLU(name='relu3')(y)
        ###################################    
        alpha = y
        #alpha = self.model_encoder_mask(x)
        h_hat, distance = self.model.get_layer('lc')([h, alpha])
        model = Model(x, [h_hat, distance])
        return model


    def train(self, data_train, data_val, weights_path='./',
              optimizer='Adam', learning_rate=0.001, early_stopping=100,
              considered_improvement=0.01,
              loss_weights=[10,5,5,5], sequence_time_sec=0.5,
              metric_resolution_sec=1.0, label_list=[],
              shuffle=True, init_last_layer=False, loss_classification='categorical_crossentropy',
              **kwargs_keras_fit):

        """
        Specific training function for APNet model
        """
        n_classes = len(label_list)
        # define optimizer and compile
        losses = [
            loss_classification,
            'mean_squared_error',
            prototype_loss,
            dummy_loss
        ]

        super().train(
            data_train, data_val,
            weights_path=weights_path, optimizer=optimizer,
            learning_rate=learning_rate, early_stopping=early_stopping,
            considered_improvement=considered_improvement,
            losses=losses, loss_weights=loss_weights,
            sequence_time_sec=sequence_time_sec,
            metric_resolution_sec=metric_resolution_sec,
            label_list=label_list, shuffle=shuffle,
            **kwargs_keras_fit
        )

    def create_encoder(self, activation = 'linear', name='encoder', use_batch_norm=False):
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input')
        print('encoder x', x.shape)
        #y = Reshape((self.n_frames_cnn, self.n_freq_cnn, 1))(x)
        y = Reshape((128, 256, 1))(x)
        #y = Lambda(lambda x: K.expand_dims(x,-1), name='expand_dims')(x)
        print('encoder y', y.shape)
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='conv1', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = ReLU(name='leaky_relu1')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling1')(y)
        bool_mask1 = Lambda(lambda t: K.greater_equal(t[0], t[1]), name='bool_mask1')([orig, y_up])
        #print('bool_mask1', bool_mask1)
        print('bool_mask1 shape', bool_mask1.shape)
        mask1 = Lambda(lambda t: K.cast(t, dtype='float32'), name='mask1')(bool_mask1)
        #print('mask1', mask1)
        print('mask1 shape', mask1.shape)
        
        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='conv2', dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = ReLU(name='leaky_relu2')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling2')(y)
        bool_mask2 = Lambda(lambda t: K.greater_equal(t[0], t[1]), name='bool_mask2')([orig, y_up])
        mask2 = Lambda(lambda t: K.cast(t, dtype='float32'), name='mask2')(bool_mask2)    
        
        y = Conv2D(self.N_filters[2], self.filter_size_cnn, padding='same', activation=activation, name='conv3')(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = ReLU(name='leaky_relu3')(y)

        model = Model(inputs=x, outputs=[y, mask1, mask2], name=name)
        return model

    def create_decoder(self, input_shape, mask1_shape, mask2_shape,
                       activation = 'linear', final_activation='tanh',
                       name='decoder', use_batch_norm=False, N_filters_out=1):
    
        x = Input(shape=input_shape, dtype='float32', name='input')
        mask1 = Input(shape=mask1_shape, dtype='float32', name= 'input_mask1')
        mask2 = Input(shape=mask2_shape, dtype='float32', name= 'input_mask2')

        deconv = Conv2DTranspose(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='deconv1')(x) 
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling2_1')(deconv)
        deconv = Multiply(name='multiply2')([mask2, deconv]) 
        deconv = ReLU(name='leaky_relu4')(deconv)

        deconv = Conv2DTranspose(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='deconv2')(deconv)     
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling3_1')(deconv)
        deconv = Multiply(name='multiply3')([mask1, deconv])    
        deconv = ReLU(name='leaky_relu5')(deconv)

        deconv = Conv2DTranspose(N_filters_out, self.filter_size_cnn, padding='same', activation='linear', name='deconv3')(deconv)
        print('1:deconv', deconv.shape)
        

        
        if N_filters_out == 1:
            #deconv = Lambda(lambda x: K.squeeze(x, axis=-1), name='input_reconstructed')(deconv)
            #shape = (self.n_frames_cnn, self.n_freq_cnn)
            deconv = Reshape((self.n_frames_cnn, self.n_freq_cnn), name='input_reconstructed')(deconv)
            print('2:deconv', deconv.shape)
        if use_batch_norm:
            deconv = BatchNormalization()(deconv) 
        deconv = Activation(final_activation, name='final_activation')(deconv)

        model = Model(inputs=[x, mask1, mask2], outputs=deconv, name=name)        
        return model

    def create_decoder_mask(self, input_shape,activation = 'linear', final_activation='tanh',
                       name='decoder', use_batch_norm=False, N_filters_out=1):
    
        x = Input(shape=input_shape, dtype='float32', name='input')

        deconv = Conv2DTranspose(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='deconv1')(x) 
        deconv = ReLU(name='leaky_relu4')(deconv)

        deconv = Conv2DTranspose(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='deconv2')(deconv)      
        deconv = ReLU(name='leaky_relu5')(deconv)

        deconv = Conv2DTranspose(N_filters_out, self.filter_size_cnn, padding='same', activation='linear', name='deconv3', activity_regularizer=l1(0.000001))(deconv)
        print('3:deconv', deconv.shape)
        if N_filters_out == 1:
            #deconv = Lambda(lambda x: K.squeeze(x, axis=-1), name='input_reconstructed')(deconv)
            #deconv = tf.squeeze(deconv, [-1], name='input_reconstructed')
            #deconv = Reshape((-1, 1))(deconv)
            deconv = Reshape((self.n_frames_cnn, self.n_freq_cnn), name='input_reconstructed')(deconv)
            print('4:deconv', deconv.shape)
        if use_batch_norm:
            deconv = BatchNormalization()(deconv) 

        deconv = Activation(final_activation, name='final_activation')(deconv)

        model = Model(inputs=x, outputs=deconv, name=name)  