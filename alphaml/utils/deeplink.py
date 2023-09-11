# From DeepLINK
from __future__ import absolute_import, division, print_function, unicode_literals

# construct knockoffs using autoencoder #

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from .utils import min_max

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# define pairwise connected layers


class PairwiseConnected(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PairwiseConnected, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0
        self.feat_dim = input_shape[-1] // 2
        self.w = self.add_weight(name="weight", shape=(input_shape[-1],),
                                 initializer="uniform", trainable=True)
        super(PairwiseConnected, self).build(input_shape)

    def call(self, x, **kwargs):
        elm_mul = x * self.w
        output = elm_mul[:, 0:self.feat_dim] + elm_mul[:, self.feat_dim:]

        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.feat_dim
        return tuple(output_shape)


class DeepLink(tf.keras.Model):
    
    def __init__(self,
                 data,
                 dataY,
                 latent_dim,
                 min_sel_features,
                 fdr,
                 result_path):

        """
        Feature selection using ModelX knockoffs.
    
        Args:
            data: data matrix.
            dataY: Labels for data.
            latent_dim: latent dimensions for autoencoder.
            min_sel_features: expected minimum features.
            fdr: fdr value
            result_path: Path to save results
    
        Returns:
            Selected data.
        """
        super(DeepLink, self).__init__()

        date_ = datetime.now().strftime("%Y%m%d_%I%M%S%p")


        dataX = min_max(data.T).T
        dataX.to_csv(os.path.join(result_path, "ModelX"+date_+"_dataX.csv"), sep=',')

        data_training, data_validation = train_test_split(dataX, test_size=0.25)
        train_data = np.array(data_training)
        val_data = np.array(data_validation)
        fdr = fdr
        epochs = int(1000)
        batch_size = int(32)
        learning_rate = 0.005
        original_dim = data_training.shape[1]
        X = np.array(dataX)
        y = np.array(dataY)

        print("knockoffs construction starts!\n")
        ES = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=0)  # mode ??
        MC = tf.keras.callbacks.ModelCheckpoint(os.path.join(result_path, date_+'_AE_model.h5'),
                                                monitor='val_loss', patience=50, verbose=0, save_best_only=True)
        
        autoencoder = tf.keras.models.Sequential()
        autoencoder.add(tf.keras.layers.Dense(latent_dim, activation="relu", use_bias=False, input_shape=(original_dim,)))
        autoencoder.add(tf.keras.layers.Dense(original_dim, activation="relu", use_bias=False))
        autoencoder.compile(loss="mean_squared_error",
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        history = autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size,
                                  verbose=0, validation_data=(val_data, val_data), callbacks=[ES, MC])
        
        # Save plots
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('AE Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig = plt.gcf()
        fig.set_size_inches(14.5, 8.5)
        hist_plot_fileL = os.path.join(result_path, date_+"_AE_loss.pdf")
        fig.savefig(hist_plot_fileL)
        plt.close(fig)
        
        n = X.shape[0]
        p = X.shape[1]
        C = autoencoder.predict(X)
        E = X - C
        sigma = np.sqrt(np.sum(E ** 2) / (n * p))
        X_ko = C + sigma * np.random.randn(n, p)
        Xnew = np.hstack((X, X_ko))
        data_data_ko_df = pd.DataFrame(Xnew)
        data_data_ko_df.to_csv(os.path.join(result_path, date_+"_AE_data_data_ko.csv"), sep=',')
        print("knockoffs construction done!\n")
        
        print("knockoff statistics computation starts!\n")
        MC1 = tf.keras.callbacks.ModelCheckpoint(os.path.join(result_path, date_+'_DPNK_model.h5'),
                                                 monitor='val_loss', patience=50, verbose=0, save_best_only=True)
        coeff = 0.05 * np.sqrt(2.0 * np.log(p) / n)

        dp = tf.keras.models.Sequential()
        dp.add(PairwiseConnected(input_shape=(2 * p,)))
        dp.add(tf.keras.layers.Dense(p, activation='elu', kernel_regularizer=tf.keras.regularizers.l1(coeff)))
        dp.add(tf.keras.layers.Dense(1, activation='elu', kernel_regularizer=tf.keras.regularizers.l1(coeff)))
        dp.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        history1 = dp.fit(Xnew, y, epochs=epochs, batch_size=batch_size, verbose=0,
                          validation_split=0.2, callbacks=[ES, MC1])

        # Save plots
        plt.plot(history1.history['loss'])
        plt.plot(history1.history['val_loss'])
        plt.title('DPNK Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig = plt.gcf()
        fig.set_size_inches(14.5, 8.5)
        hist_plot_fileL = os.path.join(result_path, date_+"_DPNK_loss.pdf")
        fig.savefig(hist_plot_fileL)
        plt.close(fig)

        weights = dp.get_weights()
        w = weights[1] @ weights[3] #np.matmul https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        # w = np.matmul(weights[1], weights[3])
        w = w.reshape(p, )
        z = weights[0][:p]
        z_tilde = weights[0][p:]
        W = (w * z) ** 2 - (w * z_tilde) ** 2

        q = fdr

        print("variable selection starts!\n")
        t = np.sort(np.concatenate(([0], abs(W))))

        ratio = [sum(W <= -tt) / max(1, sum(W >= tt)) for tt in t[:p]]
        ind = np.where(np.array(ratio) <= q)[0]
        if len(ind) == 0:
            T = float("inf")
        else:
            T = t[ind[0]]
        selected = np.where(W >= T)[0]

        all_features = dataX.columns.tolist()
        if len(selected) > 0:
            features = selected.tolist()
            selected_feature = [all_features[i] for i in features]
            df1 = pd.DataFrame(selected_feature, columns=['Selected features'])
            df1.to_csv(f"{result_path}{date_}_selected_variable_ko.csv", index=False)
        else:
            selected_feature = []

        ratio_plus = [(1 + sum(W <= -tt)) / max(1, sum(W >= tt)) for tt in t[:p]]
        ind_plus = np.where(np.array(ratio_plus) <= q)[0]
        if len(ind_plus) == 0:
            T_plus = float("inf")
        else:
            T_plus = t[ind_plus[0]]
        selected_plus = np.where(W >= T_plus)[0]

        if len(selected_plus) > 0:
            features_plus = selected_plus.tolist()
            selected_feature_plus = [all_features[i] for i in features_plus]
            df2 = pd.DataFrame(selected_feature_plus, columns=['Selected features plus'])
            df2.to_csv(f"{result_path}{date_}_selected_variable_ko_plus.csv", index=False)
        else:
            selected_feature_plus = []
        print("variable selection done!\n")

        if len(selected_feature) > len(selected_feature_plus):
            out_feature = selected_feature
        else:
            out_feature = selected_feature_plus

        if len(out_feature) >= min_sel_features:
            self.out_feature = out_feature
        else:
            self.out_feature = []


    def get_features(self):
        return self.out_feature


# To run model ##

def modelx(data, dataY, latent_dim, min_sel_features, fdr, result_path):
    features = DeepLink(data, dataY, latent_dim, min_sel_features, fdr, result_path).get_features()
    return features


"""
from .deeplINK.main import DeepLink
features = DeepLink(data, dataY, data_for_fs, result_path)
selected_features = features.get_features()
"""
