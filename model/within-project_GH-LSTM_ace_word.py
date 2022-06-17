from keras.layers import Input, LSTM, Dense, Masking, Dropout, Multiply,concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
import numpy as np
from keras import backend as K
from keras.backend import clear_session
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
import os
import pandas as pd
def pre_process(path,train,word_length):
    data = pd.read_csv(path)
    commit_data = np.load('/data01/zwy/ACE/data/within-project/{}/{}_{}_commits.npy'.format(word_length,project, train))
    data = data[data['contains_bug'].isin([True, False])]
    data['fix'] = data['fix'].map(lambda x: 1 if x > 0 else 0)
    data['contains_bug'] = data['contains_bug'].map(lambda x: 1 if x > 0 else 0)
    data = data.iloc[::-1, :].reset_index()

    project_data = data[data["commit_hash"].isin(commit_data)]
    project_data = project_data.reset_index()
    project_data = project_data[
        ['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
    features = np.array(project_data.iloc[:, :])
    max_num = np.max(features,1).reshape(features.shape[0], 1)
    min_num = np.min(features,1).reshape(features.shape[0], 1)
    features = (features - min_num) / (max_num-min_num)
    features[np.isnan(features)] = 0
    return features.reshape((-1, 14, 1))

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# set GPU memory
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from tensorflow.compat.v1.keras.backend import set_session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess =tf.compat.v1.Session(config=config)


projects =["ambari", "ant", "aptoide", "camel", "cassandra", "egeria", "felix", "jackrabbit", "jenkins",
                     "lucene-solr"]

# LSTM classifier with ace features
if __name__=='__main__':
     for word_length in [50]:
        for project in projects:
            for i in range(10):
                file_names = os.listdir('../res/within-project/GH-LSTM_ace/{}'.format(word_length))
                if '{}_{}.npy'.format(project, i) in file_names:
                    print('{}_{}.npy exist'.format(project, i))
                    continue
                task_name = '{}_{}_{}'.format(project, 'LSTM',i)
                project_path = '../data/traditional_data/{}.csv'.format(project)
                train_Y = np.load('/data01/zwy/ACE/data/within-project/{}/{}_train_Y.npy'.format(word_length,project)).astype(np.float64)

                test_Y = np.load('/data01/zwy/ACE/data/within-project/{}/{}_test_Y.npy'.format(word_length,project)).astype(np.float64)


                weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=[0, 1],
                                                             y=train_Y.tolist())))
                clear_session()
                train_X = np.load('/data01/zwy/ACE/data/within-project/{}/{}_train_X.npy'.format(word_length,project)).astype(np.float64)
                test_X = np.load('/data01/zwy/ACE/data/within-project/{}/{}_test_X.npy'.format(word_length,project)).astype(np.float64)
                train_trd_X = pre_process(project_path,"train",word_length)
                test_trd_X = pre_process(project_path, "test",word_length)

                ace_input = Input(shape=(150, word_length), name='ace_input')
                ace_mask = Masking()(ace_input)
                ace_lstm_out = LSTM(128, name='ace_lstm')(ace_mask)
                middle_layer = Dense(64, activation='sigmoid', name='middle_output')(ace_lstm_out)
                ace_gate = Dense(128, activation='sigmoid', name='sce_gate')(ace_lstm_out)
                ace_gated_res = Multiply(name='sce_gated_res')([ace_gate, ace_lstm_out])
                traditional_input = Input(shape=(14, 1), name='traditional_input')
                traditional_lstm_out = LSTM(128, name='traditional_lstm')(traditional_input)
                traditional_gate = Dense(128, activation='sigmoid', name='traditional_gate')(traditional_lstm_out)
                traditional_gated_res = Multiply(name='traditional_gated_res')([traditional_gate, traditional_lstm_out])
                merge = concatenate([ace_gated_res, traditional_gated_res])

                main_output = Dense(1, activation='sigmoid', name='main_output')(merge)
                model = Model(inputs=[ace_input,traditional_input], outputs=[main_output])

                model.compile(loss=f1_loss, optimizer='adam', metrics=['accuracy', f1])
                val_data = ({'ace_input': test_X,"traditional_input": test_trd_X}, {'main_output': test_Y})
                print(len(test_Y))
                model.fit(x={'ace_input': train_X, 'traditional_input': train_trd_X},
                          y=train_Y,
                          batch_size=8192,
                          epochs=50,
                          class_weight=weight,
                          validation_data=val_data)
                predict_y = model.predict(x={'ace_input': test_X, 'traditional_input': test_trd_X})
                np.save('../res/within-project/GH-LSTM_ace/{}/{}_{}.npy'.format(word_length,project,i), predict_y)
                predict_y=np.round(predict_y)

                with open('../res/within-project/GH-LSTM_ace/{}/res.txt'.format(word_length), 'a+', encoding='utf-8') as f:
                    f.write('{} {} {} {} {}\n'.format(
                        task_name,
                        precision_score(y_true=test_Y, y_pred=predict_y),
                        recall_score(y_true=test_Y, y_pred=predict_y),
                        f1_score(y_true=test_Y, y_pred=predict_y),
                        accuracy_score(y_true=test_Y, y_pred=predict_y)
                    ))