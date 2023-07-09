#MAX_EPOCHS = 20
import tensorflow as tf
import keras.models
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Input, Dropout
from tensorflow.keras.layers import LSTM
#####################################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN

# Definir la arquitectura de la MLP
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(6,)))  # Capa densa con 64 unidades y función de activación ReLU
# model.add(Dense(32, activation='relu'))  # Capa densa con 32 unidades y función de activación ReLU
# model.add(Dense(1))  # Capa de salida con una unidad lineal

# # Compilar el modelo
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Entrenar el modelo
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# # Evaluar el modelo
# loss = model.evaluate(X_test, y_test)

# # Realizar predicciones
# predictions = model.predict(X_test)

##################

def compile_and_fit(model,trainX,trainY,testX,testY,BATCH_SIZE, MAX_EPOCHS,LR):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=10,mode='min')
    model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.optimizers.Adam(learning_rate=LR),
                  metrics= [tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error'),tf.keras.metrics.MeanAbsolutePercentageError(name='MeanAbsolutePercentageError')]
                 ) #tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred) | tf.keras.metrics.MeanAbsolutePercentageError()
    history = model.fit( x=trainX,   y=trainY,
    validation_data=(testX, testY),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE )
                    
    return history

def lstm():
    inputs  = Input(shape=(6,1)) # 6,1
    x = LSTM(64,activation="relu",return_sequences=True)(inputs)
    x = Dropout(0.7)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.7)(x)
    x = Dense(16, activation="relu")(x)
    #lstm_model = Model(inputs, x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs, x)
    return model


def red_neuronal1():
    inputs  = Input(shape=(6,1)) # 6,1
    x = LSTM(64,activation="relu",return_sequences=True)(inputs)
    x = Dropout(0.7)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.7)(x)
    x = Dense(16, activation="relu")(x)
    lstm_model = Model(inputs, x)
    return lstm_model

def red_neuronal2(combinedInput,a, b):
    z = Dense(64, activation="relu")(combinedInput)
    z = Dense(1, activation="linear")(z)
    model = Model(inputs=[a, b], outputs=z)
    return model

def red_neuronal3(combinedInput,a, b,c):
    z = Dense(64, activation="relu")(combinedInput)
    z = Dense(1, activation="linear")(z)
    model = Model(inputs=[a, b,c], outputs=z)
    return model

def mlp3():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(6,)))  # Capa densa con 64 unidades y función de activación ReLU
    model.add(Dense(32, activation='relu'))  # Capa densa con 32 unidades y función de activación ReLU
    model.add(Dense(16, activation='relu'))  # Capa densa con 32 unidades y función de activación ReLU
    model.add(Dense(1))  # Capa de salida con una unidad lineal
    return model

def rnn():
    model3 = Sequential()
    model3.add(SimpleRNN(64, activation='relu', input_shape=(6, 1)))  # Capa RNN con 64 unidades y función de activación ReLU
    model3.add(Dropout(0.25))  # Dropout con tasa de retención de 0.25
    model3.add(Dense(32, activation='relu'))  # Capa densa con 32 unidades y función de activación ReLU
    model3.add(Dropout(0.25))  # Dropout con tasa de retención de 0.25
    model3.add(Dense(1))  # Capa de salida
    return model3

def mlp2():
    model1 = Sequential()
    model1.add(Dense(64, activation='relu', input_shape=(6,)))  # Capa densa con 64 unidades y función de activación ReLU
    model1.add(Dense(32, activation='relu'))  # Capa densa con 32 unidades y función de activación ReLU
    model1.add(Dense(1))  # Capa de salida con una unidad lineal
    return model1