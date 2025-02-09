# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:46:42 2024

@author: ADMIN
"""



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter






import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import mean_absolute_error
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from matplotlib.ticker import StrMethodFormatter
import pandas_ta as ta

R = pd.Series([0,20,40,50,60,80])
I = pd.Series(np.arange(0, len(R), 1))
R_actual = 50 #set the ratio to be "removed" here
I_actual = R.index[R == R_actual][0]
R_training = R[R!=R_actual]
I_training = I[I!=I_actual]
R_actual_norm = round((R_actual - min(R))/(max(R)-min(R)),5)

def get_df(R, I, label):
    df_name = pd.DataFrame()
    for r, i in zip(R, I):
        url = "https://raw.githubusercontent.com/yd145763/Mixed_pitch_ML_full_data/main/"+str(r)+"mixedpitch.csv"
        df = pd.read_csv(url)

        df=df.assign(mixture=r)
        df.rename(columns={'horizontal_full': 'Horizontal'}, inplace=True)
        df.rename(columns={'verticle_full': 'Vertical'}, inplace=True)
        df['RSI'] = ta.rsi(df[label], length = 10)
        df['EMAF'] = ta.ema(df[label], length = 20)
        df['EMAM'] = ta.ema(df[label], length = 30)
        df['EMAS'] = ta.ema(df[label], length = 40)
        df['TargetNextClose'] = df[label].shift(-1)
        df_name = pd.concat([df_name, df], axis= 0)
    return df_name
"""
import os
# Specify the path for the new folder
folder_path = 'C:\\Users\\ADMIN\Downloads\\transformer_codes\\oneshot_farfield_horizontal'  # Replace with the desired path

# Check if the folder already exists or not
if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")
"""




backcandlesS = 5,10,20

head_sizeS=16,32,64
num_headsS=2,3,4
ff_dimS=2,3,4
num_transformer_blocksS=2,3,4

train_test_ape = []
head_size_list = []
num_head_list = []
ff_dim_list = []
num_transformer_blocks_list = []
ape_label = []
time_list = []
sequence_length_list = []

backcandles = 5
head_size = 16
num_heads = 2
ff_dim = 2
num_transformer_blocks = 2
                    
label = 'e316'
df_main = get_df(R, I, label)

master_data = pd.DataFrame([])

start = time.time()
#add record transformer model parameters
head_size_list.append(head_size)
num_head_list.append(num_heads)
ff_dim_list.append(ff_dim)
num_transformer_blocks_list.append(num_transformer_blocks)
sequence_length_list.append(backcandles)
#set training data
test_radius = R_actual

data_full_original = df_main
data_full = pd.DataFrame()

for r in R:
    data_full_filtered = data_full_original[data_full_original["mixture"] ==r]
    data_full_filtered_sorted = data_full_filtered.sort_values(by='x', axis=0)
    data_full_filtered_sorted_shortened = data_full_filtered_sorted.iloc[:int(len(data_full_filtered_sorted['x'])*0.6),:]
    data_full = pd.concat([data_full, data_full_filtered_sorted_shortened], axis= 0)
    
data = data_full[['x',label, 'RSI', 'EMAF', 'mixture', 'EMAM', 'EMAS', 'TargetNextClose']]

data.dropna(inplace = True)
data.reset_index(inplace = True)
data.drop(['index'], axis=1, inplace = True)
data_set = data
pd.set_option('display.max_columns', None)
print(data_set.head(5))

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data_set)
data_set_scaled = np.insert(data_set_scaled, 4, 0, axis=1)
print(data_set_scaled)

X = []

for j in range(data_set_scaled.shape[1]-1):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i-backcandles:i, j])
        print(data_set_scaled[i-backcandles:i, j])
        print(" ")
X = np.moveaxis(X, [0], [2])
X_train = np.array(X)
yi = np.array(data_set_scaled[backcandles:,-1])
y_train = np.reshape(yi, (len(yi), 1))



#functions to define transformer model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

input_shape = X_train.shape[1:]

model = build_model(
    input_shape,
    head_size=head_size,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)

N = np.arange(0,316,1)

for n in N:
    label = 'e'+str(n)
    df_main = get_df(R, I, label)

    data_full_original = df_main
    data_full = pd.DataFrame()
    
    
    for r in R:
        data_full_filtered = data_full_original[data_full_original["mixture"] ==r]
        data_full_filtered_sorted = data_full_filtered.sort_values(by='x', axis=0)
        data_full_filtered_sorted_shortened = data_full_filtered_sorted.iloc[:int(len(data_full_filtered_sorted['x'])*0.6),:]
        data_full = pd.concat([data_full, data_full_filtered_sorted_shortened], axis= 0)
        
    data = data_full[['x',label, 'RSI', 'EMAF', 'mixture', 'EMAM', 'EMAS', 'TargetNextClose']]
    
    data.dropna(inplace = True)
    data.reset_index(inplace = True)
    data.drop(['index'], axis=1, inplace = True)
    data_set = data
    pd.set_option('display.max_columns', None)
    print(data_set.head(5))
    
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = sc.fit_transform(data_set)
    data_set_scaled = np.insert(data_set_scaled, 4, n/316, axis=1)
    print(data_set_scaled)
    
    X = []
    
    for j in range(data_set_scaled.shape[1]-1):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i-backcandles:i, j])
            print(data_set_scaled[i-backcandles:i, j])
            print(" ")
    X = np.moveaxis(X, [0], [2])
    X_train = np.array(X)
    yi = np.array(data_set_scaled[backcandles:,-1])
    y_train = np.reshape(yi, (len(yi), 1))
    
    #validation dataset
    data_full1 = df_main[df_main["mixture"] == test_radius]
    data_full1 = data_full1.sort_values(by='x', axis=0)
    data_full1 = data_full1.iloc[int(len(data_full1['x'])*0.6):,:]
    
    data1 = data_full1[['x',label, 'RSI', 'EMAF', 'mixture', 'EMAM', 'EMAS', 'TargetNextClose']]
    
    data1.dropna(inplace = True)
    data1.reset_index(inplace = True)
    data1.drop(['index'], axis=1, inplace = True)
    data_set1 = data1.iloc[:, 0:11]
    pd.set_option('display.max_columns', None)
    print(data_set1.head(5))
    
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled1 = sc.fit_transform(data_set1)
    data_set_scaled1 = np.insert(data_set_scaled1, 4, n/316, axis=1)
    print(data_set_scaled1)
    
    X1 = []
    
    for j in range(data_set_scaled1.shape[1]-1):
        X1.append([])
        for i in range(backcandles, data_set_scaled1.shape[0]):
            X1[j].append(data_set_scaled1[i-backcandles:i, j])
            print(data_set_scaled1[i-backcandles:i, j])
            print(" ")
    X1 = np.moveaxis(X1, [0], [2])
    X_test1 = np.array(X1)
    yi1 = np.array(data_set_scaled1[backcandles:,-1])
    y_test1 = np.reshape(yi1, (len(yi1), 1))
    
    splitlimit1 = int(len(X1)*0.8)
    
    
    
    
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test1, y_test1),
        epochs=100,
        batch_size=64,
        #callbacks=callbacks,
    )
    
    training_loss = pd.Series(history.history['loss'])
    validation_loss = pd.Series(history.history['val_loss'])
    
    diff = (validation_loss[50:] - training_loss[50:])
    ape = sum(diff)/len(diff)
    train_test_ape.append(ape)
    
    epochs = range(1, 100 + 1)
    
    fig = plt.figure(figsize=(20, 13))
    ax = plt.axes()
    ax.plot(epochs, training_loss, color = "blue", linewidth = 5)
    ax.plot(epochs, validation_loss, color = "red", linewidth = 5)
    #graph formatting     
    ax.tick_params(which='major', width=2.00)
    ax.tick_params(which='minor', width=2.00)
    ax.xaxis.label.set_fontsize(35)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(35)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    plt.xlabel("Epochs "+str(n))
    plt.ylabel("Loss")
    plt.legend(["Training loss", "Validation Loss"], prop={'weight': 'bold','size': 35}, loc = "best")
    plt.title("Training-Validation Loss\n"+"sequence length is "+str(backcandles)+"head_size is "+str(head_size)+"\n"+"num_heads is "+str(num_heads)+"\n"+"ff_dim is "+str(ff_dim)+"\n"+"num_transformer_blocks is "+str(num_transformer_blocks)+"\n"+"\n", fontweight = 'bold')
    plt.show()
    plt.close()
    
    
    
    



    y_pred = model.predict(X_test1)
    #y_pred=np.where(y_pred > 0.43, 1,0)
    
    
    
    nextclose = np.array(data['TargetNextClose'])
    nextclose = nextclose.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(nextclose)
    denormalized_data = scaler.inverse_transform(normalized_data)
    y_pred_ori = scaler.inverse_transform(y_pred)
    y_test_ori = scaler.inverse_transform(y_test1)
    
    z_plot1 = data1['x'][:len(y_pred)]
    z_plot = z_plot1
    
    
    diff = (pd.Series(y_test_ori.flatten()) - pd.Series(y_pred_ori.flatten())).abs()
    rel_error = diff / pd.Series(y_test_ori.flatten())
    pct_error = rel_error * 100
    ape = pct_error.mean()
    ape_label.append(ape)
    
    fig = plt.figure(figsize=(20, 13))
    ax = plt.axes()
    ax.scatter(z_plot1,[i*1000 for i in y_test_ori], s=50, facecolor='blue', edgecolor='blue')
    ax.plot(z_plot1,[i*1000 for i in y_pred_ori], color = "red", linewidth = 5)
    #graph formatting     
    ax.tick_params(which='major', width=5.00)
    ax.tick_params(which='minor', width=5.00)
    ax.xaxis.label.set_fontsize(35)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(35)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    plt.xlabel("x (µm)")
    plt.ylabel("E-field (meV)")
    plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 35}, loc = "best")
    plt.title("Label\n"+"sequence length is "+str(backcandles)+"head_size is "+str(head_size)+"\n"+"num_heads is "+str(num_heads)+"\n"+"ff_dim is "+str(ff_dim)+"\n"+"num_transformer_blocks is "+str(num_transformer_blocks)+"\n"+"\n", fontweight = 'bold')
    plt.show()
    plt.close()
    
    master_data['pred_e'+str(n)] = [i for i in y_test_ori]
    master_data['ori_e'+str(n)] = [i for i in y_pred_ori]

master_data['x'] = z_plot1

master_data.to_csv('/home/grouptan/Documents/yudian/data/master_data.csv')

calculated_data = pd.DataFrame([])
calculated_data['train_test_ape'] = train_test_ape
calculated_data['ape_label'] = ape_label

calculated_data.to_csv('/home/grouptan/Documents/yudian/data/calculated_data.csv')

