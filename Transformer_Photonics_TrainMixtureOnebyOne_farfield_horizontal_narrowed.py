# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:20:26 2024

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:53:03 2023

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:25:53 2023

@author: limyu
"""



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
import time

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
        url = "https://raw.githubusercontent.com/yd145763/Beam_ML_mixed_pitch/main/mixedpitch"+str(r)+".csv"
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

import os
# Specify the path for the new folder
folder_path = 'C:\\Users\\ADMIN\Downloads\\transformer_codes\\onebyone_farfield_horizontal_narrowed'  # Replace with the desired path

# Check if the folder already exists or not
if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")

label = 'Horizontal'

df_main = get_df(R, I, label)


backcandlesS = 5,10,20
num_transformer_blocksS=2,3,4

num_heads = 2
ff_dim = 2
head_size = 16

train_test_ape = []
head_size_list = []
num_head_list = []
ff_dim_list = []
num_transformer_blocks_list = []
ape_label = []
time_list = []
sequence_length_list = []
rms_list = []
datalength_list = []


for num_transformer_blocks in num_transformer_blocksS:
    for backcandles in backcandlesS:

        from keras.models import Sequential
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import Dense
        
        
        
        from keras import optimizers
        from keras.callbacks import History
        from keras.models import Model
        from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
        import numpy as np
        
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
        
        #validation dataset
        data_full1 = df_main[df_main["mixture"] == test_radius]
        data_full1 = data_full1.sort_values(by='z', axis=0)
        data_full1 = data_full1.iloc[int(len(data_full1['z'])*0.6):,:]
        
        data1 = data_full1[['z',label, 'RSI', 'EMAF', 'mixture', 'EMAM', 'EMAS', 'TargetNextClose']]

        data1.dropna(inplace = True)
        data1.reset_index(inplace = True)
        data1.drop(['index'], axis=1, inplace = True)
        data_set1 = data1.iloc[:, 0:11]
        pd.set_option('display.max_columns', None)
        print(data_set1.head(5))
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0,1))
        data_set_scaled1 = sc.fit_transform(data_set1)
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
        
        input_shape = X_test1.shape[1:]
        
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
        model.summary()
        for r in R[R !=R_actual]:
            data_full_filtered = data_full_original[data_full_original["mixture"] ==r]
            data_full_filtered_sorted = data_full_filtered.sort_values(by='z', axis=0)
            data_full_filtered_sorted_shortened = data_full_filtered_sorted.iloc[:int(len(data_full_filtered_sorted['z'])*0.6),:]

            data = data_full_filtered_sorted_shortened[['z',label, 'RSI', 'EMAF', 'mixture', 'EMAM', 'EMAS', 'TargetNextClose']]

            data.dropna(inplace = True)
            data.reset_index(inplace = True)
            data.drop(['index'], axis=1, inplace = True)
            data_set = data
            pd.set_option('display.max_columns', None)
            print(data_set.head(5))
            
            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler(feature_range=(0,1))
            data_set_scaled = sc.fit_transform(data_set)
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
            

        
        
        #callbacks = [keras.callbacks.EarlyStopping(patience=10, \
         #   restore_best_weights=True)]
        

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
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines['bottom'].set_linewidth(5)
            ax.spines['left'].set_linewidth(5)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(["Training loss", "Validation Loss"], prop={'weight': 'bold','size': 35}, loc = "best")
            plt.title("head_size is "+str(head_size)+"sequence length is "+str(backcandles)+"\n"+"num_heads is "+str(num_heads)+"\n"+"ff_dim is "+str(ff_dim)+"\n"+"num_transformer_blocks is "+str(num_transformer_blocks)+"\n"+"mixture ratio is "+str(r)+"1.1um "+str(100-r)+"1.2um ", fontweight = 'bold')
            plt.savefig(folder_path+"\\TV_"+"seq"+str(backcandles)+"_headsize"+str(head_size)+"_numheads"+str(num_heads)+"_ffdim"+str(ff_dim)+"_blocks"+str(num_transformer_blocks)+"_mix"+str(r)+"11um_"+str(100-r)+"12um"+".jpg", format='jpg')
            plt.show()
            plt.close()
        
        
        
        
        #prediction dataset
        data_full2 = df_main[df_main["mixture"] == test_radius]
        data_full2 = data_full2.sort_values(by='z', axis=0)
        data_full2 = data_full2.iloc[int(len(data_full1['z'])*0.6):,:]
        
        data2 = data_full2[['z',label, 'RSI', 'EMAF', 'mixture', 'EMAM', 'EMAS', 'TargetNextClose']]

        data2.dropna(inplace = True)
        data2.reset_index(inplace = True)
        data2.drop(['index'], axis=1, inplace = True)
        data_set2 = data2.iloc[:, 0:11]
        pd.set_option('display.max_columns', None)
        print(data_set2.head(5))
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0,1))
        data_set_scaled2 = sc.fit_transform(data_set2)
        print(data_set_scaled2)
        
        X2 = []
        
        for j in range(data_set_scaled2.shape[1]-1):
            X2.append([])
            for i in range(backcandles, data_set_scaled2.shape[0]):
                X2[j].append(data_set_scaled2[i-backcandles:i, j])
                print(data_set_scaled2[i-backcandles:i, j])
                print(" ")
        X2 = np.moveaxis(X2, [0], [2])
        X_test2 = np.array(X2)
        yi2 = np.array(data_set_scaled2[backcandles:,-1])
        y_test2 = np.reshape(yi2, (len(yi2), 1))
        
        
        y_pred = model.predict(X_test1)
        #y_pred=np.where(y_pred > 0.43, 1,0)

        
        
        

        
        nextclose = np.array(data['TargetNextClose'])
        nextclose = nextclose.reshape(-1, 1)
        
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(nextclose)
        denormalized_data = scaler.inverse_transform(normalized_data)
        y_pred_ori = scaler.inverse_transform(y_pred)
        y_test_ori = scaler.inverse_transform(y_test1)
        
        z_plot1 = data1['z'][:len(y_pred)]
        z_plot = [i*1000000 for i in z_plot1]
        
        #z_plot1 = np.linspace(-5,45, len(data2))
        #z_plot = z_plot1[backcandles:]
        
        y_test_ori = y_test_ori[25-backcandles:]
        y_pred_ori = y_pred_ori[25-backcandles:]
        z_plot = z_plot[25-backcandles:]
        
        diff = (pd.Series(y_test_ori.flatten()) - pd.Series(y_pred_ori.flatten())).abs()
        rel_error = diff / pd.Series(y_test_ori.flatten())
        pct_error = rel_error * 100
        ape = pct_error.mean()
        ape_label.append(ape)
        
        from sklearn.metrics import mean_squared_error
        rms = np.sqrt(mean_squared_error(y_test_ori, y_pred_ori))
        rms_list.append(rms)
        
        fig = plt.figure(figsize=(20, 13))
        ax = plt.axes()
        ax.scatter(z_plot,[i*1000000 for i in y_test_ori], s=50, facecolor='blue', edgecolor='blue')
        ax.plot(z_plot,[i*1000000 for i in y_pred_ori], color = "red", linewidth = 5)
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
        plt.xlabel("z (µm)")
        plt.ylabel(label+" Beam Waist (µm)")
        plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 35}, loc = "best")
        plt.title("Label\n"+"sequence length is "+str(backcandles)+"head_size is "+str(head_size)+"\n"+"num_heads is "+str(num_heads)+"\n"+"ff_dim is "+str(ff_dim)+"\n"+"num_transformer_blocks is "+str(num_transformer_blocks)+"\n"+"\n", fontweight = 'bold')
        plt.savefig(folder_path+"\\PA_"+"seq"+str(backcandles)+"_headsize"+str(head_size)+"_numheads"+str(num_heads)+"_ffdim"+str(ff_dim)+"_blocks"+str(num_transformer_blocks)+"_mix"+str(r)+"11um_"+str(100-r)+"12um"+".jpg", format='jpg')
        plt.show()
        plt.close()
        
        plt.figure(figsize=(16,8))
        plt.plot(z_plot,y_test_ori, color = 'black', label = 'Test')
        plt.plot(z_plot,y_pred_ori, color = 'green', label = 'pred')
        plt.title("head_size is "+str(head_size)+"\n"+"num_heads is "+str(num_heads)+"\n"+"ff_dim is "+str(ff_dim)+"\n"+"num_transformer_blocks is "+str(num_transformer_blocks)+"\n")
        plt.legend()
        plt.show()
        plt.close()
        
        end = time.time()
        time_list.append(end-start)
        datalength_list.append(len(z_plot))
        
        for i in range(20):
            print(y_pred_ori[i], y_test_ori[i])

# Calculate the differences between corresponding elements
differences = [abs(x - y) for x, y in zip(y_pred_ori, y_test_ori)]

# Calculate the average difference
average_difference = sum(differences) / len(differences)

max(y_test_ori) - min(y_test_ori)

df_result = pd.DataFrame()

df_result['head_size_list'] = head_size_list
df_result['num_head_list'] = num_head_list
df_result['ff_dim_list'] = ff_dim_list
df_result['num_transformer_blocks_list'] = num_transformer_blocks_list
df_result['ape_label'] = ape_label
df_result['time_list'] = time_list
df_result['train_test_ape r = 0'] = train_test_ape[0::6]
df_result['train_test_ape r = 20'] = train_test_ape[1::6]
df_result['train_test_ape r = 40'] = train_test_ape[2::6]
df_result['train_test_ape r = 50'] = train_test_ape[3::6]
df_result['train_test_ape r = 60'] = train_test_ape[4::6]
df_result['train_test_ape r = 80'] = train_test_ape[5::6]
df_result['sequence_length_list'] = sequence_length_list
df_result['rms_list'] = rms_list
df_result['datalength_list'] = datalength_list

df_result.to_csv(folder_path+"\\full_result.csv", index=True) 

import seaborn as sns

mat = df_result.pivot('sequence_length_list', 'num_transformer_blocks_list', 'ape_label')
mat_list = mat.values.tolist()

plt.figure(figsize=(6, 4))
ax = sns.heatmap(mat_list, annot=True, cmap='rainbow', fmt=".1f")

mat.columns
ax.set_xticklabels(mat.columns, fontweight="bold", fontsize = 12)
ax.set_yticklabels(mat.index, fontweight="bold", fontsize = 12)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Average\nPercentage\nError (%)", fontweight="bold")
font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Number of Layers, L", fontdict=font)
ax.set_xlabel("Number of Nodes, N\n(First Layer)", fontdict=font)
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
plt.show()
plt.close()

df_result['rms_list'] = [i*1e6 for i in df_result['rms_list']]

mat = df_result.pivot('sequence_length_list', 'num_transformer_blocks_list', 'rms_list')
mat_list = mat.values.tolist()

plt.figure(figsize=(6, 4))
ax = sns.heatmap(mat_list, annot=True, cmap='rainbow', fmt=".2f")

mat.columns
ax.set_xticklabels(mat.columns, fontweight="bold", fontsize = 12)
ax.set_yticklabels(mat.index, fontweight="bold", fontsize = 12)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Average\nPercentage\nError (%)", fontweight="bold")
font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Number of Layers, L", fontdict=font)
ax.set_xlabel("Number of Nodes, N\n(First Layer)", fontdict=font)
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
plt.show()
plt.close()