# capsule part refer to https://github.com/bojone/Capsule

import seaborn as sns
import matplotlib.pyplot as plt
from Visualization_Capsule_Keras import *
from keras.models import Model
from keras.layers import *
from keras import backend as K
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

#################################################################################################################################
parser = argparse.ArgumentParser(description='MultiCapsNet')

parser.add_argument('--inputdata', type=str, default='data/1_variant_call_data.npy', help='address for input data')
parser.add_argument('--inputcelltype', type=str, default='data/1_variant_call_type.npy', help='address for celltype label')
parser.add_argument('--source_division', type=str, default='data/1_variant_call_data_length.npy', help='data source length')
parser.add_argument('--num_classes', type=int, default=3, help='number of class need to specify')
parser.add_argument('--randoms', type=int, default=30, help='random number to split dataset')
parser.add_argument('--dim_capsule', type=int, default=4, help='dimension of the capsule')
parser.add_argument('--activation_function', type=str, default='relu', help='activation function for primary capsule')
parser.add_argument('--training_weights', type=str, default='weights/1_variant_call.weights', help='training_weights')


args = parser.parse_args()

inputdata = args.inputdata
inputcelltype = args.inputcelltype
num_classes = args.num_classes
randoms = args.randoms
z_dim = args.dim_capsule
source_division = args.source_division
activation_function = args.activation_function
training_weights = args.training_weights

#####################################################################################################################
#training data and test data
data = np.load(inputdata)
labels = np.load(inputcelltype)
regulon_length = np.load(source_division)
num_primary_capsule = regulon_length.shape[0]

print(type(data))
print(data.shape)

data = np.transpose(data)

print(data.shape)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state= randoms)

#divide dataset
total = regulon_length[0]
X_train = [x_train[:,0:total]]
X_test = [x_test[:,0:total]]
for length in regulon_length[1:]:
    X_train.append(x_train[:,total:total+length])
    X_test.append(x_test[:,total:total+length])
    total = total + length

###########################################################################################################################
# model
num_key = regulon_length.shape[0]
print(num_key)
allinputs = [Input(shape=(regulon_length[0],))]
for length in regulon_length[1:]:
    allinputs.append(Input(shape=(length,)))

x_added = Dense(z_dim, activation=activation_function)(allinputs[0])
for i in range(1,num_key):
    x_added = Concatenate()([x_added, Dense(z_dim, activation=activation_function)(allinputs[i])])

x = Reshape((num_key, z_dim))(x_added)
capsule = Capsule(num_classes, z_dim, 3, False)(x)
print(capsule.shape)

output = capsule

model = Model(inputs=allinputs, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

#Loading model weights
model.load_weights(training_weights)

###########################################################################################################################
#heatmap and high rank data source
predict = model.predict(X_train)
primary_capsule_num = regulon_length.shape[0]

Ycategory = []
for i in range(len(y_train)):
    category = y_train[i]
    for j in range(len(category)):
        if (category[j] == 1):
            Ycategory.append(j)
            continue

value = {}
count = {}

for i in range(len(predict)):
    ind = int(Ycategory[i])
    if ind in value.keys():
        value[ind] = value[ind] + predict[i]
        count[ind] = count[ind] + 1
    if ind not in value.keys():
        value[ind] = predict[i]
        count[ind] = 1

total = np.zeros((num_classes,primary_capsule_num))
sns.cubehelix_palette(as_cmap=True, reverse=True)
cmap = sns.cm.rocket_r

if num_primary_capsule<10:
    output_high_rank_num = num_primary_capsule
else:
    output_high_rank_num = 10

plt.figure(figsize=(20,3.5*np.ceil(num_classes/3)))
for i in range(num_classes):
    res = value[i]/count[i]
    if(i==0):
        all_coupling_coef = res
    else:
        all_coupling_coef = np.vstack((all_coupling_coef,res))

    Lindex = i + 1
    plt.subplot(np.ceil(num_classes/3),3,Lindex)
    total[i] = res[i]
    maximum_index = np.argsort(res[i])[num_primary_capsule-output_high_rank_num:num_primary_capsule]

    if Lindex==1:
        top_rank_primary_capsule = maximum_index
    else:
        top_rank_primary_capsule = np.vstack((top_rank_primary_capsule, maximum_index))

    colnum = []
    for i in range(primary_capsule_num):
        colnum.append(i+1)

    df = DataFrame(np.asmatrix(res), columns=colnum)
    if Lindex == 1 or Lindex == 4 or Lindex == 7:
        heatmap = sns.heatmap(df, cmap=cmap)
    else:
        heatmap = sns.heatmap(df, yticklabels=[],cmap=cmap)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=16)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=15)

np.savetxt('output/top_rank_primary_capsule.txt',top_rank_primary_capsule,fmt='%d')

plt.savefig("output/heatmaps.png")
#plt.show()

plt.figure(figsize=(9,6))
df = DataFrame(np.asmatrix(total),columns=colnum)
heatmap = sns.heatmap(df,cmap=cmap)
plt.xticks(fontsize=13)
plt.yticks(fontsize=16)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=15)
plt.savefig("output/overall_heatmaps.png")
#plt.show()