# capsule part refer to https://github.com/bojone/Capsule

from keras import optimizers
from Capsule_Keras import *
from keras import utils
from keras import callbacks
from keras.models import Model
from keras.layers import *
from keras import backend as K
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import argparse

#####################################################################################################################
parser = argparse.ArgumentParser(description='MultiCapsNet')

parser.add_argument('--inputdata', type=str, default='data/1_variant_call_data.npy', help='address for input data')
parser.add_argument('--inputcelltype', type=str, default='data/1_variant_call_type.npy', help='address for celltype label')
parser.add_argument('--source_division', type=str, default='data/1_variant_call_data_length.npy', help='data source length')
parser.add_argument('--num_classes', type=int, default=3, help='number of class need to specify')
parser.add_argument('--randoms', type=int, default=30, help='random number to split dataset')
parser.add_argument('--dim_capsule', type=int, default=4, help='dimension of the capsule')
parser.add_argument('--activation_function', type=str, default='relu', help='activation function for primary capsule')
parser.add_argument('--batch_size', type=int, default=200, help='training parameters_batch_size')
parser.add_argument('--epochs', type=int, default=200, help='training parameters_epochs')

args = parser.parse_args()

inputdata = args.inputdata
inputcelltype = args.inputcelltype
num_classes = args.num_classes
randoms = args.randoms
z_dim = args.dim_capsule
epochs = args.epochs
batch_size = args.batch_size
source_division = args.source_division
activation_function = args.activation_function


#####################################################################################################################
#training data and test data
data = np.load(inputdata)
labels = np.load(inputcelltype)
data_source_length = np.load(source_division)

print(type(data))
print(data.shape)

data = np.transpose(data)

print(data.shape)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state= randoms)

total = data_source_length[0]
X_train = [x_train[:,0:total]]
X_test = [x_test[:,0:total]]
for length in data_source_length[1:]:
    X_train.append(x_train[:,total:total+length])
    X_test.append(x_test[:,total:total+length])
    total = total + length

###########################################################################################################################
# model
z_dim =z_dim

num_key = data_source_length.shape[0]
print(num_key)
allinputs = [Input(shape=(data_source_length[0],))]
for length in data_source_length[1:]:
    allinputs.append(Input(shape=(length,)))

x_added = Dense(z_dim, activation=activation_function)(allinputs[0])
for i in range(1,num_key):
    x_added = Concatenate()([x_added, Dense(z_dim, activation=activation_function)(allinputs[i])])

x = Reshape((num_key, z_dim))(x_added)
capsule = Capsule(num_classes, z_dim, 3, False)(x)
print(capsule.shape)

output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(num_classes,))(capsule) #wang the norm of the vector

model = Model(inputs=allinputs, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

model.save_weights('output/Model_training.weights')


