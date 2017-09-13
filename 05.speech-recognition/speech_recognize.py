import tflearn
import pyaudio
import speech_data
import numpy
import sys

# Parse command line argument
argument_file =  sys.argv[1]

# Time vs accuracy tradeoff
learning_rate   = 0.0001 # def: 0.0001

# Steps
training_iters  = 100 # def: 300 000

batch_size      = 64

classes         = 10 # 10 ditigts

amount_n_epoch  = 1

# Data: A set of recorded digits
# Use Mel-frequency cepstrum for transformation
# Return labeled speech files as a batch
batch=speech_data.wave_batch_generator(10000,target=speech_data.Target.digits)

# Split the batch of data to:
# [training]
# [testing]

X,Y = next(batch)

#* Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

trainX, trainY = X, Y
testX, testY = X, Y

# Neural network: Init
# width  = 20          -  mfcc features
# height = 80          -  (max) length of utterance
net = tflearn.input_data(shape=[None, 8192])

# Feed the output from the previous layer as an input to the next one
# [dropout] - how many neurons should be dropped, to prevent overtraing
#             so data is forced to find new paths, allowing generalization
#net = tflearn.lstm(net, 128, dropout=0.8)

# Fully connected layer
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')

# Regression - output a single number as a result
# gradient type - [adam] (https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
# [adam] - maintains a gradient for all parameters, unlike the classic stochastic gradient function
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

# Initialize the network
model = tflearn.DNN(net, tensorboard_verbose=1)

current_step = 0

while current_step < training_iters:
    print('cur:', current_step)
    print('max:', training_iters)
    model.fit(trainX, trainY, n_epoch=amount_n_epoch, validation_set=(testX, testY),
              show_metric=True, batch_size=batch_size)

    _y = model.predict(X)
    current_step = current_step + 1

# Save the model, do not train it everytime
# model.save('0-10_digits_model_02.tflearn')
# Load a model
model.load('0-10_digits_model.tflearn')


demo_file = argument_file
demo = speech_data.load_wav_file(demo_file)

result = model.predict([demo])
result = numpy.argmax(result)

print("predicted digit for %s : result = %d "%(demo_file,result))
