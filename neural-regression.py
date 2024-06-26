from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error
from numpy import arange

from numpy.random import seed
seed(1)
from tensorflow import random, config
random.set_seed(1)
config.experimental.enable_op_determinism()
import random
random.seed(2)

# Read dataset:
dataset = pd.read_csv('/Users/bogdanoancea/Library/CloudStorage/OneDrive-Personal/Scoala/BazeleInformaticii/MachineLearning/Saptamana4/train.csv')
print(f"There are {len(dataset.index)} instances.")
print(dataset.head())


plt.figure()
plt.scatter(dataset['x'], dataset['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

def split_dataset(dataset, train_frac=0.7):
    train = dataset.sample(frac=train_frac)
    val = dataset.drop(train.index)
    return train, val

# Split dataset into train and validation:
train, validation = split_dataset(dataset, train_frac=0.7)
plt.scatter(train['x'], train['y'], c='blue', alpha=0.4)
plt.scatter(validation['x'], validation['y'], c='red', alpha=0.4)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['train set', 'validation set'], framealpha=0.3)
plt.show()


# Create model:
model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
print(model.summary())

# Train:
epochs = 2000
x_train, y_train = train['x'], train['y']
x_val, y_val = validation['x'], validation['y']
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=1, validation_data=(x_val, y_val))


# Display loss:
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# Display metric:
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model mean absolute error (mae)')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

test = pd.read_csv('/Users/bogdanoancea/Library/CloudStorage/OneDrive-Personal/Scoala/BazeleInformaticii/MachineLearning/Saptamana4/test.csv')
print(f"There are {len(dataset.index)} instances.")

test_results = model.evaluate(test['x'], test['y'], verbose=1)
print(f'Test set: - loss: {test_results[0]} - mae: {test_results[1]}')

# Other metrics:
train_pred = model.predict(x_train)
val_pred = model.predict(validation['x'])
test_pred = model.predict(test['x'])
print("Displaying other metrics:")
print("\t\tMedAE\tMAPE")
print(f"Train:\t{round(median_absolute_error(y_train, train_pred) ,3)}\t{round(mean_absolute_percentage_error(y_train, train_pred), 3)}")
print(f"Val  :\t{round(median_absolute_error(validation['y'], val_pred) ,3)}\t{round(mean_absolute_percentage_error(validation['y'], val_pred), 3)}")
print(f"Test :\t{round(median_absolute_error(test['y'], test_pred) ,3)}\t{round(mean_absolute_percentage_error(test['y'], test_pred), 3)}")

# Display function:
plt.scatter(dataset['x'], dataset['y'])
x = arange(0, 25, 0.1)
df = pd.DataFrame(data = x)
plt.plot(df, model.predict(df), c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

