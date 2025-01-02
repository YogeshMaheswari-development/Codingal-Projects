import pandas as pd
import numpy as np

df = pd.read_csv('pokemon.csv')

print(df.head())

y = df.pop('is_legendary')
x = df
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_dim=33, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train.select_dtypes(include=[int, float]).fillna(0).replace([np.inf, -np.inf], 0), y_train, epochs=100, verbose=2, validation_data=(x_test.select_dtypes(include=[int, float]).fillna(0).replace([np.inf, -np.inf], 0), y_test))

model.summary()

y_pred = model.predict(x_test.select_dtypes(include=[int, float]).fillna(0).replace([np.inf, -np.inf], 0))
from sklearn import metrics
y_pred = np.nan_to_num(y_pred)
y_pred = np.where(y_pred > 0.5, 1, 0)
print(y_pred)
print('Confusion Matrix: \n', metrics.confusion_matrix(y_test, y_pred))
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

