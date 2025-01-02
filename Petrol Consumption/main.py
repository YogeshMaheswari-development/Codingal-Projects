import pandas as pd

df = pd.read_csv('https://drive.usercontent.google.com/u/0/uc?id=1SBy2zzKsIrsma-m5xOasFkjhlNVm97Oj&export=download')

print(df.head())

y = df.pop('Petrol_Consumption')
x = df
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

history = model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))

model.summary()

y_pred = model.predict(x_test)
print(y_pred)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))