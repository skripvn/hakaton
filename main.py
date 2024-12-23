# Импорт необходимых библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


df = pd.read_csv('dataset(in).csv', delimiter=';')  # Убедитесь, что путь к файлу правильный


bool_columns = ['firstBlood', 'firstTower', 'firstBaron', 'firstDragon', 'firstRiftHerald']  # Укажите все булевые столбцы
for col in bool_columns:
    df[col] = df[col].map({True: 1, False: 0})


df['teamId'] = df['teamId'].map({100: 0, 200: 1})


df['win'] = df['win'].map({'Win': 1, 'Fail': 0})


df.drop(columns=['dominionVictoryScore'], inplace=True)


scaler = MinMaxScaler()
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()  # Выбор всех числовых колонок
df[num_cols] = scaler.fit_transform(df[num_cols])


df = df.astype(np.float32)


X = df.drop('win', axis=1)
y = df['win']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Тестовая точность: {accuracy:.4f}')

model.save("FILE_NAME.h5")
