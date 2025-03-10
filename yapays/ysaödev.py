# Gerekli Kütüphaneler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Veri Seti: Örnek Veri (Iris Dataset)
from sklearn.datasets import load_iris

data = load_iris()
X = data.data  # Özellikler
y = data.target  # Sınıflar

# Sınıf Etiketlerini Categorical Formata Dönüştürme
y = to_categorical(y, num_classes=len(np.unique(y)))

# Veri Seti Hakkında Bilgi
print("Veri Seti Özellikleri:")
print(f"- Özellik Sayısı: {X.shape[1]}")
print(f"- Örnek Sayısı: {X.shape[0]}")
print(f"- Sınıf Sayısı: {y.shape[1]}")

# Eğitim ve Test Verisi Ayırma (%66-%34)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)


# Yapay Sinir Ağı Modeli
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='softmax')  # Çok sınıflı sınıflandırma
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Eğitim/Test (%66-%34)
model = build_model(X_train.shape[1], y_train.shape[1])
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Model Performansı
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Doğruluğu: {test_accuracy:.4f}")

# Konfüzyon Matrisi ve Sınıflandırma Raporu
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nKonfüzyon Matrisi:")
print(confusion_matrix(y_true, y_pred))
print("\nSınıflandırma Raporu:")
print(classification_report(y_true, y_pred))

# 5-Fold Cross Validation
print("\n5-Fold Cross Validation Sonuçları:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = build_model(X_train.shape[1], y_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    fold_accuracies.append(val_accuracy)

print(f"Ortalama Doğruluk: {np.mean(fold_accuracies):.4f}")
print(f"Doğruluk Standart Sapması: {np.std(fold_accuracies):.4f}")

# Sonuçları Görselleştirme
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kaybı')
plt.legend()
plt.show()
