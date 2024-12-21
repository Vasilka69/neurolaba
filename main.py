import os
from random import randint

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import datasets, layers, models, optimizers, utils, Model
from sklearn.metrics import accuracy_score


def get_trained_mlp_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train,
        y_train,
        epochs=30,
        validation_data=(x_test, y_test)
    )

    return model


class Hamming:
    def __init__(self):
        self.load_and_preprocess_mnist()

        images = []
        for i in range(10):
            images.append(self.X_train[self.y_train == i].mean(axis=0))
        self.prototypes = np.array(images)

    def classify(self, input_vector):
        distances = np.sum(np.abs(self.prototypes - input_vector), axis=1)
        return np.argmin(distances)

    def load_and_preprocess_mnist(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)

        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test


def get_trained_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    opt = optimizers.SGD(learning_rate=0.001, momentum=0.9)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train,
        y_train,
        epochs=30,
        validation_data=(x_test, y_test),
        shuffle=True
    )

    return model


def train_or_load_keras_model(train_model_func, trained_model_path: str, force_train: bool = False):
    if not force_train and os.path.exists(trained_model_path):
        model = keras.models.load_model(trained_model_path)
    else:
        model = train_model_func()

        model.save(trained_model_path)

    model.summary()
    return model


def show_random_images(count=10):
    fig, ax = plt.subplots(2, count, figsize=(10, 5))

    train_axes = ax[0].flatten()
    test_axes = ax[1].flatten()
    for i in range(count):
        train_axes[i].axis('off')
        test_axes[i].axis('off')

        train_image_index = randint(0, len(x_train))
        test_image_index = randint(0, len(x_test))
        train_axes[i].imshow(x_train[train_image_index])
        test_axes[i].imshow(x_test[test_image_index])

        train_axes[i].set_title(np.argmax(y_train[train_image_index]))
        test_axes[i].set_title(np.argmax(y_test[test_image_index]))

    train_y = train_axes[0].get_position().ymax
    test_y = test_axes[0].get_position().ymax

    fig.text(0.5, train_y + 0.1, "Образцы из тренировочного набора", ha='center', fontsize=28)
    fig.text(0.5, test_y + 0.1, "Образцы из тестового набора", ha='center', fontsize=28)

    plt.show()


def show_model_training_metrics(history: dict, label):
    try:
        loss = history['loss']
        val_loss = history['val_loss']
        accuracy = history['accuracy']
        val_accuracy = history['val_accuracy']
    except Exception:
        print(f'Не удалось извлечь историю обучения модели {label} для построения графиков.')
        return

    px = 1 / plt.rcParams['figure.dpi']
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(1800*px, 500*px))

    fig.text(0.5, 0.9, label, ha='center', fontsize=28)

    ax1.plot(loss, color='green')
    ax1.plot(val_loss, color='red')
    ax1.legend(['loss', 'validate loss'])
    ax1.grid(True)

    ax2.plot(accuracy, color='green')
    _ = ax2.plot(val_accuracy, color="red")
    ax2.legend(['accuracy', 'validate accuracy'])

    plt.show()


def show_predictions(mlp_model: Model, hamming_model: Hamming, cnn_model: Model, count=10):
    fig, ax = plt.subplots(3, count, figsize=(10, 5))

    mlp_axes = ax[0].flatten()
    hamming_axes = ax[1].flatten()
    cnn_axes = ax[2].flatten()
    for i in range(count):
        mlp_axes[i].axis('off')
        hamming_axes[i].axis('off')
        cnn_axes[i].axis('off')

        test_image_index = randint(0, len(x_test))
        test_image = x_test[test_image_index]
        test_answer = np.argmax(y_test[test_image_index])
        hamming_test_image = hamming_model.X_test[test_image_index]

        mlp_axes[i].imshow(test_image)
        hamming_axes[i].imshow(test_image)
        cnn_axes[i].imshow(test_image)

        mlp_prediction = np.argmax(mlp_model.predict(np.expand_dims(test_image, 0)))
        hamming_prediction = hamming_model.classify(hamming_test_image)
        cnn_prediction = np.argmax(cnn_model.predict(np.expand_dims(test_image, 0)))

        mlp_axes[i].set_title(f'{mlp_prediction}{"=" if mlp_prediction == test_answer else "<>"}{test_answer}', color="g" if mlp_prediction == test_answer else "r")
        hamming_axes[i].set_title(f'{hamming_prediction}{"=" if hamming_prediction == test_answer else "<>"}{test_answer}', color="g" if hamming_prediction == test_answer else "r")
        cnn_axes[i].set_title(f'{cnn_prediction}{"=" if cnn_prediction == test_answer else "<>"}{test_answer}', color="g" if cnn_prediction == test_answer else "r")

    fig.text(mlp_axes[0].get_position().xmin, mlp_axes[0].get_position().ymax + 0.07, "MLP", ha='left', fontsize=20)
    fig.text(hamming_axes[0].get_position().xmin, hamming_axes[0].get_position().ymax + 0.07, "Сеть Хемминга", ha='left', fontsize=20)
    fig.text(cnn_axes[0].get_position().xmin, cnn_axes[0].get_position().ymax + 0.07, "CNN", ha='left', fontsize=20)

    fig.text(0.5, 0.05, "Результаты работы нейросетей", ha='center', fontsize=24)

    plt.show()


def show_accuracies_comparison(accuracies: dict[str, str]):
    plt.bar(accuracies.keys(), accuracies.values())
    plt.xlabel('Модель нейронной сети')
    plt.ylabel('Точность')
    plt.title('Сравнение эффективности \nмоделей классификации рукописных цифр')
    plt.show()


def init_keras_dataset():
    global x_train, y_train, x_test, y_test
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)


TRAINED_MODELS_DIR = 'trained_models'

MLP_TRAINED_MODEL_PATH = f'{TRAINED_MODELS_DIR}/mnist_mlp.keras'
CNN_TRAINED_MODEL_PATH = f'{TRAINED_MODELS_DIR}/mnist_cnn.keras'

x_train = np.array([])
y_train = np.array([])
x_test = np.array([])
y_test = np.array([])

FORCE_RETRAIN = True


def main():
    init_keras_dataset()

    show_random_images()

    # Модели
    mlp_model = train_or_load_keras_model(get_trained_mlp_model, MLP_TRAINED_MODEL_PATH, FORCE_RETRAIN)
    hamming_model = Hamming()
    cnn_model = train_or_load_keras_model(get_trained_cnn_model, CNN_TRAINED_MODEL_PATH, FORCE_RETRAIN)

    # Графики процесса обучения моделей
    if FORCE_RETRAIN:
        show_model_training_metrics(mlp_model.history.history, label="MLP")
        show_model_training_metrics(cnn_model.history.history, label="CNN")

    # Точности
    mlp_accuracy = mlp_model.evaluate(x_test, y_test)[1]
    hamming_accuracy = accuracy_score(hamming_model.y_test, [hamming_model.classify(x) for x in hamming_model.X_test])
    cnn_accuracy = cnn_model.evaluate(x_test, y_test)[1]

    print('\nТочности работы нейросетей на тестовых данных:')
    print(f"MLP: {mlp_accuracy:.4f}")
    print(f"Сеть Хемминга: {hamming_accuracy:.4f}")
    print(f"CNN: {cnn_accuracy:.4f}")
    print()

    # Графики
    ## Эффективность
    labels = ['MLP', 'Сеть Хемминга', 'CNN']
    accuracies = [mlp_accuracy, hamming_accuracy, cnn_accuracy]
    show_accuracies_comparison(dict(zip(labels, accuracies)))

    ## Примеры предсказывания нейросетей
    show_predictions(mlp_model=mlp_model, hamming_model=hamming_model, cnn_model=cnn_model)


if __name__ == '__main__':
    main()
