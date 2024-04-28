!pip install scikit-plot

import random
import os
import glob
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Sequential
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scikitplot.metrics import plot_roc

class CFG:
    EPOCHS = 30
    BATCH_SIZE = 32
    SEED = 42
    TF_SEED = 768
    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3
    IMAGE_SIZE = (224, 224, 3)

# Yolların tanımlanması
DATASET_PATH = "/content/drive/MyDrive/data_source/data/real-and-fake-face-detec/real-vs-fake/real-vs-fake/"
TRAIN_PATH = '/content/drive/MyDrive/data_source/data/real-and-fake-face-detec/real-vs-fake/real-vs-fake/train/'
TEST_PATH = '/content/drive/MyDrive/data_source/data/real-and-fake-face-detec/real-vs-fake/real-vs-fake/test/'

# Veri kümesinin bir özetinin oluşturulması
print('DATASET SUMMARY')
print('========================\n')
for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")
print('\n========================')

# Commented out IPython magic to ensure Python compatibility.
# %%time
# train_images = glob.glob(f"{TRAIN_PATH}**/*.jpg")
# test_images = glob.glob(f"{TEST_PATH}**/*.jpg")

# Eğitim ve test seti boyutlarının alınması
train_size = len(train_images)
test_size = len(test_images)

# Veri kümesi boyutunun alınması
total = train_size + test_size

# Örnek sayılarının görüntülenmesi
print(f'train samples count:\t\t{train_size}')
print(f'test samples count:\t\t{test_size}')
print('=======================================')
print(f'TOTAL:\t\t\t\t{total}')

def generate_labels(image_paths):
    return [_.split('/')[-2:][0] for _ in image_paths]


def build_df(image_paths, labels):
    # Dataframe oluşturulması
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': generate_labels(labels)
    })

    # Etiket kodlamalarının oluşturulması
    df['label_encoded'] = df.apply(lambda row: 0 if row.label == 'fake' else 1, axis=1)

   # Veri çerçevesinin karıştırılması ve döndürülmesi
    return df.sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)

# DataFrames'i oluşturulması
train_df = build_df(train_images, generate_labels(train_images))
test_df = build_df(test_images, generate_labels(test_images))

# Eğitim setindeki ilk 5 örneğin görüntülenmesi
train_df.head(5)

def _load(image_path):
  # Bir görüntü dosyasınının uint8 tensöre okunması ve kodununun çözülmesi
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)

    # Görüntünün yeniden boyutlandırılması
    image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH],
                            method=tf.image.ResizeMethod.LANCZOS3)

    # Görüntü tipinin float32'ye dönüştürülmesi ve normalize edilmesi
    image = tf.cast(image, tf.float32)/255.

    # Görüntünün döndürülmesi
    return image

def view_sample(image, label, color_map='rgb', fig_size=(8, 10)):
    plt.figure(figsize=fig_size)

    if color_map=='rgb':
        plt.imshow(image)
    else:
        plt.imshow(tf.image.rgb_to_grayscale(image), cmap=color_map)

    plt.title(f'Label: {label}', fontsize=16)
    return

# train_df'den rastgele örnek seçilmesi
idx = random.sample(train_df.index.to_list(), 1)[0]

# Rastgele örneğin yüklenmesi ve etiketlenmesi
sample_image, sample_label = _load(train_df.image_path[idx]), train_df.label[idx]

# Rastgele örneğin görüntülenmesi
view_sample(sample_image, sample_label, color_map='inferno')

def view_mulitiple_samples(df, sample_loader, count=10, color_map='rgb', fig_size=(14, 10)):
    rows = count//5
    if count%5 > 0:
        rows +=1

    idx = random.sample(df.index.to_list(), count)
    fig = plt.figure(figsize=fig_size)

    for column, _ in enumerate(idx):
        plt.subplot(rows, 5, column+1)
        plt.title(f'Label: {df.label[_]}')

        if color_map=='rgb':
            plt.imshow(sample_loader(df.image_path[_]))
        else:
            plt.imshow(tf.image.rgb_to_grayscale(sample_loader(df.image_path[_])), cmap=color_map)

    return

view_mulitiple_samples(train_df, _load,
                       count=25, color_map='inferno',
                       fig_size=(20, 24))

fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10))

# Alt grafikler arasındaki boşluğun ayarlanması
fig.tight_layout(pad=6.0)

# Eğitim Etiketleri Dağılımının Çizilmesi
ax1.set_title('Train Labels Distribution', fontsize=20)
train_distribution = train_df['label'].value_counts().sort_values()
sns.barplot(x=train_distribution.values,
            y=list(train_distribution.keys()),
            orient="h",
            ax=ax1)

# Test Etiketleri Dağılımının Çizilmesi
ax2.set_title('Test Labels Distribution', fontsize=20)
test_distribution = test_df['label'].value_counts().sort_values()
sns.barplot(x=test_distribution.values,
            y=list(test_distribution.keys()),
            orient="h",
            ax=ax2);

# Eğitim Seti ile Train/Val ayrımı oluşturulması
train_split_idx, val_split_idx, _, _ = train_test_split(train_df.index,
                                                        train_df.label_encoded,
                                                        test_size=0.15,
                                                        stratify=train_df.label_encoded,
                                                        random_state=CFG.SEED)

# Yeni eğitim ve doğrulama verilerinin alınması
train_new_df = train_df.iloc[train_split_idx].reset_index(drop=True)
val_df = train_df.iloc[val_split_idx].reset_index(drop=True)

# Şekillerin görüntülenmesi
train_new_df.shape, val_df.shape

fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10))

# Alt grafikler arasındaki boşluğun ayarlanması
fig.tight_layout(pad=6.0)

# Yeni Eğitim Etiketlerinin Çizilmesi
ax1.set_title('New Train Labels Distribution', fontsize=20)
train_new_distribution = train_new_df['label'].value_counts().sort_values()
sns.barplot(x=train_new_distribution.values,
            y=list(train_new_distribution.keys()),
            orient="h",
            ax=ax1)

# Doğrulama Etiketleri Dağılımının Çizilmesi
ax2.set_title('Validation Labels Distribution', fontsize=20)
val_distribution = val_df['label'].value_counts().sort_values()
sns.barplot(x=val_distribution.values,
            y=list(val_distribution.keys()),
            orient="h",
            ax=ax2);

# Büyütme katmanının oluşturulması
augmentation_layer = Sequential([
    layers.RandomFlip(mode='horizontal_and_vertical', seed=CFG.TF_SEED),
    layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=CFG.TF_SEED),
], name='augmentation_layer')

image = tf.image.rgb_to_grayscale(sample_image)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

# Alt grafikler arasındaki boşluğun ayarlanması
fig.tight_layout(pad=6.0)

# Orijinal Resmin Görüntülenmesi
ax1.set_title('Original Image', fontsize=20)
ax1.imshow(image, cmap='inferno');

# Artırılmış Görüntünün Görüntülenmesi
ax2.set_title('Augmented Image', fontsize=20)
ax2.imshow(augmentation_layer(image), cmap='inferno');

def plot_training_curves(history):

    loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])

    accuracy = np.array(history.history['accuracy'])
    val_accuracy = np.array(history.history['val_accuracy'])

    epochs = range(len(history.history['loss']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Çizim Kaybı
    ax1.plot(epochs, loss, label='training_loss', marker='o')
    ax1.plot(epochs, val_loss, label='val_loss', marker='o')

    ax1.fill_between(epochs, loss, val_loss, where=(loss > val_loss), color='C0', alpha=0.3, interpolate=True)
    ax1.fill_between(epochs, loss, val_loss, where=(loss < val_loss), color='C1', alpha=0.3, interpolate=True)

    ax1.set_title('Loss (Lower Means Better)', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.legend()
    # Çizim Doğruluğu
    ax2.plot(epochs, accuracy, label='training_accuracy', marker='o')
    ax2.plot(epochs, val_accuracy, label='val_accuracy', marker='o')

    ax2.fill_between(epochs, accuracy, val_accuracy, where=(accuracy > val_accuracy), color='C0', alpha=0.3, interpolate=True)
    ax2.fill_between(epochs, accuracy, val_accuracy, where=(accuracy < val_accuracy), color='C1', alpha=0.3, interpolate=True)

    ax2.set_title('Accuracy (Higher Means Better)', fontsize=16)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.legend();

def plot_confusion_matrix(y_true, y_pred, classes='auto', figsize=(10, 10), text_size=12):
    # Karışıklık matrisi oluşturun
    cm = confusion_matrix(y_true, y_pred)

    # Çizim boyutunu ayarlayın
    plt.figure(figsize=figsize)

    # Karışıklık matrisi ısı haritası oluşturun
    disp = sns.heatmap(
        cm, annot=True, cmap='Greens',
        annot_kws={"size": text_size}, fmt='g',
        linewidths=1, linecolor='black', clip_on=False,
        xticklabels=classes, yticklabels=classes)

    # Başlık ve eksen etiketlerini ayarlayın
    disp.set_title('Confusion Matrix', fontsize=24)
    disp.set_xlabel('Predicted Label', fontsize=20)
    disp.set_ylabel('True Label', fontsize=20)
    plt.yticks(rotation=0)

    # Karışıklık matrisini çizin
    plt.show()

    return

def generate_preformance_scores(y_true, y_pred, y_probabilities):

    model_accuracy = accuracy_score(y_true, y_pred)
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true,
                                                                                 y_pred,
                                                                                 average="weighted")
    model_matthews_corrcoef = matthews_corrcoef(y_true, y_pred)

    print('=============================================')
    print(f'\nPerformance Metrics:\n')
    print('=============================================')
    print(f'accuracy_score:\t\t{model_accuracy:.4f}\n')
    print('_____________________________________________')
    print(f'precision_score:\t{model_precision:.4f}\n')
    print('_____________________________________________')
    print(f'recall_score:\t\t{model_recall:.4f}\n')
    print('_____________________________________________')
    print(f'f1_score:\t\t{model_f1:.4f}\n')
    print('_____________________________________________')
    print(f'matthews_corrcoef:\t{model_matthews_corrcoef:.4f}\n')
    print('=============================================')

    preformance_scores = {
        'accuracy_score': model_accuracy,
        'precision_score': model_precision,
        'recall_score': model_recall,
        'f1_score': model_f1,
        'matthews_corrcoef': model_matthews_corrcoef
    }
    return preformance_scores

def encode_labels(labels, encode_depth=2):
    return tf.one_hot(labels, depth=encode_depth).numpy()

def create_pipeline(df, load_function, augment=False, batch_size=32, shuffle=False, cache=None, prefetch=False):

    # DataFrame'den görüntü yollarını ve etiketleri alın
    image_paths = df.image_path
    image_labels = encode_labels(df.label_encoded)
    AUTOTUNE = tf.data.AUTOTUNE

    # DataFrame'den ham verilerle veri kümesi oluşturun
    ds = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))

    # Artırma katmanı ve yükleme işlevini veri kümesi girdileriyle eşleştirin, eğer artırma True ise
    # Else yalnızca yükleme işlevini eşle
    if augment:
        ds = ds.map(lambda x, y: (augmentation_layer(load_function(x)), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (load_function(x), y), num_parallel_calls=AUTOTUNE)

    # Koşula göre karıştırma uygulayın
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    # Toplu iş uygula
    ds = ds.batch(batch_size)

    # Koşula göre önbelleğe alma uygulanması
    if cache != None:
        ds = ds.cache(cache)
    # Koşula göre ön-getirme uygulayın
    # Not: Bu, bellekten ödün verilmesine neden olacaktır
    if prefetch:
        ds = ds.prefetch(buffer_size=AUTOTUNE)

    # Veri kümesinin döndürülmesi
    return ds

# Eğitim Girişi Boru Hattı Oluşturulması
train_ds = create_pipeline(train_new_df, _load, augment=True,
                           batch_size=CFG.BATCH_SIZE,
                           shuffle=False, prefetch=True)

# Doğrulama Girdisi Boru Hattı Oluşturulması
val_ds = create_pipeline(val_df, _load,
                         batch_size=CFG.BATCH_SIZE,
                         shuffle=False, prefetch=False)

# Test Girdisi Boru Hattı Oluşturun
test_ds = create_pipeline(test_df, _load,
                          batch_size=CFG.BATCH_SIZE,
                          shuffle=False, prefetch=False)

# Veri kümelerinin dize gösteriminin görüntülenmesi
print('========================================')
print('Train Input Data Pipeline:\n\n', train_ds)
print('========================================')
print('Validation Input Data Pipeline:\n\n', val_ds)
print('========================================')
print('Test Input Data Pipeline:\n\n', test_ds)
print('========================================')

"""CNN"""

def cnn_model():

    initializer = tf.keras.initializers.GlorotNormal()

    cnn_sequential = Sequential([
        layers.Input(shape=CFG.IMAGE_SIZE, dtype=tf.float32, name='input_image'),

        layers.Conv2D(16, kernel_size=3, activation='relu', kernel_initializer=initializer),
        layers.Conv2D(16, kernel_size=3, activation='relu', kernel_initializer=initializer),
        layers.MaxPool2D(pool_size=2, padding='valid'),

        layers.Conv2D(8, kernel_size=3, activation='relu', kernel_initializer=initializer),
        layers.Conv2D(8, kernel_size=3, activation='relu', kernel_initializer=initializer),
        layers.MaxPool2D(pool_size=2),

        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_initializer=initializer),
        layers.Dense(2, activation='sigmoid', kernel_initializer=initializer)
    ], name='cnn_sequential_model')

    return cnn_sequential

# Model Oluşturulması
model_cnn = cnn_model()

# Modelin Özetinin Oluşturulması
model_cnn.summary()

# Modelin görsel olarak keşfedilmesi
plot_model(
    model_cnn, dpi=60,
    show_shapes=True
)

def train_model(model, num_epochs, callbacks_list, tf_train_data,
                tf_valid_data=None, shuffling=False):

    model_history = {}

    if tf_valid_data != None:
        model_history = model.fit(tf_train_data,
                                  epochs=num_epochs,
                                  validation_data=tf_valid_data,
                                  validation_steps=int(len(tf_valid_data)),
                                  callbacks=callbacks_list,
                                  shuffle=shuffling)

    if tf_valid_data == None:
        model_history = model.fit(tf_train_data,
                                  epochs=num_epochs,
                                  callbacks=callbacks_list,
                                  shuffle=shuffling)
    return model_history

# Erken Durdurma Geri Çağrısı Tanımlanması
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True)

# Öğrenme Oranını Azalt Geri Çağrısının Tanımlanması
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.1,
    verbose=1)

# Geri Çağırmaları ve Metrik listelerin tanımlanması
CALLBACKS = [early_stopping_callback, reduce_lr_callback]
METRICS = ['accuracy']

tf.random.set_seed(CFG.SEED)

# Modelin derlenmesi
model_cnn.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=METRICS
)

# Modelin eğitilmesi
print(f'Training {model_cnn.name}.')
print(f'Train on {len(train_new_df)} samples, validate on {len(val_df)} samples.')
print('----------------------------------')

cnn_history = train_model(
    model_cnn, CFG.EPOCHS, CALLBACKS,
    train_ds, val_ds,
    shuffling=False
)

# Modelin değerlendirilmesi
cnn_evaluation = model_cnn.evaluate(test_ds)

# Model olasılıkları ve ilişkili tahminlerin oluşturulması
cnn_test_probabilities = model_cnn.predict(test_ds, verbose=1)
cnn_test_predictions = tf.argmax(cnn_test_probabilities, axis=1)

# CNN modeli eğitim geçmişinin çizilmesi
plot_training_curves(cnn_history)

class_names = ['Fake', 'Real']

plot_confusion_matrix(
    test_df.label_encoded,
    cnn_test_predictions,
    figsize=(8, 8),
    classes=class_names)

plot_roc(test_df.label_encoded,
         cnn_test_probabilities,
         figsize=(10, 10), title_fontsize='large');

# CNN ROC Eğrileri
print(classification_report(test_df.label_encoded,
                            cnn_test_predictions,
                            target_names=class_names))

# CNN modeli performans skoru
cnn_performance = generate_preformance_scores(test_df.label_encoded,
                                              cnn_test_predictions,
                                              cnn_test_probabilities)

# Tensorflow hub'ından herhangi bir model/önişlemci almak için bir fonksiyon
def get_tfhub_model(model_link, model_name, model_trainable=False):
    return hub.KerasLayer(model_link,
                          trainable=model_trainable,
                          name=model_name)

"""EFFİCİENTNET"""

# EfficientNet V2 B0
efficientnet_v2_url = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2'
model_name = 'efficientnet_v2_b0'

# Yalnızca çıkarım için eğitilebilirin False olarak ayarlanması
set_trainable=False

efficientnet_v2_b0 = get_tfhub_model(efficientnet_v2_url,
                                     model_name,
                                     model_trainable=set_trainable)

def efficientnet_v2_model():

    initializer = tf.keras.initializers.GlorotNormal()

    efficientnet_v2_sequential = Sequential([
        layers.Input(shape=CFG.IMAGE_SIZE, dtype=tf.float32, name='input_image'),
        efficientnet_v2_b0,
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_initializer=initializer),
        layers.Dense(2, dtype=tf.float32, activation='sigmoid', kernel_initializer=initializer)
    ], name='efficientnet_v2_sequential_model')

    return efficientnet_v2_sequential

# Modelin Oluşturulması
model_efficientnet_v2 = efficientnet_v2_model()

# Modelin Özetinin Oluşturulması
model_efficientnet_v2.summary()

# Modelin görsel olarak keşfedilmesi
plot_model(
    model_efficientnet_v2, dpi=60,
    show_shapes=True
)

tf.random.set_seed(CFG.SEED)

# Modelin Derlenmesi
model_efficientnet_v2.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=METRICS
)

# Modelin eğitilmesi
print(f'Training {model_efficientnet_v2.name}.')
print(f'Train on {len(train_new_df)} samples, validate on {len(val_df)} samples.')
print('----------------------------------')

efficientnet_v2_history = train_model(
    model_efficientnet_v2, CFG.EPOCHS, CALLBACKS,
    train_ds, val_ds,
    shuffling=False
)

# Modelin değerlendirilmesi
efficientnet_v2_evaluation = model_efficientnet_v2.evaluate(test_ds)

# Model olasılıkları ve ilişkili tahminlerin oluşturulması
efficientnet_v2_test_probabilities = model_efficientnet_v2.predict(test_ds, verbose=1)
efficientnet_v2_test_predictions = tf.argmax(efficientnet_v2_test_probabilities, axis=1)

# EfficientNet V2 modelinin eğitim geçmişinin çizilmesi
plot_training_curves(efficientnet_v2_history)

plot_confusion_matrix(
    test_df.label_encoded,
    efficientnet_v2_test_predictions,
    figsize=(8, 8),
    classes=class_names)

plot_roc(test_df.label_encoded,
         efficientnet_v2_test_probabilities,
         figsize=(10, 10), title_fontsize='large');

# EfficientNet V2 modelinin ROC Eğrileri
print(classification_report(test_df.label_encoded,
                            efficientnet_v2_test_predictions,
                            target_names=class_names))

# EfficieNet model performans skoru
efficientnet_v2_performance = generate_preformance_scores(test_df.label_encoded,
                                                          efficientnet_v2_test_predictions,
                                                          efficientnet_v2_test_probabilities)

"""VGG16"""

from tensorflow.keras.applications import VGG16

def vgg16_model():

    initializer = tf.keras.initializers.GlorotNormal()

    # VGG-16 modelini alalım
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=CFG.IMAGE_SIZE)

    # VGG-16 modelinin katmanlarını donatalım
    vgg16_model.trainable = set_trainable
    for layer in vgg16_model.layers:
        layer.trainable = set_trainable

    # Modelimizi oluşturalım
    vgg16_sequential = Sequential([
        layers.Input(shape=CFG.IMAGE_SIZE, dtype=tf.float32, name='input_image'),
        vgg16_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_initializer=initializer),
        layers.Dense(2, dtype=tf.float32, activation='sigmoid', kernel_initializer=initializer)
    ], name='vgg16_sequential_model')

    return vgg16_sequential

# Modeli oluşturalım
model_vgg16 = vgg16_model()

# Modelin özetini oluşturalım
model_vgg16.summary()

# Modeli görsel olarak gösterelim
plot_model(model_vgg16, dpi=60, show_shapes=True)

# Modeli derleyelim
model_vgg16.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=METRICS
)

# Modeli eğitelim
print(f'Training {model_vgg16.name}.')
print(f'Train on {len(train_new_df)} samples, validate on {len(val_df)} samples.')
print('----------------------------------')

vgg16_history = train_model(
    model_vgg16, CFG.EPOCHS, CALLBACKS,
    train_ds, val_ds,
    shuffling=False
)

# Modeli değerlendirelim
vgg16_evaluation = model_vgg16.evaluate(test_ds)

# Model olasılıkları ve ilişkili tahminleri oluşturalım
vgg16_test_probabilities = model_vgg16.predict(test_ds, verbose=1)
vgg16_test_predictions = tf.argmax(vgg16_test_probabilities, axis=1)

# VGG16 modelinin eğitim geçmişinin çizilmesi
plot_training_curves(vgg16_history)

plot_confusion_matrix(
    test_df.label_encoded,
    vgg16_test_predictions,
    figsize=(8, 8),
    classes=class_names)

plot_roc(test_df.label_encoded,
         vgg16_test_probabilities,
         figsize=(10, 10), title_fontsize='large');

# VGG16 modelinin ROC Eğrileri
print(classification_report(test_df.label_encoded,
                            vgg16_test_predictions,
                            target_names=class_names))

# VGG16 model performans skoru
vgg16_performance = generate_preformance_scores(test_df.label_encoded,
                                                          vgg16_test_predictions,
                                                          vgg16_test_probabilities)

"""DENSENET"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, Sequential

def create_densenet_model():

    initializer = tf.keras.initializers.GlorotNormal()

    # Önceden eğitilmiş DenseNet121 modelini yükleyin
    densenet_model = DenseNet121(include_top=False, weights='imagenet', input_shape=CFG.IMAGE_SIZE)

    # Yalnızca çıkarım için eğitilebilirin False olarak ayarlanması
    densenet_model.trainable = False

    densenet_sequential = Sequential([
        layers.Input(shape=CFG.IMAGE_SIZE, dtype=tf.float32, name='input_image'),
        densenet_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_initializer=initializer),
        layers.Dense(2, dtype=tf.float32, activation='sigmoid', kernel_initializer=initializer)
    ], name='densenet_sequential_model')

    return densenet_sequential

# Modelin Oluşturulması
model_densenet = create_densenet_model()

# Modelin özetini oluşturalım
model_densenet.summary()

# Modelin görsel olarak keşfedilmesi
plot_model(
    model_densenet, dpi=60,
    show_shapes=True
)

tf.random.set_seed(CFG.SEED)

# Modelin Derlenmesi
model_densenet.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=METRICS
)

# Modelin eğitilmesi
print(f'Training {model_densenet.name}.')
print(f'Train on {len(train_new_df)} samples, validate on {len(val_df)} samples.')
print('----------------------------------')

densenet_history = train_model(
    model_densenet, CFG.EPOCHS, CALLBACKS,
    train_ds, val_ds,
    shuffling=False
)

# Modelin değerlendirilmesi
densenet_evaluation = model_densenet.evaluate(test_ds)

# Model olasılıkları ve ilişkili tahminlerin oluşturulması
densenet_test_probabilities = model_densenet.predict(test_ds, verbose=1)
densenet_test_predictions = tf.argmax(densenet_test_probabilities, axis=1)

# Model olasılıkları ve ilişkili tahminleri oluşturalım
densenet_test_probabilities = model_densenet.predict(test_ds, verbose=1)
densenet_test_predictions = tf.argmax(densenet_test_probabilities, axis=1)

# Densenet modelinin eğitim geçmişinin çizilmesi
plot_training_curves(densenet_history)

plot_confusion_matrix(
    test_df.label_encoded,
    densenet_test_predictions,
    figsize=(8, 8),
    classes=class_names)

plot_roc(test_df.label_encoded,
         densenet_test_probabilities,
         figsize=(10, 10), title_fontsize='large');

# DENSENET modelinin ROC Eğrileri
print(classification_report(test_df.label_encoded,
                            densenet_test_predictions,
                            target_names=class_names))

# DENSENET model performans skoru
densenet_performance = generate_preformance_scores(test_df.label_encoded,
                                                          densenet_test_predictions,
                                                          densenet_test_probabilities)

"""RESNET50"""

from tensorflow.keras.applications import ResNet50

# RESNET50 Model
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=CFG.IMAGE_SIZE)

def resnet50_model():
    initializer = tf.keras.initializers.GlorotNormal()

    resnet50_sequential = Sequential([
        layers.Input(shape=CFG.IMAGE_SIZE, dtype=tf.float32, name='input_image'),
        resnet50,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_initializer=initializer),
        layers.Dense(2, dtype=tf.float32, activation='sigmoid', kernel_initializer=initializer)
    ], name='resnet50_sequential_model')

    return resnet50_sequential

# Modelin Oluşturulması
model_resnet50 = resnet50_model()

# Modelin Özetinin Oluşturulması
model_resnet50.summary()

# Modelin görsel olarak keşfedilmesi
plot_model(model_resnet50, dpi=60, show_shapes=True)

tf.random.set_seed(CFG.SEED)

# Modelin Derlenmesi
model_resnet50.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=METRICS
)

# Modelin eğitilmesi
print(f'Training {model_resnet50.name}.')
print(f'Train on {len(train_new_df)} samples, validate on {len(val_df)} samples.')
print('----------------------------------')

resnet50_history = train_model(
    model_resnet50, CFG.EPOCHS, CALLBACKS,
    train_ds, val_ds,
    shuffling=False
)

# Modelin değerlendirilmesi
resnet50_evaluation = model_resnet50.evaluate(test_ds)

# Model olasılıkları ve ilişkili tahminlerin oluşturulması
resnet50_test_probabilities = model_resnet50.predict(test_ds, verbose=1)
resnet50_test_predictions = tf.argmax(resnet50_test_probabilities, axis=1)

# RESNET50 modelinin eğitim geçmişinin çizilmesi
plot_training_curves(resnet50_history)

plot_roc(test_df.label_encoded,
         resnet50_test_probabilities,
         figsize=(10, 10), title_fontsize='large');

# RESNET50 modelinin ROC Eğrileri
print(classification_report(test_df.label_encoded,
                            resnet50_test_predictions,
                            target_names=class_names))

# RESNET50 model performans skoru
resnet50_performance = generate_preformance_scores(test_df.label_encoded,
                                                          resnet50_test_predictions,
                                                          resnet50_test_probabilities)

"""FACENET"""

# import tensorflow as tf
# from tensorflow.keras.applications import InceptionResNetV2
# from tensorflow.keras import layers, models, Sequential
# from tensorflow.keras.utils import plot_model

def create_facenet_model():

    initializer = tf.keras.initializers.GlorotNormal()

    # Önceden eğitilmiş InceptionResNetV2 modelini yükleyin
    facenet_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=CFG.IMAGE_SIZE)

    # Yalnızca çıkarım için eğitilebilirin False olarak ayarlanması
    facenet_model.trainable = False

    facenet_sequential = Sequential([
        layers.Input(shape=CFG.IMAGE_SIZE, dtype=tf.float32, name='input_image'),
        facenet_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_initializer=initializer),
        layers.Dense(2, dtype=tf.float32, activation='sigmoid', kernel_initializer=initializer)
    ], name='facenet_sequential_model')

    return facenet_sequential

# Modelin Oluşturulması
model_facenet = create_facenet_model()

# Modelin özetini oluşturalım
model_facenet.summary()

# Modelin görsel olarak keşfedilmesi
plot_model(
    model_facenet, dpi=60,
    show_shapes=True
)

tf.random.set_seed(CFG.SEED)

# Modelin Derlenmesi
model_facenet.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=METRICS
)

# Modelin eğitilmesi
print(f'Training {model_facenet.name}.')
print(f'Train on {len(train_new_df)} samples, validate on {len(val_df)} samples.')
print('----------------------------------')

facenet_history = train_model(
    model_facenet, CFG.EPOCHS, CALLBACKS,
    train_ds, val_ds,
    shuffling=False
)

# Modelin değerlendirilmesi
facenet_evaluation = model_facenet.evaluate(test_ds)

# Model olasılıkları ve ilişkili tahminlerin oluşturulması
facenet_test_probabilities = model_facenet.predict(test_ds, verbose=1)
facenet_test_predictions = tf.argmax(facenet_test_probabilities, axis=1)

# FACENET modeli eğitim geçmişinin çizilmesi
plot_training_curves(facenet_history)

plot_confusion_matrix(
    test_df.label_encoded,
    facenet_test_predictions,
    figsize=(8, 8),
    classes=class_names)

plot_roc(test_df.label_encoded,
         facenet_test_probabilities,
         figsize=(10, 10), title_fontsize='large');

# FACENET ROC eğrisi
print(classification_report(test_df.label_encoded,
                            facenet_test_predictions,
                            target_names=class_names))

# FACENET model performans skoru
facenet_performance = generate_preformance_scores(test_df.label_encoded,
                                                 facenet_test_predictions,
                                                  facenet_test_probabilities)

"""VİT B16"""

!pip install -q vit-keras

!pip install tensorflow-addons

from vit_keras import vit

# Modelin indirilmesi
vit_model = vit.vit_b16(
        image_size=224,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=2)

# Model katmanlarının yalnızca çıkarım modu için dondurulması
for layer in vit_model.layers:
    layer.trainable = False

def vit_b16_model():

    initializer = tf.keras.initializers.GlorotNormal()

    vit_b16_sequential = Sequential([
        layers.Input(shape=CFG.IMAGE_SIZE, dtype=tf.float32, name='input_image'),
        vit_model,
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_initializer=initializer),
        layers.Dense(2, dtype=tf.float32, activation='sigmoid', kernel_initializer=initializer)
    ], name='vit_b16_sequential_model')

    return vit_b16_sequential

# Modelin Oluşturulması
model_vit_b16 = vit_b16_model()

# Modelin Özetinin Oluşturulması
model_vit_b16.summary()

# Modelin görsel olarak keşfedilmesi
plot_model(
    model_vit_b16, dpi=60,
    show_shapes=True
)

tf.random.set_seed(CFG.SEED)

# Modelin derlenmesi
model_vit_b16.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=METRICS
)

# Modelin eğitilmesi
print(f'Training {model_vit_b16.name}.')
print(f'Train on {len(train_new_df)} samples, validate on {len(val_df)} samples.')
print('----------------------------------')

vit_b16_history = train_model(
    model_vit_b16, CFG.EPOCHS, CALLBACKS,
    train_ds, val_ds,
    shuffling=False
)

# Modelin değerlendirilmesi
vit_b16_evaluation = model_vit_b16.evaluate(test_ds)

# Model olasılıkları ve ilişkili tahminlerin oluşturulması
vit_b16_test_probabilities = model_vit_b16.predict(test_ds, verbose=1)
vit_b16_test_predictions = tf.argmax(vit_b16_test_probabilities, axis=1)

# VIT B16 modeli eğitim geçmişinin çizilmesi
plot_training_curves(vit_b16_history)

plot_confusion_matrix(
    test_df.label_encoded,
    vit_b16_test_predictions,
    figsize=(8, 8),
    classes=class_names)

plot_roc(test_df.label_encoded,
         vit_b16_test_probabilities,
         figsize=(10, 10), title_fontsize='large');

# ViT-b16 ROC eğrisi
print(classification_report(test_df.label_encoded,
                            vit_b16_test_predictions,
                            target_names=class_names))

# ViT model performans skoru
vit_b16_performance = generate_preformance_scores(test_df.label_encoded,
                                                 vit_b16_test_predictions,
                                                  vit_b16_test_probabilities)

# DataFrame ile ölçümlerin kaydedilmesi
performance_df = pd.DataFrame({
    'model_cnn': cnn_performance,
    'model_efficientnet_v2': efficientnet_v2_performance,
    'model_vit_b16': vit_b16_performance,
    'model_vgg16' : vgg16_performance,
    'model_densenet' : densenet_performance,
    'model_resnet50' : resnet50_performance,
    'model_facenet' : resnet50_performance,
}).T

# Performans DataFrame Görüntülenmesi
performance_df

# performans ölçütleri grafiği
performance_df.plot(kind="bar", figsize=(14, 8)).legend(bbox_to_anchor=(1.0, 1.0))
plt.title('Performance Metrics', fontsize=20);

def compute_inference_time(model, ds, sample_count, inference_runs=5):
    total_inference_times = []
    inference_rates = []

    for _ in range(inference_runs):
        start = time.perf_counter()
        model.predict(ds)
        end = time.perf_counter()

        # Toplam çıkarım süresinin hesaplanması
        total_inference_time = end - start

        # Çıkarım oranının hesaplanması
        inference_rate = total_inference_time / sample_count

        total_inference_times.append(total_inference_time)
        inference_rates.append(inference_rate)
        # Belirsizlikle birlikte ortalama toplam çıkarım süresinin hesaplanması
    avg_inference_time = sum(total_inference_times) / len(total_inference_times)
    avg_inference_time_uncertainty = (max(total_inference_times) - min(total_inference_times)) / 2

    # Belirsizlikle birlikte ortalama çıkarım oranının hesaplanması
    avg_inference_rate = sum(inference_rates) / len(inference_rates)
    avg_inference_rate_uncertainty = (max(inference_rates) - min(inference_rates)) / 2

    print('====================================================')
    print(f'Model:\t\t{model.name}\n')
    print(f'Inference Time:\t{round(avg_inference_time, 6)}s \xB1 {round(avg_inference_time_uncertainty, 6)}s')
    print(f'Inference Rate:\t{round(avg_inference_rate, 6)}s/sample \xB1 {round(avg_inference_rate_uncertainty, 6)}s/sample')
    print('====================================================')

    return avg_inference_time, avg_inference_rate

cnn_inference = compute_inference_time(model_cnn, test_ds, len(test_df))

efficientnet_v2_inference = compute_inference_time(model_efficientnet_v2, test_ds, len(test_df))

vit_b16_inference = compute_inference_time(model_vit_b16, test_ds, len(test_df))

vgg16_inference = compute_inference_time(model_vgg16, test_ds, len(test_df))

resnet50_inference = compute_inference_time(model_resnet50, test_ds, len(test_df))

densenet_inference = compute_inference_time(model_facenet, test_ds, len(test_df))

facenet_inference = compute_inference_time(model_densenet, test_ds, len(test_df))

# Her model için MCC alınması
cnn_mcc = cnn_performance["matthews_corrcoef"]
efficientnet_mcc = efficientnet_v2_performance["matthews_corrcoef"]
vit_mcc = vit_b16_performance["matthews_corrcoef"]
vgg16_mcc = vgg16_performance["matthews_corrcoef"]
resnet50_mcc = resnet50_performance["matthews_corrcoef"]
densenet_mcc = densenet_performance["matthews_corrcoef"]
facenet_mcc = facenet_performance["matthews_corrcoef"]

# Çıkarım oranının MCC'ye (hareket kontrol grafiğine) karşı dağılım grafiği
plt.figure(figsize=(12, 7))

plt.scatter(cnn_inference[1], cnn_mcc, label=model_cnn.name)
plt.scatter(efficientnet_v2_inference[1], efficientnet_mcc, label=model_efficientnet_v2.name)
plt.scatter(vit_b16_inference[1], vit_mcc, label=model_vit_b16.name)
plt.scatter(vgg16_inference[1], vgg16_mcc, label=model_vgg16.name)
plt.scatter(resnet50_inference[1], resnet50_mcc, label=model_resnet50.name)
plt.scatter(densenet_inference[1], densenet_mcc, label=model_densenet.name)
plt.scatter(facenet_inference[1], facenet_mcc, label=model_facenet.name)

ideal_inference_rate = 0.0001 # İstenen çıkarım süresi
ideal_mcc = 1 # Max MCC

# Her model koordinatını ideal model koordinatlarına bağlayan çizgiler çizilmesi.
plt.scatter(ideal_inference_rate, ideal_mcc, label="Ideal Hypothetical Model", marker='s')
plt.plot([ideal_inference_rate, cnn_inference[1]], [ideal_mcc, cnn_mcc], ':')
plt.plot([ideal_inference_rate, efficientnet_v2_inference[1]], [ideal_mcc, efficientnet_mcc], ':')
plt.plot([ideal_inference_rate, vit_b16_inference[1]], [ideal_mcc, vit_mcc], ':')
plt.plot([ideal_inference_rate, vgg16_inference[1]], [ideal_mcc, vgg16_mcc], ':')
plt.plot([ideal_inference_rate, resnet50_inference[1]], [ideal_mcc, resnet50_mcc], ':')
plt.plot([ideal_inference_rate, densenet_inference[1]], [ideal_mcc, densenet_mcc], ':')
plt.plot([ideal_inference_rate, facenet_inference[1]], [ideal_mcc, facenet_mcc], ':')

plt.legend()
plt.title("Trade-Offs: Inference Rate vs. Matthews Correlation Coefficient", fontsize=20)
plt.xlabel("Inference Rate (s/sample)", fontsize=16)
plt.ylabel("Matthews Correlation Coefficient", fontsize=16);

def dist(x1, x2, y1, y2):
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

model_names = [model_cnn.name, model_efficientnet_v2.name, model_vit_b16.name, model_vgg16.name, model_resnet50.name, model_densenet.name, model_facenet.name]
model_scores = [cnn_mcc, efficientnet_mcc, vit_mcc, vgg16_mcc, resnet50_mcc, densenet_mcc, facenet_mcc]
model_rates = [cnn_inference[1], efficientnet_v2_inference[1], vit_b16_inference[1], vgg16_inference[1], resnet50_inference[1], densenet_inference[1], facenet_inference[1]]
trade_offs = [dist(ideal_inference_rate, inference_rate, ideal_mcc, score)
              for inference_rate, score in zip(model_rates, model_scores)]

print('Trade-Off Score: Inference Rate vs. MCC')
for name, inference_rate, score, trade in zip(model_names, model_rates, model_scores, trade_offs):
    print('---------------------------------------------------------')
    print(f'Model: {name}\n\nInference Rate: {inference_rate:.5f} | MCC: {score:.4f} | Trade-Off: {trade:.4f}')

# En iyi ödünleşme puanına sahip modelin görüntülenmesi
print('=========================================================')
best_model_trade = min(trade_offs)
best_model_name = model_names[np.argmin(trade_offs)]
print(f'\nBest Optimal Model:\t{best_model_name}\nTrade-Off:\t\t{best_model_trade:.4f}\n')
print('=========================================================')
