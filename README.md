
# Yapay Zeka ile Yüz Fotoğraflarının Tespiti

Bu proje, yapay zeka (AI) kullanarak sahte yüz fotoğraflarını tespit etmeyi amaçlayan bir sınıflandırma modelidir. Derin öğrenme yöntemlerini ve önceden eğitilmiş modelleri kullanarak, gerçek ve yapay (AI ile üretilmiş) yüz fotoğraflarını ayırt edebilen bir sistem geliştirilmiştir. Proje, yüz tanıma ve görsel analiz alanlarında önemli bir uygulamaya sahiptir, özellikle sahte yüz fotoğraflarının tanımlanması ve bu tür içeriklerin tespiti için kullanılabilir.

## Proje Hedefi

Projenin amacı, derin öğrenme algoritmalarını kullanarak AI tarafından üretilmiş yüz fotoğraflarını tanımaktır. Bu tür fotoğraflar, genellikle GAN (Generative Adversarial Networks) gibi yöntemler ile oluşturulur ve gerçek insan yüzlerinden ayırt edilmesi zor olabilir. Proje, bu tür sahte yüzleri tespit etmeyi amaçlar.

## Kullanılan Modeller

### 1. **EfficientNet V2**
EfficientNet V2, görüntü sınıflandırma görevlerinde yüksek performans gösteren bir modeldir. Bu projede EfficientNet V2 B0 versiyonu kullanılmıştır. Model, AI tarafından üretilmiş yüzleri tespit etmek için görüntülerden özellik çıkarımı yapar ve sınıflandırma görevini yerine getirir.

### 2. **VGG16**
VGG16, derin sinir ağı mimarisine sahip bir modeldir ve özellikle küçük nesnelerin tanınmasında iyi sonuçlar verir. VGG16, AI yüz üretiminde kullanılan belirli desenleri tespit etmek için kullanılmıştır.

### 3. **DenseNet121**
DenseNet121, katmanlar arasında daha yoğun bağlantılar kullanarak daha verimli öğrenme sağlar. Bu model, yüz fotoğraflarındaki detaylı özellikleri daha iyi anlayabilmek için kullanılmıştır.

### 4. **ResNet50**
ResNet50, residual bağlantılar kullanarak daha derin ağların öğrenmesini kolaylaştırır. Bu model, derinlemesine analizler yaparak AI yüzlerinin özelliklerini tanıyacak şekilde eğitilmiştir.

### 5. **FaceNet (InceptionResNetV2)**
FaceNet, özellikle yüz tanıma için geliştirilmiş bir modeldir. Projede, InceptionResNetV2'nin önceden eğitilmiş ağırlıkları kullanılmıştır. Yüz tanımayı doğru yapabilmek için bu model, gerçek ve sahte yüzleri ayırt etmekte kullanılmıştır.

### 6. **CNN (Convolutional Neural Network)**
CNN, derin öğrenme yöntemlerinden biridir ve görüntü verisiyle ilgili görevlerde oldukça etkilidir. Bu projede, CNN mimarisi, yüz fotoğraflarındaki özellikleri öğrenerek sahte yüzleri tanımak için kullanılmıştır.

### 7. **ViT B16 (Vision Transformer B16)**
ViT B16, görüntü sınıflandırma için kullanılan bir model olup, Transformer mimarisini temel alır. Bu model, özellikle uzun menzilli özellikleri öğrenmede etkili olup, AI ile üretilmiş yüzleri tespit etmek için kullanılmıştır.

## Kullanılan Kütüphaneler

- TensorFlow
- Keras
- Matplotlib
- Scikit-learn
- NumPy
- OpenCV (yüz tanıma ve görüntü işleme için)

## Veri Seti

Proje, AI tarafından üretilmiş ve gerçek yüz fotoğraflarını içeren bir veri seti kullanır. Veri seti, çeşitli yüz tanıma algoritmalarını test etmek için gerçek insan yüzleri ve GAN (Generative Adversarial Networks) ile üretilmiş yapay yüzlerden oluşur. Veri seti, görsel çeşitlilik açısından zengin olup, farklı ışık koşulları, yaş, cinsiyet ve etnik kökeni kapsar.

## Eğitim ve Test Süreci

### Eğitim Adımları
1. **Veri Ön İşleme:** Veri, normalleştirme ve yeniden boyutlandırma işlemlerinden geçirilmiştir. Bu işlemler, modelin daha verimli öğrenmesine yardımcı olur.
2. **Model Eğitimi:** Her bir model, ImageNet gibi büyük veri setleri üzerinde önceden eğitilmiş ağırlıklara sahip olup, daha sonra AI yüz fotoğraflarını tespit etmek amacıyla son katmanları eğitilmiştir.
3. **Transfer Öğrenme:** Modellerin eğitiminde transfer öğrenme kullanılarak, önceden eğitilmiş modellerin ağırlıkları ile modelin başarısı artırılmıştır.

### Test Adımları
Test aşamasında, modelin doğru sınıflandırma yapabilme yeteneği, çeşitli metriklerle değerlendirilmiştir:
- **Doğruluk (Accuracy)**
- **ROC Eğrisi**
- **Karışıklık Matrisi**
- **Sınıflandırma Raporu**
