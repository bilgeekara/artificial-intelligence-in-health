# artificial-intelligence-in-health
1.	Proje Mevcut Durum Değerlendirmesi
Takımımızın hazırlamış olduğu Proje Sunuş Raporunda yapılan literatür taramaları sonucu elde edilen modeller ve kullanılması planlanan ensemble metodun uygulanabilirliğinin olduğu gözlemlenmiştir.Proje raporunda bahsedilen ensembel motodun içindeki modeller değiştirilmiştir. U-Net++, AM-GAN, LinkNet34 ve backbone olarak ImageNet veri setinde önceden eğitilmiş olan ResNet34 ensembel metodu U-Net++ ve backbone olarak ImageNet veri setinde önceden eğitilmiş olan ResNet34 olarak değiştirilmiştir. Başarı oranı beklenilen sonucu vermediği için değiştirilmiştir. Ayrıca eğitim sırasında yapılacak parametre sayısı yaklaşık olarak 150 milyondan 32 milyona kadar düşürülmüştür. Bu yüzden daha iyi ve hızlı sonuç aldığımız bu mimariyi segmentasyon aşamasında kullandık.

Yarışma kapsamında paylaşılan verilerin formatın (dcm) modellerde kullanılabilmesi için png formatına dönüştürme işlemleri yapıldı. Bunun dışında segmentasyon aşamasında kullanılmak için maskeleme işlemi yapıldı. Maskeleme işleminde bizlere verilen Data.xlss dosyasındaki başlangıç ve bitiş kesitleri kullanılmıştır.

Ayrıca sınıflandırma için veriler keras.preprocessing.image.DataFrameIterator sınıfına dönüştürüldü. Bu işlemden sonra VGG16 modelini sınıflandırma işlemleri için oluşturuldu.
Segmentasyon için maskelenmiş veriler ve orijinal fotoğrafların bulunduğu dataframe DataGenerator [8] sınıfında ensemble metot da kullanılabilir hale getirildi. Bu sınıf içinde yapılan işlemler içinde standartlaştırma, normalleştirme ve yeniden şekillendirme adımları bulunmaktadır.

Şekil 1’de proje sunuş raporunda belirtilen ensemble metot, U-Net++ ve U-Net++ ve backbone olarak ImageNet veri setinde önceden eğitilmiş olan ResNet34’nin dice score, zaman ve IoU(Intersection over Union) metriklerine göre karşılaştırılmıştır.
 
Şekil 1
2.	Özgünlük
Dropout: Dropout bir hiper parametredir. Dropout kullanılarak fazladan ihtiyaç duyulmayan nöronlar derin öğrenme modelinden silinir. Kullanılan modellerde nöronun ağırlığı 0.5’den küçük olması durumunda düşürülmüştür. Böylelikle hem model hızlı öğrenebilmekte hem de başarı oranını 2-3 puan yükseltmektedir [9]. Yaptığımız testlerde başarı oranını %3 olumlu etkilemiştir.

Augmantasyon: Veri artırımı (Data Augmentation) ile elimizdeki verileri çevirme, kaydırma, yakınlaştırma gibi tekniklerle genişletilebilmektedir. Bu yöntemlerin dışında verilere noise eklenerek overfitting olayının önüne geçmek hedeflenmektedir. Bu sayede, daha fazla veriye ihtiyaç duyulduğunda emek ve zaman harcayacak olan veri toplama işlemi en aza indirilir. Ön işleme evresinde segmentasyon ve sınıflandırma işlemleri için augmentasyon yapılmıştır.
 
Ensemble metot: Ensemble metot birden fazla modelin birlikte çalışarak daha iyi başarı elde etmeyi amaçlayan bir prensiptir. Bu prensip segmentasyon aşamasında kullanılmıştır. Bu metot sayesinde başarı oranı %4-5 arttırılmıştır. Ensemble metot için U-Net++ ve backbone olarak Resnet34 modelleri kullanılmıştır. Ensemble metodu ilerleyen zamanlarda sınıflandırmada kullanılması planlanmaktadır.

Spatial Pyramid Pooling(SPP) : SPP, derin öğrenme sırasında kullanılan klasik havuzlama teknikleri yerine tercih edilmesi planlanmaktadır. SPP yapısı sayesinde görüntü deformasyonunun önüne geçmektedir. Ayrıca evrişimli sinir ağlarının gerçekleştirdiği tekrarlı hesaplamalardan kaçınarak eğitim ve test için gerekli olan süreyi kullanılan algoritmaya göre değişiklik gösterse de 24-102 kat kısaltmaktadır. Bu yöntemle başarı oranını en az %2 oranında arttırmak mümkündür. Bu yöntemi geliştirilen modellerde klasik pooling aşamaları kullanıldı fakat modellere entegre edebilmek için çalışmalara başlamamıza rağmen modellere entegrasyon sapılamamıştır.

Vision Transformer Kullanımı: Transformer kullanmanın temel amacı kısmen az veri ile eğitilen yapay zeka modelinin başarısını arttırmaktır ve eğitim/test süresini kısaltmaktır. Transformer olarak Vision Transformer (ViT) kullanmak görüntü işlemede performansı pozitif olarak etkilemektedir. Yapılan deneylerde ResNet algoritmasında doğruluk açısından +4.6 ile
+7 puan arttırmaktadır [1]. Bu yüzden tasarlanacak olan algoritmada ViT yöntemin pre-train aşamasında yani backbone kısmında kullanılması planlanmaktadır.

3.	Sonuçlar ve İnceleme
Yarışma kapsamındaki sorun için üretilen çözümde 2 farklı mimari kullanılmıştır. Bir mimari sınıflandırma için diğeri ise segmentasyon içindir. Bu mimarileri kullanmadan önce yapılması gereken ön işleme işlemlerinde zorluklar yaşadık. Genellikle maskelenmiş veri ile orijinal fotoğraf arasındaki katman farkı bizi zorladı. Bunun üstesinden gelebilmek için maskeleme işlemini 3 boyutlu olarak yaptık. Google colob’da çalışırken eğitim yaptığımız sürelerin uzun olmasından dolayı sürekli GPU kullanımı durduruldu. Bu yüzden eğitim ve test yaparken yerel bir GPU kullandık.

3.1	Sınıflandırma
Sınıflandırma işleminde kullanılan mimari VGG16 mimarisidir. VGG16 modelini seçmemizin nedeni medikal görüntüleri sınıflandırma işlemlerinde diğer modellere göre oldukça başarılı olmasıdır. Sağlık bakanlığının verdiği verilere gerekli ön işleme adımları uygulandıktan sonra VGG16 modeli oluşturuldu ve şekil 2’te modelim temel tüm katmanlarını gösterilmiştir. Şekil 3’te temel tüm katmanlara ek olarak flatten, dropout ve 2 katman dense bulunmaktadır.
 

 

Şekil 2
 

 
Şekil 3
Sınıflandırma aşamasındaki başarı ve kayıp oranlarının grafikleri şekil 4’te verilmiştir.
Ayrıca şekil 5’te F1-score ve accurucy metrikleri kullanılarak her bir hastalığın hangi oranlarda doğru sınıflandırıldığı gösterilmiştir. F1-score’a göre bir değerlendirme yapılacak olursa ortalama başarı oranı %93,7’tür. Sınıflandırma yapılırken epoch değeri 10 alınmıştır.
 
Şekil 4
 








Şekil 5
3.2	Segmentasyon
Segmentasyon aşamasında model oluşturulurken ensemble metot prensibi
kullanılmıştır. Kullanılan model U-net++ ve backbone Resnet34’tür. Modelin tüm katmanları ve katmanlarda yapılan işlemler şekil 7’de verilmiştir. Model 30 epoch da eğitilmiştir. Eğitim sırasında modelin doğruluk oranın grafikleri şekil 6’dadır. Genel olarak segmentasyon başarı yüzdesi 90,46’dır.
 
Şekil 6
 

 

 

Şekil 7
 
4.	Deney ve eğitim aşamalarında kullanılan veri setleri
Sağlık bakanlığı tarafından paylaşılan veri setinin dışında herhangi bir veri seti kullanılmamıştır. Sağlık bakanlığı tarafından sınıflandırılmış olarak 38246 adet dicom formatındaki veri sınıflandırılarak verilmiştir. Bu veri setindeki verilerin bazıları yarışma şartnamesi kapsamında sınıflandırma ve segmentasyon işlemleri için uygun değildir. Bu yüzden veri setinden ihtiyaç duyulmayan veriler eğitim ve test için kullanılmamıştır. Veri setindeki veri sayısını ve çeşitliliğini arttırmak amacıyla sınıflandırma aşamasında data augmantasyon kullanılmıştır. Data augmantasyonda işleminde high pass filter ve fotoğrafa kendi merkezinde random bir açı değerinde çevirme işlemleri uygulanarak overfitting olayının gerçekleşmesinin önüne geçilmiştir. Bunun dışında Böbrek taşı, Üreter taşı ve Akut divertikülit ile uyumlu veri sayıları diğer veri tiplerine göre 2-3 kat daha azdır. Bu sebepten ötürü yapılan veri arttırma işleminde bu veri tiplerinde diğerlerinden farklı olarak salt pepper noise eklenmiştir. Şekil 8’de Data.xlss dosyasının içindeki verilerin sayı dağılımı ve bahsedilen yöntemlerle arttırılmış verilerin sayıları (54405 adet) gösterilmiştir. En son elde edilen veri seti sınıflandırma işleminde eğitim ve test aşamasında kullanılmıştır. Ayrıca segmentasyon aşamasında veri arttırmak için sınıflandırmaya benzer bir yöntem kullanılmıştır. Segmentasyon aşamasında yapılan veri arttırma işlemleri sonucu (fotoğrafı kendi ekseninde belirli bir açıda çevirmek) bulunması gereken piksel aralıkları değişebilmektedir. Bu göz önünde bulundurularak maskeleme işlemi yapılmıştır. Sınıflandırma ve segmentasyon işlemlerinde veri seti 0.7,0.15,0.15 oranlarında 3 parçaya ayrılmaktadır. Yüzde 70 eğitim (39,083 adet veri) , yüzde 15 test (8,161 adet veri) ve yüzde 15 validation (8,161 adet veri) için kullanılmıştır.

Şekil 8
 
5.	Referanslar
[1]	Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan , Peizhao Zhang, Zhicheng Yan, Masayoshi, Tomizuka , Joseph Gonzalez, Kurt Keutzer, Peter Vajda,”Visual Transformers: Token-based Image Representation and Processing for Computer Vision” Facebook Reality Labs, Facebook AI, UC Berkeley, 20 Nov 2020.
[2]	Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, ”Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition”, 23 Apr 2015.
[3]	Jian Yin , Zhibo Zhou,Shaohua Xu , Ruiping Yang , and Kun Liu, “A Generative Adversarial Network Fused with Dual-Attention Mechanism and Its Application in Multitarget Image Fine Segmentation”, Hindawi Computational Intelligence and Neuroscience Volume 18 December 2021, Article ID 2464648, 16 pages.
[4]	Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang, “UNet++: A Nested U-Net Architecture for Medical Image Segmentation”,18 Jul 2018.
[5]	Setareh Dabiri , Karteek Popuri , Cydney Ma , Vincent Chow , Elizabeth M. Cespedes Feliciano, Bette J. Caan ,Vickie E. Baracos , Mirza Faisal Beg, “Deep learning method for localization and segmentation of abdominal CT”, 14 August 2020.
[6]	Truong Dang, Tien Thanh Nguyen, John McCall, Eyad Elyan, Carlos Francisco MorenoGarcıa School of Computing Robert Gordon University, Aberdeen, UK, “Two layer Ensemble of Deep Learning Models for Medical Image Segmentation”, 10 Apr 2021.
[7]	Abhishek Chaurasia, Eugenio Culurciello, “LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation”,14 Jun 2017.
[8]	https://www.kaggle.com/code/anantgupt/brain-mri-detection-segmentation-resunet/data
