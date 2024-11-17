## Project Overview:
Pada era digital ini, perkembangan industri kuliner semakin pesat, khususnya di Indonesia, di mana kuliner menjadi bagian penting dari budaya dan kehidupan sehari-hari. Seiring bertambahnya variasi resep dan bahan, konsumen sering kali kesulitan menemukan resep yang sesuai dengan preferensi atau bahan yang mereka miliki (Dewi et al., 2022). Hal ini menimbulkan kebutuhan untuk menciptakan sistem rekomendasi resep yang dapat memberikan saran yang relevan dan sesuai dengan kebutuhan pengguna.

Sistem Rekomendasi Resep Makanan ini dibangun menggunakan pendekatan Content-Based Filtering yang memanfaatkan informasi dari resep-resep sebelumnya untuk menyesuaikan preferensi pengguna. Selain itu, dengan kombinasi pendekatan Collaborative Filtering, sistem ini dapat menghasilkan rekomendasi hybrid yang memperhitungkan opini pengguna lain yang memiliki preferensi serupa melalui review dari setiap resep. Kombinasi ini bertujuan untuk memberikan rekomendasi yang tidak hanya relevan secara konten, tetapi juga sesuai dengan preferensi kolektif dari pengguna.

Sistem ini memiliki potensi besar dalam mempermudah pengguna dalam menemukan resep yang diinginkan dengan mempertimbangkan beberapa opsi filter, seperti kategori masakan, bahan-bahan yang tersedia, atau masakan populer. Proyek ini tidak hanya penting untuk memberikan kemudahan bagi pengguna, tetapi juga membuka peluang inovasi dalam pengembangan aplikasi kuliner di Indonesia.

## Business Understanding:
Dalam membangun sistem rekomendasi ini, penting untuk memahami tantangan dan peluang yang dapat dihadapi. Bagian ini mendefinisikan masalah utama, tujuan, dan pendekatan solusi yang digunakan.

#### Problem Statements
1. Bagaimana sistem dapat memberikan rekomendasi resep yang relevan berdasarkan bahan-bahan yang tersedia, kategori masakan, atau preferensi pengguna lainnya?
2. Bagaimana meningkatkan akurasi rekomendasi dengan memanfaatkan pendapat dan preferensi pengguna lain?

#### Goals
1. Mengembangkan sistem rekomendasi yang dapat menyarankan resep yang sesuai berdasarkan bahan yang ada, kategori tertentu, atau preferensi pribadi.
2. Menggabungkan pendekatan Content-Based dan Collaborative Filtering untuk meningkatkan kualitas rekomendasi dengan mempertimbangkan preferensi kolektif pengguna.

#### Solution Statement
1. Content-Based Filtering: Pendekatan ini digunakan untuk merekomendasikan resep berdasarkan karakteristik resep yang serupa dengan resep yang disukai oleh pengguna.
2. Collaborative Filtering: Dengan memperhitungkan preferensi pengguna lain yang memiliki kesamaan, sistem akan memberikan rekomendasi berdasarkan kecenderungan pengguna kolektif.



## Data Understanding
Metadata yang digunakan pada proyek sistem rekomendasi resep makanan diambil dari situs web Food.com lalu disimpan pada kaggle berikut ini:
https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews

Terdapat dua dataset yang disediakan yaitu `resep` dan `ulasan`. `resep` berisikan informasi detail setiap resep makanan seperti waktu, bahan-bahan, nutrisi, kalori, tahapan, dan lain-lain. Data mengenai resep ini mengandung 522,517 baris data dan memiliki 312 kategori resep makanan yang berebeda.

Sedangkan `ulasan` berisikan informasi ulasan pelanggan dalam resep makanan yang mereka pilih seperti nama pembua resep, rating, teks_review, dan lain-lain.

Berikut merupakan variabel-variabel yang digunakan:

#### DataFrame `resep`

| Kolom                       | Deskripsi                                                                                     |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| `RecipeId`                  | ID unik untuk setiap resep.                                                                  |
| `Name`                      | Nama atau judul resep.                                                                       |
| `AuthorId`                  | ID unik dari penulis atau pembuat resep.                                                     |
| `AuthorName`                | Nama dari penulis atau pembuat resep.                                                        |
| `CookTime`                  | Waktu yang dibutuhkan untuk memasak resep.                                                   |
| `PrepTime`                  | Waktu yang dibutuhkan untuk persiapan resep.                                                 |
| `TotalTime`                 | Total waktu yang dibutuhkan untuk resep (persiapan + memasak).                               |
| `DatePublished`             | Tanggal resep dipublikasikan.                                                                |
| `Description`               | Deskripsi singkat tentang resep.                                                             |
| `Images`                    | Link atau path gambar terkait resep.                                                         |
| `RecipeCategory`            | Kategori atau jenis makanan resep tersebut.                                                  |
| `Keywords`                  | Kata kunci yang terkait dengan resep untuk memudahkan pencarian.                             |
| `RecipeIngredientQuantities`| Jumlah atau kuantitas bahan yang diperlukan dalam resep.                                     |
| `RecipeIngredientParts`     | Daftar bahan-bahan yang diperlukan dalam resep.                                              |
| `AggregatedRating`          | Rating rata-rata dari pengguna untuk resep ini.                                              |
| `ReviewCount`               | Jumlah ulasan yang diterima resep ini.                                                       |
| `Calories`                  | Jumlah kalori dalam resep ini.                                                               |
| `FatContent`                | Kandungan lemak dalam resep (dalam gram).                                                    |
| `SaturatedFatContent`       | Kandungan lemak jenuh dalam resep (dalam gram).                                              |
| `CholesterolContent`        | Kandungan kolesterol dalam resep (dalam miligram).                                           |
| `SodiumContent`             | Kandungan natrium dalam resep (dalam miligram).                                              |
| `CarbohydrateContent`       | Kandungan karbohidrat dalam resep (dalam gram).                                              |
| `FiberContent`              | Kandungan serat dalam resep (dalam gram).                                                    |
| `SugarContent`              | Kandungan gula dalam resep (dalam gram).                                                     |
| `ProteinContent`            | Kandungan protein dalam resep (dalam gram).                                                  |
| `RecipeServings`            | Jumlah porsi yang dihasilkan oleh resep ini.                                                 |
| `RecipeYield`               | Jumlah hasil akhir resep, biasanya berupa kuantitas atau ukuran tertentu.                    |
| `RecipeInstructions`        | Langkah-langkah atau instruksi untuk membuat resep.                                          |

#### DataFrame `ulasan`

| Kolom            | Deskripsi                                                                                               |
|------------------|---------------------------------------------------------------------------------------------------------|
| `ReviewId`       | ID unik untuk setiap ulasan.                                                                            |
| `RecipeId`       | ID unik resep yang diulas.                                                                              |
| `AuthorId`       | ID unik dari penulis ulasan.                                                                            |
| `AuthorName`     | Nama dari penulis ulasan.                                                                               |
| `Rating`         | Rating yang diberikan oleh penulis ulasan untuk resep ini.                                              |
| `Review`         | Teks ulasan dari penulis mengenai resep.                                                                |
| `DateSubmitted`  | Tanggal ketika ulasan disubmit.                                                                         |
| `DateModified`   | Tanggal terakhir ulasan diubah atau diperbarui.                                                         |


#### Hasil Analisis Melalui Exploratory Data Analysis
Waktu memasak kebanyakan ada di sekitar 10 menit sampai 30 menit, sedangkan kalori sepertinya persebarannya terlalu luas, untuk rating rata-rata pengguna memberikan nilai 4 sampai 5. Penulis `Sydney Mike` memiliki jumlah ulasan yang paling banyak menunjukkan bahwa beliau seorang yang popular baik dari segi baik atau hal lainnya.

Berdasarkan visualisasi yang dibuat menggunakan data resep dan ulasan ditemukan bahwa persebaran data point untuk resep yang memiliki kalori banyak itu rata-rata dimasak pada rentang 10 sampai 30 menit. Selain itu kategori yang memiliki rata-rata jumlah kalori paling banyak yaitu Guatemalan, Buttermilk Biscuits, dan Labor Day. Artinya makanan berlemak yang merupakan tentu saja makanan berat.


## Data Preparation

1. Handling Missing Value dengan string ""
Membuat fungsi Impute Missing Value terhadap suatu variabel sebagai input supaya pada tahap model itu bisa dibaca datanya bukannya dalam format yang tidak diketahui sehingga menghasilkan error.

2. Data Cleaning Untuk Content Based Filtering

- Menghapus kolom waktu yang tidak berkaitan dengan sistem rekomendasi seperti `DateSubmitted`, `DateModified`, dll.

- Menghapus string "PT" pada variabel waktu yang berkaitan dengan sistem rekomendasi.
Supaya variabel waktu atau lama masak resep tersebut bisa digunakan untuk rekomendasi yang menjelaskan tentang waktu seperti "H" untuk Hour.

- Membuat fungsi untuk `reformatRecipe` variabel `RecipeInstructions`.
Karena data pada variabel tersebut sepertinya dalam bentuk list tuple dan perlu diubah menjadi format string.

- Membuat fungsi `reformatKolom`
Hal ini dilakukan karena terdapat beberapa kolom seperti `Images`, `Keywords`, dll yang formatnya masih berantakan. Seperti ada yang memiliki tanda kurung tidak diperlukan, penggunaan huruf "c", dan adanya teks unik seperti "/" dan "\n", selain itu NA diubah menjadi format string. Supaya model CBF dapat menerima string teks tersebut.

- Menggunakan 10% dari metadata
Hal ini dilakkuan karena adanya keterbatasan sumber daya dan waktu, 10% data dipilih berdasarkan popularitas melalui jumlah rating dari seluruh resep.

- Menghapus baris yang missing value
Hal ini hanya dilakukan pada variabel yang akan digunakan pada modeling dengan Content Based Filtering.

- Menyiapkan variabel `Resep` untuk digunakan pada model CBF
Hal ini dilakukan dengan menggabungkan beberapa variabel seperti kata kunci resep, bahan-bahan, kategori resep, dan lama resep dimasak. Khusus untuk bahan-bahan resep itu diubah dan diurutkan menjadi string terlebih dahulu dari yang awalnya sebuah list.

3. Data Cleaning Untuk Collaborative Filtering
- Menyiapkan data untuk Collaborative Filtering
Membuat dataframe baru dengan menggabungkan data ulasan dan resep berdasarkan RecipeId, mengambil kolom penting seperti RecipeCategory, Rating, AuthorId, dan RecipeId, lalu menangani nilai kosong pada Rating dengan mengisinya menggunakan rata-rata rating.

- Encoding nominal
melakukan encoding pada kolom AuthorId, RecipeId, dan RecipeCategory menggunakan LabelEncoder untuk mengonversi nilai kategori menjadi numerik dan nominal.

- Normalisasi `Rating`
nilai Rating dinormalisasi ke rentang 0-1 berdasarkan nilai minimum dan maksimum dalam dataset. data fitur (X2) terdiri dari kolom yang telah di-encode, dan data target (y2) adalah rating yang sudah dinormalisasi.

- Splitting dataset
Split atau membagi dataset dengan bobot 80:20 untuk data train dan data validasi.


## Modeling and Result

### A. Content Based Filtering
Membuat sistem rekomendasi untuk merekomendasikan bahan-bahan terhadap suatu resep makanan dengan memperhatikan data berupa teks yaitu bahan makanan, kategori resep, dan lamanya resep tersebut dimasak.

Tahapannya:
1. Count Vectorizer dengan TF-IDF
Teknik TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk merepresentasikan resep sebagai vektor numerik berdasarkan kata-kata yang muncul dalam deskripsinya.
- Term Frequency (TF): Mengukur seberapa sering suatu kata muncul dalam resep tertentu.
- Inverse Document Frequency (IDF): Memberi bobot lebih besar pada kata-kata unik yang tidak sering muncul di seluruh resep.
- Parameter `ngram_range=(1, 2)` memungkinkan analisis kata tunggal (unigrams) dan pasangan kata (bigrams).
- Kata-kata umum dalam bahasa Inggris dihapus dengan stop_words='english'.

2. Cosine Similarity
Menggunakan Cosine Similarity untuk mengukur tingkat kemiripan antar resep berdasarkan vektor TF-IDF. Nilai kemiripan dihitung berdasarkan sudut antar-vektor:
- Nilai mendekati 1 berarti kedua resep sangat mirip.
- Nilai mendekati 0 berarti keduanya tidak mirip sama sekali.

3. Mapping Nama Resep ke Indeks:
Membuat indeks berdasarkan nama resep untuk mempermudah pencarian resep tertentu.

4. Membangun Fungsi Rekomendasi:
Fungsi ini menghasilkan daftar rekomendasi berdasarkan:
- Input resep: Resep yang dimasukkan pengguna.
- Kemiripan kosinus: Dihitung dengan skor kemiripan untuk semua resep lainnya.

Langkah-langkah dalam fungsi:
- Mencari indeks dari resep input.
- Mengurutkan skor kemiripan dari tertinggi ke terendah (kecuali resep itu sendiri).
- Mengambil nama resep, bahan, dan ID dari rekomendasi teratas.
- Mengembalikan hasil dalam bentuk DataFrame.

5. Menghasilkan Output:
Memberikan 8 resep yang paling mirip dengan "Banana Bread" berdasarkan bahan atau deskripsi yang ada.


### B. Collaborative Filtering Menggunakan Neural Network
Membuat model berbasis neural network yang mampu mempelajari interaksi antar entitas melalui neural network dan memprediksi rating untuk resep yang belum dicoba oleh pengguna. Dengan pendekatan ini, model dapat memberikan rekomendasi yang lebih kompleks dan mempertimbangkan berbagai faktor selain sekadar interaksi pengguna-item.


#### Tahapan Dalam Membuat Model Berbasis Jaringan Syaraf
1. Membuat class Model `RecommenderNet`:
- num_users: Jumlah pengguna yang unik dalam dataset (total AuthorId).
- num_recipes: Jumlah resep yang unik dalam dataset (total RecipeId).
- num_categories: Jumlah kategori resep yang unik.
- embedding_size: Ukuran dimensi vektor embedding untuk setiap entitas (pengguna, resep, kategori).

2. Layer Embedding:
Dilakukan karena setiap data yang digunakan itu merupakan data nominal dan perlu untuk di kategorikan khususnya variabel `RecipeCategory` yang memiliki 200+ kategori resep makanan.
- num_users, num_recipes, dan num_categories adalah jumlah kategori unik.
- embedding_size: Dimensi dari representasi embedding.
- embeddings_initializer="he_normal": Teknik inisialisasi bobot untuk menghindari vanishing gradient.
- embeddings_regularizer=tf.keras.regularizers.l2(1e-5): Regularisasi L2 untuk mencegah overfitting pada embedding.

3. Menghitung Skor Prediksi dengan dua komponen:
- Dot product antara user dan resep: Mengukur interaksi antara pengguna dan resep berdasarkan embedding mereka.
- Dot product antara resep dan kategori: Mengukur relevansi antara resep dan kategori yang terkait.

Sehingga outputnya didapat dari fungsi aktivasi sigmoid yang digunakan untuk mengubah skor prediksi menjadi rentang antara 0 dan 1 (seperti rating).

4. Menyiapkan Optimizer dan Model
- Optimizer: ExponentialDecay digunakan untuk mengurangi laju pembelajaran secara eksponensial seiring berjalannya waktu. Ini membantu model untuk belajar dengan cepat di awal dan stabil di akhir pelatihan.
- Loss Function: MeanSquaredError digunakan untuk mengukur perbedaan antara rating yang diprediksi dengan rating yang sebenarnya. Model berusaha untuk meminimalkan kesalahan ini.
- Metrics: RootMeanSquaredError (RMSE) dan MeanAbsoluteError (MAE) digunakan untuk mengevaluasi kinerja model.

5. Melatih Model
- Pelatihan: Model dilatih menggunakan data pelatihan (X_train, y_train) selama 10 epoch dengan batch size 128.
- Validation: X_val dan y_val digunakan untuk memverifikasi kinerja model pada data yang tidak terlihat selama pelatihan.

6. Membuat Fungsi Rekomendasi Resep
- Menyaring resep yang sudah dinilai oleh pengguna.
- Menghitung rating yang diprediksi untuk resep yang belum dinilai oleh pengguna.
- Menyortir resep berdasarkan rating yang diprediksi dan memilih top_n resep teratas.

7. Mendapatkan Rekomendasi
- Menghasilkan 5 resep teratas yang direkomendasikan untuk pengguna secara acak (random_author_id).

#### Kelebihan Collaborative Filtering dengan Neural Network
1. Daya Prediksi Lebih Baik:
Dapat memprediksi rating secara langsung dengan memperhitungkan banyak faktor, bukan hanya interaksi antara pengguna dan item.

2. Kemampuan Menggunakan Fitur Tambahan:
Dengan menggunakan embedding untuk kategori dan resep, model ini dapat memanfaatkan fitur tambahan selain hanya data rating pengguna. Ini memungkinkan model untuk lebih memahami keterkaitan antar item (resep) dan kategori.

3. Regularisasi:
Regularisasi L2 pada embedding membantu mencegah overfitting, membuat model lebih robust.


## Evaluation

#### A. Content Based Filtering
Terdapat beberapa parameter evaluasi yang akan diuji untuk melihat performa dari model rekomendasi Content Based Filtering sebelumnya yaitu:
1. Precision @ K
2. Recall @ K
3. Average Precision @ K
4. Mean Average Precision @ K

Untuk mengevaluasi sistem rekomendasi CBF tersebut saya menggunakan tiga metrik untuk mengukur kinerja model, yaitu Precision@K, Recall@K, dan Average Precision@K (AP@K). Di samping itu, ada Mean Average Precision@K (MAP@K) yang menggabungkan hasil evaluasi dari semua resep untuk menghasilkan skor keseluruhan.

1. Precision@K mengukur seberapa banyak item yang relevan ada di dalam K rekomendasi teratas yang diberikan oleh model.

Precision@K = Jumlah item relevan dalam top K rekomendasi / K

Precision@K memberi tahu kita berapa banyak dari K rekomendasi yang benar-benar relevan bagi pengguna. Misalnya, jika 5 dari 40 rekomendasi teratas relevan, precision akan menjadi 0.125 (5 relevan / 40 rekomendasi).

2. Recall@K mengukur seberapa banyak item relevan yang ada di dalam K rekomendasi teratas dibandingkan dengan seluruh item relevan yang tersedia.

Recall@K = Jumlah item relevan dalam top K rekomendasi / Jumlah total item relevan

Recall@K menunjukkan kemampuan model untuk menangkap semua item relevan dalam daftar rekomendasi teratas. Jika terdapat banyak item relevan tapi model hanya berhasil merekomendasikan sedikit dari mereka, recall akan rendah.

3. Average Precision@K (AP@K) adalah rata-rata precision pada setiap level rekomendasi dalam K teratas.

AP@K = (1 / |relevan|) . i=1 ∑ k Precision@i x I(rekomendasi i . ∈ relevan)

AP@K memberikan skor rata-rata yang lebih memperhatikan kualitas posisi dalam daftar rekomendasi. AP@K lebih memberi nilai pada posisi item relevan yang lebih tinggi dalam daftar rekomendasi.

4. Mean Average Precision@K (MAP@K) adalah rata-rata dari AP@K untuk semua pengguna, yang memberikan gambaran umum kinerja model untuk seluruh dataset.

MAP@K = (1/N) u=1 ∑ N AP@K u

MAP@K memberikan gambaran kinerja model pada skala yang lebih luas, dengan menghitung rata-rata dari skor AP@K untuk setiap pengguna. Nilai ini mencerminkan seberapa baik model dalam memberikan rekomendasi yang relevan di seluruh dataset.

Hasil evaluasi menunjukkan bahwa Precision@40 dan Recall@40 bervariasi untuk masing-masing resep yang dievaluasi. Sebagai contoh, untuk "Banana Bread", precision sebesar 0.025 dan recall sebesar 0.111 menunjukkan bahwa hanya sebagian kecil dari rekomendasi yang relevan dengan total item relevan yang ada. Namun, Average Precision lebih tinggi di "Sweet Potato Casserole" (AP@40 = 0.292), yang menunjukkan bahwa meskipun jumlah item relevan tidak banyak, posisi item relevan dalam rekomendasi cukup baik.

MAP@40 untuk semua resep di dataset adalah 0.0395, yang merupakan nilai rata-rata dari AP@K untuk setiap resep yang diuji. Ini menunjukkan bahwa meskipun ada beberapa rekomendasi yang sangat baik, banyak rekomendasi yang kurang relevan untuk banyak resep, yang menandakan bahwa model dapat ditingkatkan.


#### B. Collaborative Filtering

1. **Mean Squared Error (MSE)**  
   MSE mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual (rating yang sebenarnya). MSE memberi penalti yang lebih besar pada kesalahan yang lebih besar, karena kesalahan dihitung dalam bentuk kuadrat.  

   \[
   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   \]

   di mana 𝑦𝑖 adalah nilai aktual dan 𝑦^𝑖 adalah prediksi model.

2. Root Mean Squared Error (RMSE) adalah akar kuadrat dari MSE, memberikan nilai yang lebih mudah diinterpretasikan karena berada pada satuan yang sama dengan data asli (misalnya, rating).

   \[
   \text{RMSE} = \sqrt{\text{MSE}}
   \]

RMSE memberi gambaran tentang seberapa besar kesalahan model dalam skala yang sama dengan data asli. Nilai RMSE yang lebih rendah menunjukkan prediksi yang lebih akurat.

3. Mean Absolute Error (MAE) mengukur rata-rata absolut dari selisih antara nilai prediksi dan nilai aktual. Berbeda dengan MSE, MAE tidak memberikan penalti lebih besar pada kesalahan besar dan lebih mudah dipahami secara intuitif.

   \[
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   \]

MAE memberikan gambaran tentang seberapa jauh prediksi model dari nilai sebenarnya, tanpa memperbesar kesalahan yang lebih besar seperti pada MSE.


Dari hasil pelatihan model yang ditampilkan, kita bisa melihat beberapa metrik yang terukur pada setiap epoch, baik pada data pelatihan (train) maupun data validasi (validation):
- Loss (MSE) berkurang secara keseluruhan, yang menunjukkan bahwa model semakin baik dalam meminimalkan kesalahan prediksi antara prediksi dan nilai aktual.
- Root Mean Squared Error (RMSE) menunjukkan penurunan dari 0.3158 pada epoch pertama menjadi 0.2803 pada epoch terakhir, yang menandakan bahwa kesalahan prediksi semakin kecil seiring berjalannya waktu.
- Mean Absolute Error (MAE) juga mengalami penurunan yang menunjukkan perbaikan akurasi prediksi, dengan nilai berkurang dari 0.1486 pada epoch pertama menjadi 0.1202 pada epoch terakhir.


Namun, meskipun ada penurunan yang konsisten pada MSE, RMSE, dan MAE, nilai val_loss, val_rmse, dan val_mae relatif stabil, yang menunjukkan bahwa model mulai mengalami overfitting setelah beberapa epoch. Hal ini terlihat pada epoch ke-7 di mana val_loss mulai meningkat meskipun metrik pelatihan terus menurun.

### Conclusion & Result
1. Content-Based Filtering menunjukkan keterbatasan besar dalam relevansi rekomendasi (MAP@40 = 0.0395) meskipun ada beberapa rekomendasi yang baik. Precision dan Recall yang rendah pada sebagian besar resep mengindikasikan bahwa model ini perlu diperbaiki lebih lanjut untuk meningkatkan kualitas rekomendasi secara keseluruhan.

2. Collaborative Filtering, meskipun menunjukkan penurunan yang baik pada MSE, RMSE, dan MAE, juga menunjukkan tanda-tanda overfitting pada data validasi. Hal ini menandakan bahwa meskipun model ini lebih baik dalam hal prediksi akurasi, ia mungkin tidak cukup generalizable tanpa tambahan penyesuaian.

3. Dengan mempertimbangkan penurunan kesalahan prediksi yang lebih signifikan pada Collaborative Filtering, dan meskipun terdapat indikasi overfitting pada data validasi, model CF tetap menunjukkan hasil yang lebih baik dari sisi akurasi prediksi dibandingkan dengan model CBF yang memiliki MAP yang sangat rendah.
- RMSE turun dari 0.3158 pada epoch pertama menjadi 0.2803 pada epoch terakhir.
- MAE berkurang dari 0.1486 menjadi 0.1202.

Namun tentu harus dilakukan perbaikan terhadap Collaborative Filtering dengan penyesuaian parameter untuk mengurangi overfitting, seperti penerapan regularisasi lebih kuat atau early stopping, serta fine-tuning untuk memperbaiki generalizability dan mengoptimalkan performa pada data validasi.


### References:
Dewi, I. C., Indrianto, A. T. L., Soediro, M., Winarno, P. S., Minantyo, H., Sondak, M. R., & Leoparjo, F. (2022). Trend bisnis food and beverages menuju 2030.