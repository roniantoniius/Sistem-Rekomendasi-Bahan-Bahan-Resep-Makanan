from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from collections.abc import Iterable
from pyforest import *
import re # regex
import warnings
warnings.filterwarnings("ignore")

resep =  pd.read_csv('/kaggle/input/foodcom-recipes-and-reviews/recipes.csv')
ulasan = pd.read_csv('/kaggle/input/foodcom-recipes-and-reviews/reviews.csv')

def data_summary(df, df_name="DataFrame"):
    """
    Menampilkan ringkasan informasi dan kondisi dari DataFrame, termasuk jumlah data, bentuk, 
    informasi kolom, nilai deskriptif, nilai kosong, dan nilai unik.
    
    Args:
        df (pd.DataFrame): DataFrame yang akan dianalisis.
        df_name (str): Nama dari DataFrame yang akan ditampilkan dalam output. Default adalah "DataFrame".
    
    Returns:
        None
    """
    
    print(f"\n{'='*40}")
    print(f"Ringkasan untuk {df_name}")
    print(f"{'='*40}\n")
    
    print("Jumlah data:")
    print(df.shape[0], "baris,", df.shape[1], "kolom\n")
    
    print("Informasi kolom:")
    print(df.info(), "\n")
    
    print("Deskripsi Statistik:")
    print(df.describe().T, "\n")
    
    print("Jumlah nilai kosong (NaN) per kolom:")
    print(df.isnull().sum(), "\n")
    
    print("Jumlah nilai unik per kolom:")
    unique_counts = df.nunique()
    print(unique_counts, "\n")

data_summary(resep, "Resep")

data_summary(ulasan, "Ulasan")

resep = resep[['RecipeId', 'Name', 'AuthorId', 'AuthorName', 
                   'CookTime', 'PrepTime', 'TotalTime', 'Description', 'Images', 
                   'RecipeCategory', 'Keywords', 'RecipeIngredientQuantities', 'RecipeIngredientParts', 
                   'AggregatedRating', 'ReviewCount', 'Calories', 'FatContent',
                   'SaturatedFatContent', 'CholesterolContent', 'SodiumContent','CarbohydrateContent', 
                   'FiberContent', 'SugarContent', 'ProteinContent','RecipeServings', 'RecipeInstructions']]


def hitung_null(df, kolom):
    return df[kolom].isnull().sum()

def ganti_nan(df, kolom):
    df[kolom] = df[kolom].fillna('NA')
    return df

ganti_nan(resep, kolom='CookTime')

ulasan = ulasan[['ReviewId','RecipeId','AuthorId','AuthorName','Rating','Review']]

ganti_nan(ulasan, kolom='Review')

resep.CookTime = resep.CookTime.str.replace('PT', '')
resep.PrepTime = resep.PrepTime.str.replace('PT', '')
resep.TotalTime = resep.TotalTime.str.replace('PT', '')

def reformatRecipe(recipe_series):
    return recipe_series.apply(lambda i: i.replace("\n", "")
                                        .replace('c("', '')
                                        .replace('")', '')
                                        .replace('". "', '. '))

resep['RecipeInstructions'] = reformatRecipe(resep['RecipeInstructions'])

def reformatKolom(variabel):
    def proses_item(i):
        if isinstance(i, str):
            i = i.replace("NA", '"NA"')
            i = i.replace("character(0)", 'c("character(0)")')
            i = i.replace("\n", "")
            if i.startswith('"http'):
                return [[i[1:-1]]]
            else:
                return i[3:-2].split('", "')
        else:
            return []
    
    return variabel.apply(proses_item)

resep['Images'] = reformatKolom(resep['Images'])
resep['Keywords'] = reformatKolom(resep['Keywords'])
resep['RecipeIngredientParts'] = reformatKolom(resep['RecipeIngredientParts'])
resep['RecipeIngredientQuantities'] = reformatKolom(resep['RecipeIngredientQuantities'])

jumlahR = ulasan.shape[0]
jumlahA = len(np.unique(ulasan.AuthorId))
totalResep = len(np.unique(ulasan.RecipeId))

print(f"Total jumlah rating ada {jumlahR}, Total jumlah penulis resep yaitu {jumlahA}, dan Total jumlah resep adalah {totalResep}")

sepuluh = round(0.1*len(np.unique(ulasan.RecipeId)))

jumlahRating = ulasan.groupby('RecipeId')['Rating'].count()

resep1 = resep.iloc[:sepuluh]

resepPopuler = resep1.merge(jumlahRating, on='RecipeId')


cbf_populer = resepPopuler
cbf_populer.dropna(subset=['Name',
                           'RecipeId',
                           'Description',
                           'Keywords'], inplace=True)

def ubah_list(kolom):
    return f" {' '.join(kolom)} "

cbf_populer.Keywords = cbf_populer.Keywords.apply(ubah_list)
cbf_populer = cbf_populer.reset_index()

cbf_rekomendasi1 = cbf_populer.copy()
cbf_rekomendasi1['Resep'] = cbf_rekomendasi1['RecipeCategory']
cbf_rekomendasi1['Resep'] = cbf_rekomendasi1['Resep'].fillna('')

# Untuk bahan kita ubah jadi string
cbf_rekomendasi1['Bahan-Bahan'] = cbf_rekomendasi1['RecipeIngredientParts'].apply(ubah_list)

cbf_rekomendasi1['Resep'] = cbf_rekomendasi1['Resep'] + cbf_rekomendasi1['Bahan-Bahan']

cbf_rekomendasi1['Resep'] = cbf_rekomendasi1['Resep'] + cbf_rekomendasi1['TotalTime']

cbf_rekomendasi1['Resep'] = cbf_rekomendasi1['Resep'].fillna('')

data2 = ulasan[ulasan['RecipeId'].isin(resep1['RecipeId'])]
data2 = data2.merge(resep1[['RecipeId', 'RecipeCategory']], on='RecipeId')

df2 = data2[['RecipeCategory', 'Rating', 'AuthorId', 'RecipeId']]

df2['Rating'] = df2['Rating'].fillna(df2['Rating'].mean())

author_encoder = LabelEncoder()
df2['AuthorId_encoded'] = author_encoder.fit_transform(df2['AuthorId'])

recipe_encoder = LabelEncoder()
df2['RecipeId_encoded'] = recipe_encoder.fit_transform(df2['RecipeId'])

category_encoder = LabelEncoder()
df2['RecipeCategory_encoded'] = category_encoder.fit_transform(df2['RecipeCategory'])

min_rating = df2['Rating'].min()
max_rating = df2['Rating'].max()

df2['NormalizedRating'] = (df2['Rating'] - min_rating) / (max_rating - min_rating)

X2 = df2[['AuthorId_encoded', 'RecipeId_encoded', 'RecipeCategory_encoded']].values
y2 = df2['NormalizedRating'].values

X_train, X_val, y_train, y_val = train_test_split(X2, y2, test_size=0.2, random_state=42)

model_tfidf = TfidfVectorizer(analyzer='word',
                              ngram_range=(1, 2),
                              min_df=0,
                              stop_words='english')

tfidf = model_tfidf.fit_transform(cbf_rekomendasi1['Resep'])

cosine_resep = linear_kernel(tfidf, tfidf)

cbf_resep = cbf_rekomendasi1.reset_index()
nama = cbf_resep['Name'] 
indice = pd.Series(cbf_resep.index, index=cbf_resep['Name'])

def rekomendasi_cbf_resep_dan_bahan(resep, n=10):
    """
    Menghasilkan rekomendasi resep dan bahan-bahan berdasarkan cosine similarity.

    Args:
        title (str): Nama resep untuk mencari rekomendasi.
        n (int): Jumlah rekomendasi yang diinginkan (default 10 resep).

    Returns:
        pd.DataFrame: DataFrame berisi resep yang direkomendasikan, nama, dan bahan-bahannya.
    """
    # cari indeks berdasarkan nama resep
    idx = indice[resep]

    # kalau indeks berupa iterable (multiple matches), gunakan salah satu
    if isinstance(idx, Iterable):
        idx = idx[0]

    # cari skor kemiripan
    similarity_scores = list(enumerate(cosine_resep[idx]))

    # terus diurut skor dari tertinggi ke terendah, lewati indeks pertama (resep yang sama)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1 : n + 1]

    # ambil indeks dari rekomendasi
    recipe_indices = [i[0] for i in similarity_scores]

    # nama resep, bahan, dan RecipeId dari DataFrame
    recommended_names = nama.iloc[recipe_indices]
    recommended_ingredients = cbf_rekomendasi1['RecipeIngredientParts'].iloc[recipe_indices]
    recommended_ids = cbf_resep['RecipeId'].iloc[recipe_indices]

    # output dalam DataFrame
    rekomendasi_cbf = pd.DataFrame({
        'RecipeId': recommended_ids.values,
        'Name': recommended_names.values,
        'RecipeIngredientParts': recommended_ingredients.values
    })

    return rekomendasi_cbf

rekomendasi_cbf_resep_dan_bahan("Banana Bread", n=8)

def fungsi_bobot(df):
    v = df['Rating']
    R = df['AggregatedRating']
    return (v / (v + m) * R) + (m / (m + v) * C)

hybrid_resep = cbf_resep

def rekomendasi_hybrid_bobot(resep, n=10, m=50, C=None):
    """
    Menghasilkan rekomendasi resep berdasarkan Bayesian Weighted Average.

    Args:
        resep (str): Nama resep untuk mencari rekomendasi.
        n (int): Jumlah rekomendasi yang diinginkan (default 10).
        m (int): Minimum jumlah ulasan untuk mempertimbangkan item.
        C (float): Rata-rata rating di seluruh dataset. Jika None, dihitung otomatis.

    Returns:
        pd.DataFrame: DataFrame berisi hasil rekomendasi dengan kolom RecipeId, Name, Rating, AggregatedRating, WR, dan RecipeIngredientParts.
    """
    # mencari nilai C
    if C is None:
        C = cbf_resep['AggregatedRating'].mean()

    # indeks dari resep berdasarkan nama resep
    idx = indice[resep]
    if isinstance(idx, Iterable):
        idx = idx[0]

    # cari skor kemiripan lalu diurutkan dari tertinggi ke terendah dan yang indeks pertama dilewat karena resep itu sendiri
    similarity_scores = list(enumerate(cosine_resep[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1 : n + 1]
    recipe_indices = [i[0] for i in similarity_scores]

    recommendations = hybrid_resep.iloc[recipe_indices]

    # hitung Weighted Rank (WR)
    recommendations['WR'] = (
        (recommendations['ReviewCount'] / (recommendations['ReviewCount'] + m)) * recommendations['AggregatedRating']
        + (m / (recommendations['ReviewCount'] + m)) * C
    )

    rekomendasi_hybrid = recommendations[[
        'RecipeId', 'Name', 'ReviewCount', 'AggregatedRating', 'WR', 'RecipeIngredientParts'
    ]].sort_values('WR', ascending=False).head(n)

    return rekomendasi_hybrid

rekomendasi_hybrid_bobot("Sweet Potato Casserole",
                         n=8,
                         m=10)

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_recipes, num_categories, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            num_users, embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-5)
        )
        self.user_bias = layers.Embedding(num_users, 1)

        self.recipe_embedding = layers.Embedding(
            num_recipes, embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.recipe_bias = layers.Embedding(num_recipes, 1)

        self.category_embedding = layers.Embedding(
            num_categories, embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])

        recipe_vector = self.recipe_embedding(inputs[:, 1])
        recipe_bias = self.recipe_bias(inputs[:, 1])

        category_vector = self.category_embedding(inputs[:, 2])

        dot_user_recipe = tf.tensordot(user_vector, recipe_vector, 2)
        dot_recipe_category = tf.tensordot(recipe_vector, category_vector, 2)

        x = dot_user_recipe + dot_recipe_category + user_bias + recipe_bias
        return tf.nn.sigmoid(x)
    
num_users = df2['AuthorId_encoded'].nunique()
num_recipes = df2['RecipeId_encoded'].nunique()
num_categories = df2['RecipeCategory_encoded'].nunique()
embedding_size = 100

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

model2 = RecommenderNet(num_users, num_recipes, num_categories, embedding_size)
model2.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
)

history2 = model2.fit(
    x=X_train,
    y=y_train,
    batch_size=128,
    epochs=10,
    validation_data=(X_val, y_val)
)

def recommend_recipes(model, author_id, top_n=5):
    # mengambil semua recipe id yang ada
    all_recipe_ids = df2['RecipeId_encoded'].unique()
    
    # mengambil resep yang sudah diberi rating oleh author_id
    rated_recipes = df2[df2['AuthorId_encoded'] == author_id]['RecipeId_encoded'].values
    
    # menentukan resep yang belum pernah dicoba oleh author_id
    unseen_recipes = [recipe_id for recipe_id in all_recipe_ids if recipe_id not in rated_recipes]
    
    # membuat input untuk prediksi: AuthorId dan RecipeId yang belum dicoba
    unseen_data = []
    for recipe_id in unseen_recipes:
        category = df2[df2['RecipeId_encoded'] == recipe_id]['RecipeCategory_encoded'].values[0]
        unseen_data.append([author_id, recipe_id, category])
    
    unseen_data = np.array(unseen_data)
    
    # memprediksi rating untuk resep yang belum pernah dicoba
    predicted_ratings = model.predict(unseen_data)
    
    # menambahkan prediksi rating ke dalam dataframe
    unseen_recipes_with_ratings = list(zip(unseen_recipes, predicted_ratings.flatten()))
    
    # mengurutkan berdasarkan rating tertinggi dan memilih top n
    top_recommendations = sorted(unseen_recipes_with_ratings, key=lambda x: x[1], reverse=True)[:top_n]
    
    # membuat dataframe hasil rekomendasi langsung
    top_recommendations_df = pd.DataFrame(top_recommendations, columns=['RecipeId_encoded', 'PredictedRating'])
    
    # menambahkan informasi RecipeId dan RecipeCategory dari df2
    top_recommendations_df = top_recommendations_df.merge(
        df2[['RecipeId_encoded', 'RecipeId', 'RecipeCategory']].drop_duplicates(),
        on='RecipeId_encoded',
        how='left'
    )
    
    # menambahkan kolom Name dari resep1
    top_recommendations_df = top_recommendations_df.merge(
        resep1[['RecipeId', 'Name']],
        on='RecipeId',
        how='left'
    )
    
    return top_recommendations_df

# menggunakan fungsi untuk mendapatkan top 5 rekomendasi untuk random author_id
random_author_id = np.random.choice(df2['AuthorId_encoded'].unique())

recommended_recipes = recommend_recipes(model2, random_author_id, 5)
recommended_recipes

def precision_at_k(rekomendasi, relevan, k):
    """
    Hitung Precision@K.
    Args:
        rekomendasi (list): Daftar item yang direkomendasikan.
        relevan (list): Daftar item yang relevan.
        k (int): Jumlah rekomendasi yang dipertimbangkan.
    Returns:
        float: Precision@K.
    """

    rekomendasi = rekomendasi[:k]
    relevan_set = set(relevan)
    hit = len([item for item in rekomendasi if item in relevan_set])
    return hit / k

def recall_at_k(rekomendasi, relevan, k):
    """
    Hitung Recall@K.
    Args:
        rekomandasi (list): Daftar item yang direkomendasikan.
        relevan (list): Daftar item yang relevan.
        k (int): Jumlah rekomendasi yang dipertimbangkan.
    Returns:
        float: Recall@K.
    """

    rekomendasi = rekomendasi[:k]
    relevan_set = set(relevan)
    hit = len([item for item in rekomendasi if item in relevan_set])
    return hit / len(relevan) if relevan else 0

def average_precision_at_k(rekomendasi, relevan, k):
    """
    Hitung Average Precision@K.
    Args:
        rekomendasi (list): Daftar item yang direkomendasikan.
        relevan (list): Daftar item yang relevan.
        k (int): Jumlah rekomendasi yang dipertimbangkan.
    Returns:
        float: Rata-rata Precision@K.
    """

    relevan_set = set(relevan)
    skor_precision = [
        precision_at_k(rekomendasi, relevan, i+1)
        for i in range(min(k, len(rekomendasi)))
        if rekomendasi[i] in relevan_set
    ]

    return sum(skor_precision) / len(relevan) if relevan else 0

def mean_average_precision_at_k(semua_rekomendasi, semua_relevan, k):
    """
    Hitung Mean Average Precision@K.
    Args:
        semua_rekomendasi (dict): {user_id: list of recommended items}.
        semua_relevan (dict): {user_id: list of relevant items}.
        k (int): Jumlah rekomendasi yang dipertimbangkan.
    Returns:
        float: MAP@K.
    """

    map_score = 0
    for user, rekomendasi in semua_rekomendasi.items():
        relevan = semua_relevan.get(user, [])
        map_score += average_precision_at_k(rekomendasi, relevan, k)
    return map_score / len(semua_rekomendasi)

def bikin_relevan(dataframe, rating_threshold=4.0, review_count_threshold=10):
    """
    Membuat daftar relevansi.
    
    Args:
        dataframe (pd.DataFrame): DataFrame yang mengandung data resep.
        rating_threshold (float): Ambang batas rating untuk dianggap relevan.
        review_count_threshold (int): Ambang batas jumlah ulasan untuk dianggap relevan.
    
    Returns:
        dict: Dictionary dengan format {recipe_name: [relevant_recipe_ids]}.
    """
    # Filter resep relevan berdasarkan kriteria
    relevan_df = dataframe[
        (dataframe['AggregatedRating'] >= rating_threshold) &
        (dataframe['ReviewCount'] >= review_count_threshold)
    ]
    
    # Kelompokkan relevansi berdasarkan nama resep
    relevances = relevan_df.groupby('Name')['RecipeId'].apply(list).to_dict()
    
    return relevances

top_names = cbf_rekomendasi1['Name'].value_counts().index.tolist()

relevances = bikin_relevan(cbf_resep, rating_threshold=3.5, review_count_threshold=5)

k = 40  # Jumlah rekomendasi yang dipertimbangkan
cbf_recommendations = {}
ap_scores = {}

for recipe_name in top_names:
    # Ambil rekomendasi untuk setiap nama resep
    rekomendasi = rekomendasi_cbf_resep_dan_bahan(recipe_name, n=k)['RecipeId'].tolist()
    cbf_recommendations[recipe_name] = rekomendasi

    # Ambil relevansi resep target
    relevan = relevances.get(recipe_name, [])
    
    # Evaluasi metrik
    precision = precision_at_k(rekomendasi, relevan, k)
    recall = recall_at_k(rekomendasi, relevan, k)
    ap = average_precision_at_k(rekomendasi, relevan, k)
    
    # Cetak hasil evaluasi untuk resep tertentu
    print(f"\nEvaluasi untuk '{recipe_name}':")
    print(f"Precision@{k}: {precision}")
    print(f"Recall@{k}: {recall}")
    print(f"AP@{k}: {ap}")
    
    # Simpan AP untuk perhitungan MAP
    ap_scores[recipe_name] = ap

# Hitung MAP@K
map_score = mean_average_precision_at_k(cbf_recommendations, relevances, k)
print(f"\nMAP@{k} untuk Top Nama Resep: {map_score}")

plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.title('Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history2.history['root_mean_squared_error'], label='Training RMSE')
plt.plot(history2.history['val_root_mean_squared_error'], label='Validation RMSE')
plt.title('RMSE History')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()