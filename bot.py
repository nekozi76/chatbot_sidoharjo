import pandas as pd
import nltk
import re
import json
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Memuat data dari file JSON
with open('dataset.json', 'r') as file:
    data = json.load(file)

# Menyiapkan data untuk DataFrame
intents = data['intents']
patterns = []
tags = []
responses = []

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
        responses.append(intent['responses'])

# Membuat DataFrame untuk data training
df = pd.DataFrame({'patterns': patterns, 'tags': tags, 'responses': responses})

# Pastikan NLTK telah mengunduh data yang diperlukan
nltk.download('punkt')

def clean_text(text):
    # Cleansing: hapus karakter non-alfabet dan spasi
    text = text.lower()  # Casefolding
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def preprocess(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)  # Tokenizing
    return tokens

# Menerapkan preprocessing ke dalam pola
df['cleaned_patterns'] = df['patterns'].apply(clean_text)
df['tokens'] = df['patterns'].apply(preprocess)



# Menggunakan TfidfVectorizer untuk vectorisasi teks
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())  # Menggunakan TfidfVectorizer
X = vectorizer.fit_transform(df['cleaned_patterns'])

# Label encoding untuk tags
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['tags'])

# Membagi data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Menggunakan GridSearchCV untuk mencari parameter terbaik
param_grid = {'alpha': [0.1, 0.5, 1, 2, 5]}  # Contoh parameter yang ingin diuji
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Menampilkan parameter terbaik yang ditemukan oleh GridSearchCV
print("Best parameters found: ", grid_search.best_params_)

# Melatih model menggunakan parameter terbaik
best_model = grid_search.best_estimator_


# Prediksi pada data test
y_pred = best_model.predict(X_test)

# Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Menghitung confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Menampilkan classification report
unique_labels = np.unique(np.concatenate([y_test, y_pred]))
class_report = classification_report(
    y_test, 
    y_pred, 
    labels=unique_labels,
    target_names=label_encoder.classes_[unique_labels]
)
print("Classification Report:")
print(class_report)

import random
CONFIDENCE_THRESHOLD = 0.05 # Anda bisa menyesuaikan nilai ini (misalnya antara 0.2 - 0.5)
def chatbot_response(user_input):
    cleaned_input = clean_text(user_input)
    if not cleaned_input.strip(): # Pemeriksaan input kosong (praktik yang baik)
        return "Mohon berikan pertanyaan yang lebih spesifik."

    input_vector = vectorizer.transform([cleaned_input])

    # Menggunakan predict_proba untuk mendapatkan probabilitas
    probabilities = best_model.predict_proba(input_vector)

    # Mendapatkan probabilitas tertinggi dan indeks kelasnya
    max_proba = np.max(probabilities)
    predicted_class_index = np.argmax(probabilities)

    if max_proba >= CONFIDENCE_THRESHOLD:
        # Mengubah indeks prediksi kembali ke label tag asli
        predicted_label = label_encoder.inverse_transform([predicted_class_index])
        tag = predicted_label[0]

        possible_responses_list = df[df['tags'] == tag]['responses'].values
        
        if len(possible_responses_list) > 0 and isinstance(possible_responses_list[0], list) and possible_responses_list[0]:
            # possible_responses_list[0] adalah list aktual dari string respons untuk tag tersebut
            return random.choice(possible_responses_list[0])
            
    else:
        # Ini adalah respons yang Anda inginkan ketika bot tidak cukup percaya diri
        return "Mohon maaf, saya belum bisa memahami atau menjawab pertanyaan tersebut. Saat ini, saya hanya dapat membantu terkait layanan administrasi Desa Sidoharjo. Apakah ada hal lain seputar administrasi Desa Sidoharjo yang bisa saya bantu?"

#import pandas as pd

# Pastikan patterns dan tags sudah benar dan panjangnya sama
#df = pd.DataFrame({'patterns': patterns, 'tags': tags})

# Hitung jumlah data per tag
#count_per_tag = df['tags'].value_counts()

#print(count_per_tag)

num_intents = len(intents)
print(f"Jumlah intent (tag): {num_intents}") #

# Hitung jumlah pattern per intent dan total pattern
total_patterns = 0
for intent in intents:
    tag = intent['tag']
    patterns = intent['patterns']
    num_patterns = len(patterns)
    total_patterns += num_patterns
    #print(f"Intent '{tag}': {num_patterns} pattern")

print(f"Total seluruh pattern: {total_patterns}")
print(f"Total Response: {num_intents}")

# Tes chatbot dengan debugging
#while True:
    #user_input = input("Anda: ")
    #if user_input.lower() == 'exit':
        #break
    #response = chatbot_response(user_input)
    #print(f"Chatbot: {response}")

