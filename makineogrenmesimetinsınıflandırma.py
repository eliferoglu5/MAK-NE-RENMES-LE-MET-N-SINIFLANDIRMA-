# Gerekli kütüphaneleri yüklüyoruz
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dosya yolunu tanımlıyoruz
dosya_yolu = r"C:\Users\egeli\Documents\news.json\News_Category_Dataset_v3.json"

# Dosyanın varlığını kontrol ediyoruz
print("Dosya var mı?", os.path.exists(dosya_yolu))

# 1. Veri kümesini yüklüyoruz
df = pd.read_json(dosya_yolu, lines=True)
df = df[['category', 'short_description']]

# 2. Beş kategoriye indiriyoruz (Ekonomi, Spor, Teknoloji, Sağlık, Politika)
df = df[df['category'].isin(['SPORTS', 'BUSINESS', 'TECHNOLOGY', 'POLITICS', 'HEALTH'])]

# 3. Sütun adlarını sadeleştiriyoruz
df = df.rename(columns={'short_description': 'text', 'category': 'label'})
df['label'] = df['label'].str.lower()  # etiketleri küçük harfe çeviriyoruz

# 4. Temel ön işlem: küçük harfe çevirme + noktalama temizleme
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # noktalama işaretlerini kaldır
    return text.lower()

df['clean_text'] = df['text'].apply(clean_text)

# 5. Metinleri sayısallaştırma (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# 6. Eğitim ve test verisine ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model 1: Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

print("=== Naive Bayes Sonuçları ===")
print("Doğruluk:", accuracy_score(y_test, y_pred_nb))
print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred_nb))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nb))

# 8. Model 2: Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("\n=== Logistic Regression Sonuçları ===")
print("Doğruluk:", accuracy_score(y_test, y_pred_lr))
print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred_lr))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_lr))

# 9. Karmaşıklık Matrislerini çiziyoruz
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y), ax=axs[0])
axs[0].set_title("Naive Bayes - Confusion Matrix")
axs[0].set_xlabel("Tahmin")
axs[0].set_ylabel("Gerçek")

sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(y), yticklabels=np.unique(y), ax=axs[1])
axs[1].set_title("Logistic Regression - Confusion Matrix")
axs[1].set_xlabel("Tahmin")
axs[1].set_ylabel("Gerçek")

plt.tight_layout()
plt.show()
