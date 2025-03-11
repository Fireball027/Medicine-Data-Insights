import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from warnings import filterwarnings
filterwarnings("ignore")

# Load dataset
df = pd.read_csv('medicine.csv')

# Checking the dataset
df.head()
print("Dataset Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())

# Dropping missing values
df.dropna(inplace=True)

# Checking for duplicates and removing them
df.drop_duplicates(inplace=True)

# Processing text
df['Description'] = df['Description'].apply(lambda x: x.split())
df['Reason'] = df['Reason'].apply(lambda x: x.split())
df['Description'] = df['Description'].apply(lambda x: [i.replace(" ", "") for i in x])
df['tags'] = df['Description'] + df['Reason']

df['tags'] = df['tags'].apply(lambda x: " ".join(x))
df['tags'] = df['tags'].apply(lambda x: x.lower())

# Stemming process
ps = PorterStemmer()


def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])


df['tags'] = df['tags'].apply(stem)

# Vectorization
cv = CountVectorizer(stop_words='english', max_features=5000)
vectors = cv.fit_transform(df['tags']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vectors)


def recommend(medicine):
    if medicine not in df['Drug_Name'].values:
        print("Medicine not found in dataset.")
        return
    medicine_index = df[df['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    print("Top 5 recommended medicines:")
    for i in medicines_list:
        print(df.iloc[i[0]].Drug_Name)


# Save data for later use
pickle.dump(df.to_dict(orient="records"), open('medicine_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df['tags']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Medicine Descriptions")
plt.show()

# Top 10 most frequent medicines
plt.figure(figsize=(12, 6))
df['Drug_Name'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.xlabel("Medicine Name")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Medicines in Dataset")
plt.xticks(rotation=45)
plt.show()

# Heatmap of Similarity Scores (for the first 10 medicines)
plt.figure(figsize=(12, 6))
sns.heatmap(similarity[:10, :10], cmap='coolwarm', annot=True)
plt.title("Cosine Similarity Heatmap (Top 10 Medicines)")
plt.show()
