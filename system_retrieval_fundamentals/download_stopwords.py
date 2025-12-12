import nltk
import os

try:
    nltk.download("stopwords")
    print("Download concluído!")
except Exception as e:
    print(f"Erro ao baixar: {e}")
