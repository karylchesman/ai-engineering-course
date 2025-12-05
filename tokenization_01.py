import nltk

nltk.download("punkt_tab")

text = "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados. Sem serem programados explicitamente para cada tarefa."

word_tokens = nltk.word_tokenize(text)

print(word_tokens)

setence_tokens = nltk.sent_tokenize(text)
print(setence_tokens)


def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum()]
    return tokens


documents = [
    "Machine learning é o aprendizado automático de máquinas a partir de dados.",
    "Ele permite que sistemas façam previsões e decisões sem programação explícita.",
    "É usado em áreas como reconhecimento de voz, imagens e recomendação de conteúdo.",
]

preprocessed_documents = [" ".join(preprocess(doc)) for doc in documents]
print(preprocessed_documents)
