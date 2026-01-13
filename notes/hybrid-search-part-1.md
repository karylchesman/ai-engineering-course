# Busca Hibrida parte 1

Quando se pesquisa unicamente pela semântica capturada pelos embeddings, ou *Dense Vector Search*, isso pode trazer documentos que são similares porém não relevantes. Aí entra a busca híbrida (*hybrid search*), que combina mais de um algoritmo de busca para melhorar a relevância e precisão dos resultados, como a combinação de busca semântica com busca por palavras-chave usando o BM25, por exemplo.

Ao usar o *hybrid search*, precisamos também de um algoritmo para combinar os resultados das diferentes buscas. Um algoritmo simples é o *reciprocal rank fusion* (RRF), que combina os rankings das diferentes buscas somando os recíprocos das posições dos documentos nos rankings. A fórmula do RRF é:

```RRF(d) = Σ (1 / (k + rank_i(d)))
```

Após combinar, precisamos "ajustar" o ranking final, o que chamamos de *reranking*, onde os resultados de uma busca inicial são reordenados com base em um segundo critério, como a similaridade semântica. Por exemplo, podemos primeiro buscar documentos usando BM25 e depois reordenar esses resultados com base na similaridade dos embeddings.

Então numa pipeline de busca híbrida, podemos seguir os seguintes passos:

1. Pre-fetch: Buscar um conjunto inicial de documentos com cada técnica (por exemplo, BM25 e busca por embeddings).
2. Fusion: Combinar os resultados das diferentes buscas usando um algoritmo como o RRF.
3. Rerank: Reordenar esses documentos usando a similaridade dos embeddings, como com Cross-Encoder (precisos e lentos), Late Interaction models.
