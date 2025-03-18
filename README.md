
# Classificação de Crédito com Redes Neurais

Este projeto demonstra a construção de um modelo de rede neural para classificação de dados financeiros usando Keras e Scikit-learn. O modelo realiza a previsão de uma variável de classe (de crédito) a partir de um conjunto de dados contendo diversas características dos clientes. O código inclui etapas de pré-processamento, divisão dos dados, construção do modelo, treinamento e avaliação com matriz de confusão.

## Descrição do Código

1. **Carregamento dos Dados**:
   O conjunto de dados `Credit2.csv` é carregado usando o `pandas`. A primeira coluna (ID) é descartada, e os dados são divididos em variáveis preditoras (`previsores`) e a classe (`classes`).

2. **Pré-processamento**:
   - **Label Encoding**: A primeira coluna categórica é transformada em valores numéricos.
   - **One-Hot Encoding**: A segunda coluna categórica (com valores nominais) é transformada em variáveis binárias usando `OneHotEncoder`.
   - **Padronização**: As variáveis preditoras são padronizadas usando `StandardScaler` para garantir que todas as variáveis tenham a mesma escala, o que ajuda a melhorar a performance do modelo de rede neural.

3. **Divisão de Dados**:
   O dataset é dividido em conjunto de treinamento (80%) e teste (20%) utilizando `train_test_split` do `sklearn`.

4. **Construção do Modelo**:
   O modelo de rede neural é construído usando Keras:
   - A primeira camada é uma camada densa (fully connected) com 6 neurônios e a função de ativação ReLU.
   - A segunda camada é semelhante à primeira, também com 6 neurônios e a função de ativação ReLU.
   - A camada de saída tem um neurônio, com a função de ativação `sigmoid` (adequada para problemas de classificação binária).

5. **Treinamento do Modelo**:
   O modelo é treinado com o conjunto de treinamento usando `batch_size=2` e `epochs=100`.

6. **Avaliação do Modelo**:
   Após o treinamento, o modelo é testado com o conjunto de teste. A previsão é feita usando `classifier.predict` e a saída é transformada em um valor binário (0 ou 1) com base em um limiar de 0.5.

7. **Matriz de Confusão**:
   A matriz de confusão é calculada utilizando `confusion_matrix` do `sklearn`, permitindo avaliar a performance do modelo no conjunto de teste.

## Saída

Após a execução, o código imprime a matriz de confusão, que mostra o número de previsões corretas e incorretas para as duas classes de previsão (0 e 1).

```python
Confusão:
[[ 31  27]
 [ 21 121]]
```

- **TN (True Negative)**: Previsões corretas de classe 0.
- **FP (False Positive)**: Previsões incorretas de classe 0.
- **FN (False Negative)**: Previsões incorretas de classe 1.
- **TP (True Positive)**: Previsões corretas de classe 1.




