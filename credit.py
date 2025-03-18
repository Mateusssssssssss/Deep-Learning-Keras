from tkinter import Label
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import Dense
from sklearn.compose import make_column_transformer
import pandas as pd

dados = pd.read_csv('Credit2.csv', sep=';')
print(dados.head())
#Ignora a 1 pois não tem valor semantico (ID)
previsores = dados.iloc[:, 1:10].values
classes = dados.iloc[:, 10].values
#Transformação de valores nominais em numericos (0 a 3)
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])

#make_column_transformer(...): Essa função do sklearn.compose permite aplicar 
# diferentes transformações em colunas específicas do conjunto de dados.
#Aqui, estamos dizendo que queremos aplicar One-Hot Encoding à coluna de índice 1 e manter 
# as demais colunas inalteradas.
#categories='auto': significa que o OneHotEncoder detectará automaticamente todas as categorias presentes na coluna.
# sparse=False: significa que a saída será uma matriz densa (numpy array), em vez de uma matriz esparsa.
# [1]: indica que a transformação será aplicada na segunda coluna (índice 1) do dataset.
#remainder='passthrough':  isso significa que as outras colunas do dataset não serão alteradas e serão mantidas na saída.
onehotencoder = make_column_transformer((OneHotEncoder(categories='auto',sparse_output=False), [1] ), remainder='passthrough')

# Codifica variáveis categóricas (coluna de índice 1) em valores numéricos usando One-Hot Encoding.
# Substitui a coluna categórica por colunas binárias, mantendo as outras colunas inalteradas.
# Transforma os dados em um array NumPy, pronto para ser usado em modelos de machine learning. 
previsores  = onehotencoder.fit_transform(previsores)

#Se a variável categórica foi transformada com One-Hot Encoding, 
# a primeira coluna pode ser removida para evitar dependência linear (chamada de "dummy variable trap"). 
# Isso é útil em regressão para evitar redundância nos dados.
previsores = previsores[:,1:]

#Transformação de valores nominais em numericos (0 e 1)
labelencoder_class = LabelEncoder()
classes = labelencoder_class.fit_transform(classes)


#Divisão entre treinamento e teste
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                 classes,
                                                                 test_size=0.2,
                                                                 random_state=0)

#Ver como foram destribuidos os dados
print(len(x_treinamento), len(x_teste), len(y_treinamento),len(y_teste))
#StandardScaler(): padroniza os dados, transformando-os para ter média 0 e desvio padrão 1. 
# Isso melhora o desempenho de modelos de machine learning que são sensíveis a escalas diferentes, 
# como SVM e redes neurais. 
# Padronização z-score
sc = StandardScaler()
#fit_transform(x_treinamento) faz duas coisas ao mesmo tempo.
#fit(): Calcula a média e o desvio padrão das colunas do x_treinamento.
#transform(): Usa a fórmula do Z-score para transformar os dados.
#Padronização dos dados
x_treinamento = sc.fit_transform(x_treinamento)
x_teste = sc.fit_transform(x_teste)

print(x_teste)


classifier = Sequential()

# Dense: Adiciona uma camada totalmente conectada (fully connected) à rede neural.
# units=6: Define que esta camada terá 6 neurônios.
# kernel_initializer='uniform': Inicializa os pesos da camada de forma aleatória em um intervalo uniforme.
# activation='relu': Aplica a função de ativação ReLU (Rectified Linear Unit) aos neurônios. A ReLU ajuda a introduzir não linearidade no modelo.
# input_dim=12: Esta é a camada de entrada, então ela espera 12 características de entrada.
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim= 12))

# Adiciona outra camada totalmente conectada com 6 neurônios.
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

#A função de ativação sigmoid. Ela é muito comum na camada de saída para problemas de 
# classificação binária, pois a função sigmoid retorna valores entre 0 e 1, 
# que podem ser interpretados como probabilidades.
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#optimizer='adam': O Adam é um dos otimizadores mais populares, que combina as vantagens de outros métodos 
# (como o gradiente descendente adaptativo) para ajustar os pesos da rede de forma eficiente.

# loss='binary_crossentropy': O binary cross-entropy é a função de perda usada em problemas de classificação binária. 
# Ela calcula a diferença entre as previsões da rede e os rótulos reais (0 ou 1).

# metrics=['accuracy']: A métrica usada para avaliar o desempenho da rede será a acurácia. 
# Ela mede a proporção de previsões corretas
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Treinamento
#batch_size=10 significa que o modelo vai processar 10 amostras de dados de cada vez. 
# Ou seja, após o modelo processar essas 10 amostras, 
# ele atualizará os pesos com base no erro calculado para essas 10 amostras. 
# Esse processo é repetido até que todo o conjunto de dados tenha sido usado.
classifier.fit(x_treinamento, y_treinamento,batch_size=2 ,epochs=100)
#Matrix
y_pred = classifier.predict(x_teste)
y_pred = (y_pred > 0.5)

# calcular a matriz de confusão
confusao = confusion_matrix(y_teste, y_pred)
print(f'Confusão:\n{confusao}')

