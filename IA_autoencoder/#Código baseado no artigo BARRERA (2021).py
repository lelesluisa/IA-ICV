#Código baseado no artigo BARRERA (2021)
#A ideia deste código é prever um padrão anormal da válvula ICV

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #pra normalização
from sklearn.metrics import silhouette_score #métrica pros clusters
from sklearn.cluster import KMeans #clustering escolhido
from tensorflow.keras.models import Sequential #modelo sequencial pro treinamento do autoenconder
from tensorflow.keras.layers import Dense #pra definir as camadas da rede
import pyswarms as ps #será utilizado para o PSO (particle swarm optimization) pra escolher o número de clusters
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split #Treinamento do modelo
import tensorflow as tf
import seaborn as sns # Pra criar a boxplot, ajudar a definir os clusters


#===========================================
#Leitura do banco de dados
#===========================================
#Por enquanto espera-se o banco já tratado (com menos redundâncias), arquivo csv com as features pré definidas
#Posteriormente talvez dê pra já fazer o heatmap aqui e excluir uma das variáveis que correlaciona fortemente com outra

banco=pd.read_csv("C:/Users/luisa/Downloads/bd_coracao.txt", sep=',', names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thelach', 'exang', 'oldspeak', 'slope'])
banco = banco.iloc[:, 1:] #Essa primeira coluna é só a posição 

#===========================================
#Normalização utilizando o método Z-score
#===========================================
# Z = x - (média)/(desvio), no final a média é 0 e o desvio é 1, então os valores variam 
#entre -1 e 1, isso vai melhorar a eficiência do treinamento.

scaler = StandardScaler()
X_normalizado = scaler.fit_transform(banco)

#===========================================
#Função de otimização pela métrica silhouette
#===========================================
#Serão testados n números de clusters e os valores do índice serão salvos na lista "scores"

def silhouette(n_clusters_array):
    scores = []
    for n in n_clusters_array.astype(int):
        n = int(n)  # converte para int escalar
        if n < 2 or n >= len(X_normalizado):
            scores.append(-1) #Se for menor que 1 ou maior que o número de amostras salva o indicador como -1, que será consequentemente descartado.
            continue
        kmeans = KMeans(n_clusters=n, random_state=42).fit(X_normalizado) #Executa o KMeans com n clusters
        labels = kmeans.labels_ #Obtém os rótulos
        score = silhouette_score(X_normalizado,labels) #Calcula o índice, varia de 1 (bom) até -1 (ruim)
        scores.append(score) #Salva os valores na lista
    return -np.array(scores) #Multiplica por -1 porque o PSO encontra o menor negativo como o melhor.

#===========================================
#PSO para encontrar o melhor número de clusters
#===========================================

bounds = (np.array([1]), np.array([10])) #De 1 até 10 clusters
PSO=ps.single.GlobalBestPSO(n_particles=5, dimensions=1, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)
best_cost, best_pos = PSO.optimize(silhouette, iters=10)
n_clusters = int(best_pos[0])

print(f'número ótimo de clusters:{n_clusters}')

#===========================================
#Aplicação real do KMeans
#===========================================

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_normalizado) #Vai gerar um vetor do tamanho de X_normalizado, mas com as posições de cada variáveis, tipo :(1,1,1,2,3,0,1,2,3,3,2)
print(f'Lista posição dos valores em clusters {clusters}')

#===========================================
# Visualização dos clusters em 2D com PCA
#===========================================

pca = PCA(n_components=2) #Reduzir os dados originais para 2 dimensões
X_pca = pca.fit_transform(X_normalizado)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.title(f'Visualização dos clusters (KMeans com {n_clusters} clusters)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(scatter, label='Cluster')
plt.show()

#===========================================
#Boxplot pra auxiliar a nomear os clusters
#===========================================

#Essa função vai fazer o plot de forma estatístico em cima dos valores referentes a cada cluster passando pelo for as variáveis em col.

banco['cluster'] = clusters  # clusters = resultado do KMeans 

for col in banco.columns[:-1]:  # exclui 'cluster'
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='cluster', y=col, data=banco)
    plt.title(f'Distribuição de {col} por cluster')
    plt.show()

#===========================================
#Treinamento do autoencoder por cluster
#===========================================

autoencoders = {}
erro_reconstrucaos = {}

for cluster_id in range(n_clusters):
    print(f'treinando o autoencoder referente ao cluster {cluster_id}')
    X_cluster = X_normalizado[clusters == cluster_id] #Dessa forma, os valores X_cluster serão aqueles onde cluster_id (referente ao n) for == ao vetor característico cluster
    input_dim = X_cluster.shape[1] #número de colunas, ou seja, features

    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(input_dim, activation='linear'),
    ])

    model.compile(optimizer='adam', loss='mse')

    # Separação em 80% pra treinamento e 20% para validação
    X_train, X_test = train_test_split(X_cluster, test_size=0.2, random_state=42)

    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

    # Erro_reconstrucao calculado sobre o conjunto validação
    reconstruido = model.predict(X_test, verbose=0)
    erros = np.mean(np.abs(X_test - reconstruido), axis=1)
    erro_reconstrucao = np.mean(erros)

    print(f'erro_reconstrucao para cluster {cluster_id}: {erro_reconstrucao}')

    # Salva o modelo e o erro_reconstrucao
    autoencoders[cluster_id] = model
    erro_reconstrucaos[cluster_id] = erro_reconstrucao
    
print("Treinamento finalizado")

#===============================================
#Próximos passos
#Usar o modelo treinado para detecção de falhas, criar anomalias para validar, janelas deslizantes tbm