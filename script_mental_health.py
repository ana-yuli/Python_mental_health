#Importando as bibliotecas

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#%% Tratamento de dados

# Extraindo informaçoes dos dados
dataset=pd.read_csv('Deepression.csv') #importando dados
dataset.info() #informacao da tabela
dataset.describe() #resumo estatistico
dataset.head(10) #10 primeiras linhas do arquivo
dataset.tail(10) #ultimas linhas nao possuem dados
dataset.iloc[540] #localizando onde começam os dados nulos. 


# Limpeza de dados
nulos = dataset.isnull().sum() #identificando os nulos
Duplicados = dataset.duplicated().sum() #identificando duplicados
dataset.dropna(inplace=True) #excluindo os dados nulos (NAN),
dataset['Estado_Depressao'] = (
    dataset['Estado_Depressao']
    .astype(str)
    .str.replace(r'^\d+', '', regex=True)  # remove números do início
    .str.replace('\t', '', regex=False)    # remove tabulações
    .str.strip()                           # remove espaços nas pontas
)#limpando nomes da coluna

    

#Renomeando colunas 
dataset=dataset.rename(columns={'Number':'Numero',
               'Sleep': 'Sono',
    'Appetite': 'Apetite',
    'Interest': 'Interesse',
    'Fatigue': 'Fadiga',
    'Worthlessness': 'Sentimento_Inutilidade',
    'Concentration': 'Concentracao',
    'Agitation': 'Agitacao',
    'Suicidal Ideation': 'Ideacao_Suicida',
    'Sleep Disturbance': 'Disturbio_Sono',
    'Aggression': 'Agressividade',
    'Panic Attacks': 'Ataques_Panico',
    'Hopelessness': 'Desesperanca',
    'Restlessness': 'Inquietacao',
    'Low Energy': 'Baixa_Energia',
    'Depression State': 'Estado_Depressao'})

#%% Analisando os dados

dataset.groupby ('Estado_Depressao').mean().T #media de sintomas por estado de depressão
numericos=dataset.select_dtypes(include=(['int64','float64'])) #deixando somente variaveis quantitativas

# Matriz correlação de Pearson 

matriz_corre=numericos.corr()
mask = np.triu(np.ones_like(matriz_corre, dtype=bool))
plt.figure(figsize=(12,8))
sns.heatmap(
    matriz_corre,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='Blues')
plt.title('Matriz de Correlação')
plt.show()


#Coeficiente de correlação 
Coef_corre = (numericos.std()/numericos.mean())*100


# Porcentagem estado de depressão 
Porcentagem=dataset['Estado_Depressao'].value_counts(normalize=True)*100 

# Gráfico porcetagem depressão 
plt.figure(figsize=(8,8))
plt.pie( Porcentagem , labels=Porcentagem.index, autopct='%1.1f%%')
plt.title('Estado de depressão')
plt.pie(Porcentagem,colors=plt.cm.Pastel2.colors)
plt.show()

#%% Classificação por estado de depressão 

sintomas = [
    'Sono', 'Apetite', 'Interesse', 'Fadiga',
    'Sentimento_Inutilidade', 'Concentracao', 'Agitacao',
    'Ideacao_Suicida', 'Disturbio_Sono', 'Agressividade',
    'Ataques_Panico', 'Desesperanca', 'Inquietacao',
    'Baixa_Energia']
severo = dataset[dataset['Estado_Depressao'] == 'Severe']
sintomas_severo=severo[sintomas].mean().sort_values(ascending=False)

# Gráfico dos sintomas de depressão severa 
sintomas_severo.sort_values(ascending=True).plot(kind='barh', figsize=(8,6))
plt.title('Sintomas depressão severa')
plt.xlabel('Média')
plt.show()

#%% Conclusão 

A matriz de correlação de Pearson revelou fortes associações entre diversos sintomas 
relacionados à depressão. Destacam-se correlações elevadas entre sentimentos de inutilidade, 
desesperança, ideação suicida, agitação, ataques de pânico e inquietação, sugerindo que esses 
sintomas tendem a ocorrer simultaneamente nos indivíduos analisados.


Na análise específica do grupo com depressão severa, observou-se que os sintomas
 com maiores médias foram alterações relacionadas ao sono, dificuldade de concentração, 
 distúrbios do sono, perda de interesse e alterações no apetite. Esses resultados indicam 
 que sintomas cognitivos e comportamentais apresentam elevada intensidade nos casos 
 classificados como severos.

De forma geral, o projeto permitiu aplicar técnicas de limpeza de dados, 
estatística descritiva, análise de correlação e visualização de dados utilizando 
Python, Pandas, NumPy, Matplotlib e Seaborn, transformando dados brutos 
em informações que auxiliam na compreensão dos padrões associados aos diferentes
 níveis de depressão.






