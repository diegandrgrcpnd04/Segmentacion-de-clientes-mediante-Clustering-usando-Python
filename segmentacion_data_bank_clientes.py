# -*- coding: utf-8 -*-

#%%
#pip install kmodes

#%%
#INICIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from kmodes.kmodes import KModes

#%%




datos = pd.read_csv("C:/Users/Juvina/Desktop/bank-full.csv", engine="python", sep=";")

bank=pd.DataFrame(datos)


#%%

bank.info()

#age, job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome
#%%
bank_cust=bank[["age", "job", "marital", "education", "default", "housing", "loan", "contact","day", "month","poutcome"]]


#%%
#ver el maximo de age

print(max(bank_cust["age"]))

#%%
#convertir la edad en variable categorica
bank_cust["age_bin"]=pd.cut(bank_cust["age"], [0,20,30,40,50,60,70,80,90,100],labels=["0-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"])
bank_cust = bank_cust.drop("age", axis=1)
bank_cust.head()



#%%

#Categorizar las variables

bank_cust_copy=bank_cust.copy()
le = preprocessing.LabelEncoder()
bank_cust= bank_cust.apply(le.fit_transform)
bank_cust.head()


#PROCESO
#%%
#Usando K-Mode con Inicializacion "Cao"
km_cao=KModes(n_clusters=3, init="Cao", n_init=1, verbose=1)
fitClusters_cao= km_cao.fit_predict(bank_cust)

fitClusters_cao

clusterCentroidsDf=pd.DataFrame(km_cao.cluster_centroids_)

clusterCentroidsDf.columns=bank_cust.columns

clusterCentroidsDf


#%%

#Usando K-Mode con Inicializacion "Huang"
km_huang=KModes(n_clusters=2, init="Huang", n_init=1, verbose=1)
fitClusters_huang= km_huang.fit_predict(bank_cust)

fitClusters_huang

clusterCentroidsDf=pd.DataFrame(km_huang.cluster_centroids_)

clusterCentroidsDf.columns=bank_cust.columns

clusterCentroidsDf



#%%







cost=[]
for num_clusters in list(range(1,7)):
    kmode= KModes(n_clusters=num_clusters, init= "Cao", n_init= 1, verbose= 1)
    kmode.fit_predict(bank_cust)
    cost.append(kmode.cost_)


y = np.array([i for i in range(1,7,1)])

plt.plot(y,cost);

#Numero de clusteres optimos: 3

#%%

cost=[]
for num_clusters in list(range(1,7)):
    kmode= KModes(n_clusters=num_clusters, init= "Huang", n_init= 1, verbose= 1)
    kmode.fit_predict(bank_cust)
    cost.append(kmode.cost_)

y = np.array([i for i in range(1,7,1)])

plt.plot(y,cost);

#Numero de clusteres optimos: 3


#Diego
#####################################################3
#%%

#Asignando clusteres para poder visualizarlos por variable
km_cao= KModes(n_clusters=3, init="Cao", n_init=1, verbose=1)
fitClusters_cao= km_cao.fit_predict(bank_cust)
fitClusters_cao


#%%
#Combinando los clusteres predichos con el DF original

clustersDf= pd.DataFrame(fitClusters_cao)
clustersDf.columns=["cluster_preddicted"]
combineDf= pd.concat([bank_cust_copy, clustersDf], axis=1).reset_index()

#%%

#
combineDf = combineDf.drop(["index"], axis=1)

combineDf.head()

#%%
#Identificacion del Cluster

cluster_0= combineDf[combineDf["cluster_preddicted"]==0]
cluster_1=combineDf[combineDf["cluster_preddicted"]==1]
cluster_2=combineDf[combineDf["cluster_preddicted"]==2]

cluster_0.info()
cluster_1.info()
cluster_2.info()

###############################################################

#%%
#SALIDA
#Grafico de la variable Job

plt.subplots(figsize=(15,5))

sns.countplot(x=combineDf["age_bin"], order=combineDf["age_bin"].value_counts().index, hue=combineDf["cluster_preddicted"])

plt.show()

#De ahi estudias todas las variables, solo cambias el nombre

#%%


def ploteo(variable,variable2):
    plt.subplots(figsize=(15,5))

    sns.countplot(x=combineDf[variable], order=combineDf[variable].value_counts().index, hue=combineDf[variable2])

    plt.show()

#%%

def ploteo(variable):
    plt.subplots(figsize=(15,5))

    sns.countplot(x=combineDf[variable], order=combineDf[variable].value_counts().index, hue=combineDf["cluster_preddicted"])

    plt.show()

    

sw = 1

while sw:
     print("MENU")
     print(".- Aplicando clustering... ")
     print(".- Aplicando clustering.. ")
     print(".- Aplicando clustering. ")
     print(".1 Analizar variable 'Age'--- ")
     print(".2 Analizar variable 'Job'---")
     print(".3 Analizar variable 'Marital'---")
     print(".4 Analizar variable 'Education'---")
     print(".5 Analizar variable default'---")
     print(".6- Salir")
     op = int(input("Opción (1-6): "))
     if  1<=op<=6:
          if op == 1:
              ploteo("age_bin")
          elif op == 2:
               ploteo("job")
          elif op == 3:
              ploteo("marital")
          elif op == 4:
              ploteo("education")
          elif op == 5:
              ploteo("default")
          else:
               print("Se activó Salir")
               sw = 0
     else:
          print("Error")
     


print("Hasta la vista ...baby...!!!")




