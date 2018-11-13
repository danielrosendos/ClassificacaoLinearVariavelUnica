import pandas as pd
import random
import numpy as np
import math

def processamentoDados():
    #Lendo os Arquivos
    dataSet = pd.read_csv('data.csv', sep=',', header=None).values
    random.shuffle(dataSet)

    #Salvando nas Variáveis
    Atributos = len(dataSet[0])-1
    X = dataSet[:, 0:Atributos]
    Y = dataSet[:, Atributos]

    #Separando Treino e Teste
    treinamento = 80
    treinamento = round(len(X)*treinamento/100)

    #Variaveis Treino Teste
    X_treinamento = X[0:treinamento, :]
    Y_treinamento = Y[0:treinamento]
    X_teste = X[treinamento:, :]
    Y_teste = Y[treinamento:]

    return X_treinamento, X_teste, Y_treinamento, Y_teste

def pi(Y):
    return (1/len(Y))*Y.sum()

def mis(X, Y):
    return ((1 - Y).dot(X[:, 0:len(X[0])])) / sum((1 - Y)), (Y.dot(X[:, 0:len(X[0])])) / sum(Y)

def sigma(X,Y, m0, m1):
    #inicialização do sigma
    sigma_j = np.zeros(len(X[0]))

    #calculo de sigma
    sigma_j[0:len(X[0])] = ((Y.dot((X[:, 0:len(X[0])]-m1[0:len(X[0])])**2)).sum() + ((1-Y).dot((X[:, 0:len(X[0])]-m0[0:len(X[0])])**2)).sum())/len(X)

    #matriz covariancia
    sigma_j = np.diag(sigma_j)

    return sigma_j

def funBeta(sigma, m0, m1):
    return np.linalg.inv(sigma).dot(m1-m0)

def funGama(sigma, m0, m1, Pi):
    return -1/2 * ((m1-m0).T).dot(np.linalg.inv(sigma).dot(m1+m0)) + math.log(Pi/(1-Pi))

def calcProbabilidade(beta, gama, X_teste, Y_teste):
    Y_hat = np.zeros(len(X_teste))
    acerto = 0
    p_x = np.zeros(len(X_teste))
    for i in range(len(X_teste)):
        p_x[i] = (1/(1+math.exp((-beta.T).dot(X_teste[i])-gama)))
        if p_x[i] >= 0.5:
            Y_hat[i] = 1
        if Y_teste[i] == Y_hat[i]:
            acerto += 1
    return acerto

def main():
    #Pegando os dados
    X, X_teste, Y, Y_teste = processamentoDados()

    #Valor Pi
    Pi = pi(Y)

    #calculando mi
    m0, m1 = mis(X,Y)

    #calculando sigma
    sigma_j = sigma(X,Y, m0, m1)

    #calc beta e gama
    beta = funBeta(sigma_j, m0, m1)
    gama = funGama(sigma_j, m0, m1, Pi)

    #calcular probabilidade
    acerto = calcProbabilidade(beta, gama, X_teste, Y_teste)

    print('Taxa de Acerto de: {:.2f}%'.format(acerto/len(X_teste)*100))

if __name__ == '__main__':
    main()