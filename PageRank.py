import numpy as np
import pandas as pd
import math
import sys
class PageRank:
    def __init__(self,m,max_iter = 10, d=0.15, e=0.00001,Verbose=False):
        self.d = d
        self.e = e
        self.max_it = max_iter
        self.Verbose=Verbose
        self.transitionWeights = pd.DataFrame(m)
    def ExtraireSommets(self,m):
        sommets = set()
        for colKey in m:
            sommets.add(colKey)
        for rowKey in m.T:
            sommets.add(rowKey)
        return sommets
    def norm(self,m):
        return m.div(m.sum(axis=1), axis=0)
    def MatriceCarree(self,m, sommets):
        m = m.copy()
        
        def ColonnesManquantes(m):
            for sommet in sommets:
                if not sommet in m:
                    m[sommet] = pd.Series(0, index=m.index)
            return m
        m = ColonnesManquantes(m) 
        m = ColonnesManquantes(m.T).T 
        return m.fillna(0)
    def CheckLignePositive(self,m):
        m = m.T
        for colKey in m:
            if m[colKey].sum() == 0.0:
                m[colKey] = pd.Series(np.ones(len(m[colKey])), index=m.index)
        return m.T
    def euclideanNorm(self,series):
        return math.sqrt(series.dot(series))
    def Initialisation(self,sommets):
        startProb = 1.0 / float(len(sommets))
        return pd.Series({node : startProb for node in sommets})
    def RandomSurfer(self,sommets, transitionProbabilities, rsp):
        alpha = 1.0 / float(len(sommets)) * self.d
        return transitionProbabilities.copy().multiply(1.0 - self.d) + alpha
    def main(self):
        ma = self.transitionWeights
        sommets = self.ExtraireSommets(ma)
        ma = self.MatriceCarree(ma, sommets)
        ma = self.CheckLignePositive(ma)
        state = self.Initialisation(sommets)
        transitionProbabilities = self.norm(ma)
        transitionProbabilities = self.RandomSurfer(sommets, transitionProbabilities, self.d)
        for iteration in range(self.max_it):
            oldState = state.copy()
            state = state.dot(transitionProbabilities)
            delta = state - oldState
            if self.euclideanNorm(delta) < self.e:
                if self.Verbose ==True:
                    print("Pas de changement après l itération numéro : ", iteration)
                break
        return state
def csv_to_graph(file_name,delim,header=False):
    col=[]
    maxi=0
    with open(file_name,'r') as f:
        lignes = f.readlines()
        j=0
        for ligne in lignes:
            j+=1
            ligne = ligne.replace('\n','')
            cols = ligne.split(delim)
            if header == True:
                header = False
                continue
            col+=[(int(cols[0]),int(cols[1]))]
            if maxi < int(cols[0]):
                maxi = int(cols[0])
            elif maxi < int(cols[1]):
                maxi = int(cols[1])
    return col,maxi+1
def csv_to_PR(file_name,delim,max_iter =10,Verbose=True):    
    col,maxi = csv_to_graph(file_name,delim,header=False)
    m=np.zeros(maxi**2).reshape(maxi,maxi)
    for i in col:
        m[i[0]][i[1]]=1
    m = np.matrix(m)
    return PageRank(m,max_iter =max_iter).main()