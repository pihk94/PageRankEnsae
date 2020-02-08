# PageRankEnsae


# Eléments logiciels pour le traitement des données massives
##### Melchior Prugniaud MS DS


La machine utilisé est une machine avec 32GB de RAM avec une carte graphique NVIDIA GeForce GTX 1080. Le processeur est un AMD Ryzen 7 1700 3GHz avec 8 coeurs. Au départ, ma volonté était de réaliser le projet sur Python puis de voir la différence avec PyCuda mais celà n'a pas fonctionné car mon installation semble avoir des problèmes avec la détection des librairies de C/C++. C'est pourquoi, le projet démarre sur python puis migre sur C pour enfin terminer sur un rendu en Cuda. 

# PageRank

### 1. Présentation de l'algorithme

PageRank est un algorithme inventé par les co-fondateurs de Google, Larry Page et son partenaire Sergey Brin. Il est utilisé pour analyser les liens d'un réseau internet ou d'un site (Ex : Twitter et les liens entre les différents utilisateurs). PageRank va prendre l'ensemble de ce réseau pour assigner un poids à chaque noeud ayant un poids montrant son importance dans le réseau. L'algorithme est utilisé par Google dans son moteur de recherche comme l'une des variables déterminant le rang d'apparition de la page.

Il représente la vraisemblance qu'une personne cliquant sur des liens puisse arriver à une page déterminée grâce à la sortie de l'algorithme qui est une distribution de probabilité. Il est possible de représenter un réseau à l'aide d'un graphe dirigé. En effet, prenons l'exemple issu de Wikipédia. Soit un réseau de quatre pages A, B, C et D avec B allant vers A et C. C allant vers A, A allant vers B et D allant vers les trois autres pages. Le graphique ci-dessous est la représentation de ce réseau. Il est aussi possible de représenter ce réseau sous la forme d'une matrice qui sera utilisé par la suite pour calculer les poids des différentes pages.


```python
G=nx.DiGraph()
G.add_nodes_from(['A','B','C','D'])
G.add_edges_from([('A','B'),('B','A'),('B','C'),('C','A'),('D','A'),('D','B'),('D','C')])
nx.draw(G, with_labels=True, font_weight='bold', arrows=True)
plt.show()
```
    


![png](output_2_1.png)


La matrice représente notre réseau avec en ligne les liens entrants vers la page X et en colonne les liens sortants vers la page X. A noter que par exemple pour la page D allant vers A, B et C, les liens sortants auront une valeur de 1/3 dans la matrice. On obtient alors :


    array([[0.        , 0.5       , 1.        , 0.33333333],
           [1.        , 0.        , 0.        , 0.33333333],
           [0.        , 0.5       , 0.        , 0.33333333],
           [0.        , 0.        , 0.        , 0.        ]])



A partir de ces répresentations, on peut plus facilement calculer pour ce léger exemple le pagerank de A. Sachant qu'au départ nous avons initialisé les poids de tel manière tel qu'au départ leur valeur soit de 1/N donc 1/4. 

![$ PR(A) = \frac{PR(B)}{2} + PR(C) + \frac{PR(D)}{3} $](https://render.githubusercontent.com/render/math?math=%24%20PR(A)%20%3D%20%5Cfrac%7BPR(B)%7D%7B2%7D%20%2B%20PR(C)%20%2B%20%5Cfrac%7BPR(D)%7D%7B3%7D%20%24)

Ainsi à la première itération, nous avons : 

![$ PR(A) = \frac{0.25}{2} + 0.25 + \frac{0.25}{3} = 0.458 $](https://render.githubusercontent.com/render/math?math=%24%20PR(A)%20%3D%20%5Cfrac%7B0.25%7D%7B2%7D%20%2B%200.25%20%2B%20%5Cfrac%7B0.25%7D%7B3%7D%20%3D%200.458%20%24)


De manière plus général, il est possible d'écrire : 


![$ PR(A) = \frac{PR(B)}{L(B)} + \frac{PR(C)}{L(C)} + \frac{PR(D)}{L(D)}$](https://render.githubusercontent.com/render/math?math=%24%20PR(A)%20%3D%20%5Cfrac%7BPR(B)%7D%7BL(B)%7D%20%2B%20%5Cfrac%7BPR(C)%7D%7BL(C)%7D%20%2B%20%5Cfrac%7BPR(D)%7D%7BL(D)%7D%24)


Où L(N) représente le nombre de liens sortant de la page N. 


Ainsi, nous obtenons : 


![$ PR(u) = \sum_{v \in B_u}\frac{PR(v)}{L(v)} $](https://render.githubusercontent.com/render/math?math=%24%20PR(u)%20%3D%20%5Csum_%7Bv%20%5Cin%20B_u%7D%5Cfrac%7BPR(v)%7D%7BL(v)%7D%20%24%20)



Avec $ B_u $ l'ensemble des pages liant à u.

Etant donné que le PageRank simule la navigation d'une personne cliquant aléatoirement sur les différents liens, on instaure la probabilité d qu'une personne continue sa navigation est appelé 'dampling factor'. Différentes études on réussit à déterminer que le dampling factor optimal était 0.85 et nous allons donc définir d = 0.85 dans le reste de ce notebook. Au final, en ajoutant le dampling factor à la formule précédente on obtient : 


![$PR(p_i) = \frac{1-d}{N}+d \sum_{p_j \in M(p_i)}\frac{PR(p_j)}{L(p_j)}$](https://render.githubusercontent.com/render/math?math=%24PR(p_i)%20%3D%20%5Cfrac%7B1-d%7D%7BN%7D%2Bd%20%5Csum_%7Bp_j%20%5Cin%20M(p_i)%7D%5Cfrac%7BPR(p_j)%7D%7BL(p_j)%7D%24)



Avec N le nombre de page et $ M(p_i) $ l'ensemble de page connectant à $ p_i $. La somme des $ PR(p_i) $ est alors égale à 1.


Comme montré précédemment, il est possible d'écrire cette équation de manière matricielle. 

![$ R = \begin{bmatrix} \frac{(1-d)}{N} \\ \frac{(1-d)}{N} \\... \\ \frac{(1-d)}{N} \end{bmatrix} + d \begin{bmatrix} l(p_1,p_1) & l(p_1,p_2)  & ... & l(p_1,p_N)\\ l(p_2,p_1) & l(p_2,p_2)  & ... & l(p_2,p_N)\\ ... & ... & ... & ... \\ l(p_N,p_1) & l(p_N,p_2)  & ... & l(p_N,p_N)\\\end{bmatrix} R $](https://render.githubusercontent.com/render/math?math=%24%20R%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B(1-d)%7D%7BN%7D%20%5C%5C%20%5Cfrac%7B(1-d)%7D%7BN%7D%20%5C%5C...%20%5C%5C%20%5Cfrac%7B(1-d)%7D%7BN%7D%20%5Cend%7Bbmatrix%7D%20%2B%20d%20%5Cbegin%7Bbmatrix%7D%20l(p_1%2Cp_1)%20%26%20l(p_1%2Cp_2)%20%20%26%20...%20%26%20l(p_1%2Cp_N)%5C%5C%20l(p_2%2Cp_1)%20%26%20l(p_2%2Cp_2)%20%20%26%20...%20%26%20l(p_2%2Cp_N)%5C%5C%20...%20%26%20...%20%26%20...%20%26%20...%20%5C%5C%20l(p_N%2Cp_1)%20%26%20l(p_N%2Cp_2)%20%20%26%20...%20%26%20l(p_N%2Cp_N)%5C%5C%5Cend%7Bbmatrix%7D%20R%20%24)


Avec ![$ \sum_{i=1}^{N} l(p_i,p_j) = 1$](https://render.githubusercontent.com/render/math?math=%24%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20l(p_i%2Cp_j)%20%3D%201%24)



### 2.) Première implémentation

Dans un premier temps, nous allons implémenter l'algorithme sous python entièrement. Pour vérifier son bon fonctionnement, l'algorithme sera d'abord testé sur notre matrice m puis ensuite sur un jeu de données email.txt. Il reste assez petit, d'autres jeux de données plus grand peuvent être utilisés par la suite.
Les données sont tirés de https://snap.stanford.edu/data/. 

Dans un premier temps reprenons notre premier exemple et voyons les résultats que l'on peut obtenir. Cela nous permettra aussi de voir si nos différentes fonctions rendent les mêmes résultats.

Voici une version de l'algorithme utilisant la forme matricielle pour calculer le PageRank




    matrix([[0.        , 0.5       , 1.        , 0.33333333],
            [1.        , 0.        , 0.        , 0.33333333],
            [0.        , 0.5       , 0.        , 0.33333333],
            [0.        , 0.        , 0.        , 0.        ]])


    array([[0, 1, 1, 1],
           [1, 0, 0, 1],
           [0, 1, 0, 1],
           [0, 0, 0, 0]])



On obtiens donc le PageRank suivant pour 10 itérations. La somme des différents PR fait bien un. Nous pouvons remarquer que les différents coefficients des PageRank ne varie presque pas pour notre cas au dela de 10 itérations.


```python
PR = PageRank(m).main()
print(PR, PR.sum())
```

    0    0.219240
    1    0.249702
    2    0.175232
    3    0.355827
    dtype: float64 0.9999999999999996
    


```python
PageRank(m,100).main()
```


    0    0.219237
    1    0.249705
    2    0.175231
    3    0.355828
    dtype: float64



Désormais, nous allons nous intéresser au jeu de données email.txt et mesurer le temps d'éxecution pour ce jeu de données. Pour se faire nous créeons une fonction basique permettant à partir d'un DataFrame d'obtenir la matrice nécessaire aux calculs du PageRank. Pour allons ensuite mesurer son temps d'exécution sur les données à la fois de Wikipédia.


v = csv_to_PR('email.txt',' ',max_iter =10)
v.sum()

    0.9999999999999973



Je reprends une fonction créée par vous même permettant de mesurer le temps d'exécution moyen.
Nous réalisons donc un comparatif des performances pour l'ensemble des trois datasets que nous avons sélectionnés.


![png](output_17_1.png)

### 3.) Implémentation sous C

Dans un premier temps, j'ai réalisé un fichier en C++ permettant de réaliser le PageRank. Il prend en entrée un fichier texte à renseigner, demande ensuite le nombre maximale d'itérations choisis et réalise le PageRank. A noter qu'il faut manuellement renseigner le nombre de sommets du fichier. Une version sans cette manipulation est disponible dans le fichier 'PageRankAuto.cpp' mais étant donné que je n'arrivais pas bien à implémenter les vecteurs avec Cuda, la version qui sert de comparatif avec cuda est 'PageRank.cpp'. 

La base issu du fichier email.txt se compose de 1004 sommets pour un total de 25 571 arrêtes. En utilisant le code réalisé sous Python, on remarque un temps d'exécution d'environ 250 simillisecondes. Avec notre code en C++, le temps d'exécution est de xx millisecondes.
Sur la base email.txt, le temps de traitement est de 409.753 millisecondes. 


### 4.) Implémentation sous Cuda

La partie qui prends le plus de temps a s'exécuter avec les grands fichiers est la partie concernant la multiplication matricielle. C'est cette partie qui est parallélisé. 


### 5.) Conclusion

A venir, j'ai encore des soucis de logique dans l'implémentation du code sous cuda du fichier PageRankCuda.cu. En effet, le fichier ne fonctionne que pour le pour la base de test se composant uniquement de 4 sommets. Pour celle des emails, j'ai des soucis dans le choix du nombre de blocks et de thread. En effet, dans ma fonction mul il semblerait que je n'affecte pas bien mes boucles qui dépendent justement de ce nombre de blocks et thread. Cela engendre des erreurs que je suis entrain de corriger... 

