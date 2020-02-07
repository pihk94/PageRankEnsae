#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include <string>
#include <ctime>
#include<algorithm>
#include <vector>
#define d 0.85
#define epsilon 0.00000001
#define taille 100000000
#include <time.h>
using namespace std;
using namespace std::chrono;
time_t debut,fin;
int main(){
    // On ouvre le fichier txt pour le parser
    ifstream graph;
    char chemin[500];
    int max_iter;
    cout << "Entrer le chemin entier avec des doubles backslashs"<< endl;
    cin >> chemin;
    cout << "\nCombien d iterations maximale ?";
    cin >> max_iter;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    time(&debut);
    graph.open(chemin);
    vector<int> from;
    vector<int> to;
    string word,line;
    // On va parser en fonction du separateur nos from/To
    if (!graph.is_open())
    {
        cout<<"Ouverture impossible";
        return false;
    }
    while (getline(graph,line))
    {   

        string espace;
        std::istringstream text_stream(line);
        text_stream >> word;
        from.push_back(stoi(word));
        getline(text_stream,espace,' ');
        text_stream >> word;
        to.push_back(stoi(word));
    }
    graph.close();
    cout << "Ouverture du fichier texte reussi" << endl;
    // calcul du maximum pour construire la matrice carree
    double maxiFrom = *max_element(from.begin(), from.end());
    double maxiTo = *max_element(to.begin(), to.end());
    int maxval;
    if (maxiFrom < maxiTo){
        maxval = maxiTo+1;
    }
    else
    {
        maxval = maxiFrom+1;
    }
    //creation d un vecteur 2 dim pour creer une matrice carree.
    vector<vector<float> > matrix(maxval,vector<float>(maxval));
    matrix.resize(maxval,vector<float>(maxval,0));
    int idx_i,idx_j;
    // Pour chaque noeud entrant on met 1 a lindice
    for (int i = 0; i < maxval*2 -1;i++)
    {
        idx_i  = from[i];
        idx_j  = to[i];
        matrix[idx_j][idx_i]=1;
    }
    float sum;
    // On normalise la matrice pour tenir compte du nb de pages linkantes
    for(int i =0;i<maxval;i++)
    {
        sum = 0;
        for (int j=0;j<maxval;j++)
        {
           if (matrix[i][j] >0){
                sum++;
           }
           
        } 
        if (sum == 0){
            sum = maxval;
            for (int j = 0;j<maxval;j++)
            {
                matrix[i][j] =1;
            }
        }
        for (int j=0;j<maxval;j++)
        {
            matrix[i][j]=matrix[i][j]/sum;
        } 
    }
    // Debut de l algorithme PageRank
    float maxval2 = maxval;
    // MarcheAleatoire
    vector<float> state(maxval2,(1/maxval2));
    float alpha =((1-d)/maxval);
    for (int i = 0;i<maxval;i++)
    {
        for (int j =0;j<maxval;j++)
        {
            matrix[i][j]=matrix[i][j]*d +alpha;
        }
    }
    // Matrice sauvegarde
    vector<float> old(maxval2);
    vector<float> delta(maxval2);
    vector<float> temp(maxval2);
    // iterations
    for (int i = 0;i<=max_iter;i++)
    {
        float check = 0;
        old = state;
        for (int j=0;j<maxval;j++)
        {
            float calc=0;
            for (int h = 0;h<maxval;h++)
            {
                calc+= matrix[h][j]*state[h];
            }
            temp[j] = calc;
        }
        for (int j = 0;j<maxval;j++)
        {
            state[j]=temp[j];
        }
        for (int h = 0;h<maxval;h++)
        {
            delta[h] = state[h] - old[h];
        }
        for (int h = 0;h<maxval;h++)
        {
            check+=delta[h]*delta[h];
        }
        if (check < epsilon)
        {
            cout << "On quitte la boucle apres "<< i<< " iterations.\n";
            break;
        }
    }
    // output
    ofstream out;
    out.open("C:\\Users\\Prugniaud\\projects\\output.txt");
    for(int i =0;i<maxval;i++)
    {
        out << to_string(state[i]); 
        out<< '\n';
    }
    out.close();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double, std::milli> time_span = t2 - t1;

    time(&fin);
    cout << "Temps d execution : "<<difftime(fin,debut)<<" secondes"<<endl;
    cout << "Temps d execution : "<<time_span.count()<<" millisecondes"<<endl;
    cout << "Fin du programme, Appuyez sur Entrer pour quitter" << endl;
	getchar();
    return 0;
}
