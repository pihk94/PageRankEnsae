#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include <string>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<algorithm>
#include <vector>
#define d 0.85
#define epsilon 0.00000001
#include <time.h>
using namespace std;
using namespace std::chrono;
time_t debut,fin;
__global__ void mul(float *matrice,float *state,int maxval)
{
    int idx=blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.x;
    int tx = blockIdx.x;
    float resultat;
    if (idx < maxval)
    {
        for (int i =0;i<maxval;i++)
        {
            resultat+=state[i]*matrice[maxval*i+ty];
        }
        state[idx]=resultat;
    }
}
int main(){
    const int n = 4;
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
    float* matrice = new float[n*n];
    for (int i = 0;i<maxval*maxval;i++)
    {
        matrice[i]=0;
    }
    int idx_i,idx_j;
    for (int i =0;i<maxval*2 -1;i++)
    {
        idx_i  = from[i];
        idx_j  = to[i];
        matrice[(idx_j)*maxval+idx_i]=1;
    }
    float sum;
    // On normalise la matrice pour tenir compte du nb de pages linkantes
    for (int i = 0;i<maxval;i++)
    {
        sum = 0;
        for (int j =0;j<maxval;j++)
        {
            if (matrice[i*maxval+j]>0)
            {
                sum++;
            }
        }
        if (sum ==0)
        {
            sum = maxval;
            for(int j = 0;j<maxval;j++)
            {
                matrice[i*maxval+j]=1;
            }
        }
        for(int j = 0;j<maxval;j++)
        {
            matrice[i*maxval+j]/=sum;
        }
    }

    // Debut de l algorithme PageRank
    float maxval2 = maxval;
    float state[n];
    float old[n];
    float delta[n];
    //float temp[n];
    float alpha =((1-d)/maxval);
    float* d_matrix;
    float* d_state;

    for (int i = 0;i<maxval;i++)
    {
        state[i] = 1/maxval2;
    }
    for (int i =0;i<maxval*maxval;i++)
    {
        matrice[i]=matrice[i]*d+alpha;
    } 
    cudaMalloc((void**)&d_matrix,n*n*sizeof(float));
    cudaMalloc((void**)&d_state,n*sizeof(float));
    cudaMemcpy(d_matrix,matrice,n*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_state,state,n*sizeof(float),cudaMemcpyHostToDevice);
    dim3 dimGrid(1,1);
    dim3 dimBlock(n,n);

    for (int i = 0;i<=max_iter;i++)
    {
        float check = 0;
        for (int j = 0;j<maxval;j++)
        {
            old[j] = state[j];
        }
        mul<<<4,4>>>(d_matrix,d_state,n);
        cudaDeviceSynchronize();
        cudaMemcpy(state,d_state,n*sizeof(float),cudaMemcpyDeviceToHost);
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
    cudaFree(d_matrix);
    cudaFree(d_state);
    cout << "Temps d execution : "<<difftime(fin,debut)<<" secondes"<<endl;
    cout << "Temps d execution : "<<time_span.count()<<" millisecondes"<<endl;
    cout << "Fin du programme, Appuyez sur Entrer pour quitter" << endl;
    getchar();
    return 0;
}
