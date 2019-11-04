# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:21:28 2019

@author: MateusMarcuzzo
"""
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def pure_zipf_plot(s,alphabet_size):

    x_range = np.arange(1,alphabet_size+1)
    
# like the octave code:   
#                %First we are going Zipf like Painsky's code
#                q=P;
#                n=K; %dimension of the random vector 
#                %Zipf distribution
#                s=1.05; %zipf distribution parameter
#                zipf_p=[1:1:q^n];
#                zipf_p=1./(zipf_p.^s);
#                zipf_p=zipf_p'/sum(zipf_p);
#    
    
    
    zipf_p = np.array([l for l in np.arange(1,alphabet_size+1)])
    zipf_p = 1/zipf_p**s
    zipf_p = zipf_p/zipf_p.sum()
    
    print(zipf_p)
    
    
    y_range = zipf_p

    
    plt.plot(x_range,y_range,'o')
    plt.vlines(x_range,0,y_range)
    
    plt.title('PMF da Zipf pura\n Alfabeto de tamanho {}, s={}'.format(alphabet_size,s))
    plt.xlabel('Símbolo')
    
    s = str(s)
    s = s.replace('.','_')
    plt.savefig('distribuicoes\\zipf_s{}_{}.pdf'.format(s,alphabet_size))
    
    plt.show()
    
pure_zipf_plot(1.05,27)

def shuffled_zipf_plot(s,alphabet_size):
    x_range = np.arange(1,alphabet_size+1)
    
        
    zipf_p = np.array([l for l in np.arange(1,alphabet_size+1)])
    zipf_p = 1/zipf_p**s
    zipf_p = zipf_p/zipf_p.sum()
    
    y_range = zipf_p
    
    np.random.shuffle(y_range)
    
    plt.plot(x_range,y_range,'o')
    plt.vlines(x_range,0,y_range)
    
    plt.title('PMF de uma Zipf embaralhada\n Alfabeto de tamanho {}, s={}'.format(alphabet_size,s))
    plt.xlabel('Símbolo')
    
    s = str(s)
    s = s.replace('.','_')
    plt.savefig('distribuicoes\\shuffledzipf_s{}_{}.pdf'.format(s,alphabet_size))
    
    
    
    plt.show()
    
shuffled_zipf_plot(1.05,25)


def random_pmf_plot(alphabet_size):
    y_range = np.random.random(alphabet_size)
    y_range = y_range/y_range.sum()
    x_range = np.arange(1,alphabet_size+1)
    
    plt.plot(x_range,y_range,'o')
    plt.vlines(x_range,0,y_range)
    
    plt.title('PMF aleatória\n Alfabeto de tamanho {}'.format(alphabet_size))
    plt.xlabel('Símbolo')
    plt.show()
    
random_pmf_plot(8)


def binomial_pmf_plot(alphabet_size,psucc):
    x_range = np.arange(1,alphabet_size+1)
    
    y_range = scipy.stats.binom.pmf(x_range-1,alphabet_size-1,psucc)
    
    plt.plot(x_range,y_range,'o')
    plt.vlines(x_range,0,y_range)
    
    plt.title('PMF de uma binomial \n Alfabeto de tamanho {}, p={}'.format(alphabet_size,psucc))
    plt.xlabel('Símbolo')
    
    psucc = str(psucc)
    psucc = psucc.replace('.','')
    plt.savefig('distribuicoes\\binomial_psucc{}_{}.pdf'.format(psucc,alphabet_size))
    
    
    
    plt.show()
    
binomial_pmf_plot(2**4,0.2)  
binomial_pmf_plot(2**4,0.5)
binomial_pmf_plot(2**4,0.8)