# -*- coding: utf-8 -*-
"""
Created on Wed Oct  11 11:50:50 2019

@author: MateusMarcuzzo
"""

#Directly inspired on data_extractor_bss.py

import numpy as np
from scipy.io import loadmat
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import t

pd.set_option('display.max_rows', 500)

# The colors were too close
# plt.style.use('ggplot')
plt.style.use('seaborn-bright')

# If true, generates many plots
do_plot = False

# It's really useful for getting the right indices on .loc usage
idx = pd.IndexSlice 

# 21/09/2019
# we are importing an .mat file from Octave.
# we must do rows with it, extracting the important variables.

important_variables = ['some_primes','n_sources','n_samples','algorithms_names','n_trials',]
                      
# to be used later to build a dataframe                     
rows_list = []


## Possible files names:
# ICA_shuffled_zipf.mat
# ICA_random_pmf.mat
# ICA_pure_zipf.mat
# ICA_pure_zipf_10trials.mat
# ICA_binomialp02.mat
# ICA_binomialp03.mat
# ICA_binomialp04.mat
# ICA_binomialp05.mat
# ICA_binomialp06.mat
# ICA_binomialp07.mat
# ICA_binomialp08.mat

      
setups_names = ['ICA_shuffled_zipf.mat',
                'ICA_random_pmf.mat',
                'ICA_pure_zipf.mat',
                'ICA_pure_zipf_10trials.mat',
                'ICA_binomialp02.mat',
                'ICA_binomialp03.mat',
                'ICA_binomialp04.mat',
                'ICA_binomialp05.mat',
                'ICA_binomialp06.mat',
                'ICA_binomialp07.mat',
                'ICA_binomialp08.mat']

for some_setup in setups_names:
    
    ## Here we load some ofthe above setup
    a_setup = loadmat('most_recent_sim_data/{}'.format(some_setup))
    
    # note that these a_setup[a_variable] are 2-D, we must flatten them.
    for a_variable in important_variables:
        a_setup[a_variable] = a_setup[a_variable].flatten()
        #just for checking
       # print(a_setup[a_variable])
    
    # Octave returns a matrix with just one element, here we correct this to a scalar
    a_setup['n_trials'] = int(np.min(a_setup['n_trials']))
    n_trials = a_setup['n_trials']
    
    # The columns which are data collected
    data_variables = ['trial_time','total_corr_results']
    
    # To check for index of an array:
    # itemindex = numpy.where(array==item)
    # check: https://stackoverflow.com/questions/432112/is-there-a-numpy-function-to-return-the-first-index-of-something-in-an-array#23994923
            
    
    # here I'm removing [' and '] from the strings and turning it into a list.
    a_setup['algorithms_names'] = [str(algorithm_name)[2:-2] for algorithm_name in a_setup['algorithms_names']]
    
    #trying to use this:
    # https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe/17496530#17496530
    # In a such a way that with the loadmat data, we can build a pandas dataframe
    
    for row in product(a_setup['some_primes'],a_setup['n_sources'],a_setup['n_samples'],a_setup['algorithms_names']):
        temp_dict = {}
        
        #just for checking
        #print(row)
        # I just want the minimum index. without it, returns an array with just
        # one element. that's why I use np.min
        tpi = the_prime_index = np.min(np.where(a_setup['some_primes'] == row[0]))
        temp_dict.update({'prime':row[0]})
    
        
        tnsoi = the_n_sources_index = np.min(np.where(a_setup['n_sources'] == row[1]))
        temp_dict.update({'dimension':row[1]})
    
        
        tnsai = the_n_samples_index = np.min(np.where(a_setup['n_samples'] == row[2]))
        temp_dict.update({'n_samples':row[2]})
    
        
        # It's not an np.array, its a list!
        talgi = the_algorithm_index = np.min(a_setup['algorithms_names'].index(row[3]))
        temp_dict.update({'algorithm_name': row[3]})
    
        # removing ICA_ from the name and the trailing .mat
        if(some_setup != 'ICA_pure_zipf_10trials.mat' ):
            temp_dict.update({'distribution': str(some_setup)[4:-4]})
        else:
            temp_dict.update({'distribution':str(some_setup)[4:-13]})
        
        # I've put some .copy() at the end, and It seems that it worked
        # It was giving some problems without it
        for trial in range(n_trials):
            temp_dict['total_corr'] = a_setup['total_corr_results'][(tpi,tnsoi,tnsai,talgi,trial)].copy()
            
            temp_dict['trial_time'] = a_setup['trial_time'][(tpi,tnsoi,tnsai,talgi,trial)].copy()
            
            rows_list.append(temp_dict.copy())

            

        
#just checking    
#for row in rows_list:
#    print(row)


# Here the dataframe is created
df = pd.DataFrame(rows_list)

# we change this column to  category
df['algorithm_name'] = df['algorithm_name'].astype('category')
df['distribution'] = df['distribution'].astype('category')


# and these to integer values (since they are), they come as float
int_variables = ['n_samples','dimension','prime']
for a_variable in int_variables:
    df[a_variable] = df[a_variable].astype('int')

#print(df)
#
### Above we have some examples.
#
## it shows some stuff we already know on matlab/Octave:
#print(df.groupby(['prime','n_sources','n_samples','algorithm_name','distribution']).mean())
#print(df.groupby(['prime','n_sources','n_samples','algorithm_name','distribution']).count())
#
#print(df.dtypes)
#    
#
### Some interesting commands
dfgroupby = df.groupby(['prime','dimension','n_samples','algorithm_name','distribution']) 

#drops the cases with NaN values.



dfgroupby_total_corr = dfgroupby['total_corr']
dfgroupby_trial_time = dfgroupby['trial_time']
#print(dfgroupby)

## plot() or hist() are really slow. just for testing
# print(dfgroupby.plot())
# print(dfgroupby.hist())

# Must never took the parenthesis off. & operator has a higher precedence,
# so It'll thrown an error without the parenthesis.
# print(df[(df.prime==3) & (df.n_sources == 4) & (df.n_samples == 4096) & (df.algorithm_name == 'sa4ica')])

print('primes evaluated')
print(df.prime.unique())
print('dimensions evaluated')
print(df.dimension.unique())
print('n_samples evaluated')
print(df.n_samples.unique())
print('the algorithm_names')
print(list(df.algorithm_name.unique() ) )

# https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(Shafer_and_Zhang)/06%3A_Sampling_Distributions/6.1%3A_The_Mean_and_Standard_Deviation_of_the_Sample_Mean
# pág 191 do Inferência Estatística do Casella (07/10/2019)
#print('desvio padrão amostral da média ou')
#print('erro padrão (não achei referência suficiente para esta nomeação)')
#print('sample standard deviation')
#print('Standard Deviation of the Sample Mean ')
#print(dfgroupby.std()/np.sqrt(dfgroupby.count()))







# gráfico interessante 
#dfgroupby.total_corr.mean().plot.barh(legend=True)
#plt.show()


# outro gráfico interessante:
#https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
#means= dfgroupby.mean()
#
#errors=dfgroupby.std()/np.sqrt(dfgroupby.count())
#
#means.plot.bar(yerr=errors)

# Somente o caso de BSS:
#bss_succ_means = dfgroupby.bss_succ.mean()
#
#bss_succ_errors = dfgroupby.bss_succ.std()/np.sqrt(dfgroupby.bss_succ.count())
#
#bss_succ_means.plot.bar(yerr=bss_succ_errors)

## cada caso:
#important_columns = ['bss_succ','total_corr','trial_time']
#
#for imp_val in important_columns:
#    means = dfgroupby[imp_val].mean()
#    errors = dfgroupby[imp_val].std()/np.sqrt(dfgroupby[imp_val].count())
#    
#    means.plot.barh(yerr=errors,legend=True,logx=True)
#    plt.show()
    
# Outros exemplos interessantes:
    # obs: a_setup = loadmat('most_recent_sim_data/diff2_setup1.mat')
# dfgroupby.mean().loc[2,8,256,'GLICA']
# dfgroupby.mean().loc[2,8,256]
# dfgroupby.mean().loc[2,8,:,'GLICA']
# dfgroupby.mean().loc[2,8,:,'GLICA'].plot.barh()
# dfgroupby.mean().loc[2,:,256,:].plot.barh(logx=True,subplots=True)
  
    # Neste aqui, fica tudo muito junto...
# dfgroupby.mean().loc[2,:,:,:].plot.barh(logx=True,subplots=True)

# Por alguma razão, isto aqui dá erro no índice primo:
#dfgroupby.mean().loc[:,8,256,'GLICA']
    # Mas não dá erro no índice de n_samples:
 #dfgroupby.mean().loc[2,8,:,'GLICA']   
 
 # These code seems to help the above behaviour
 # https://stackoverflow.com/questions/30781037/too-many-indexers-with-dataframe-loc
 
 ##ooooh!
 # THis actually works: (now using setup1.mat)

 # dfgroupby.mean().loc[idx[:,3,4096,'GLICA'],:]
    
# another example:
 # needs more refining, such that the x-axis shows interesting values
#dfgroupby.mean().loc[idx[7,:,4096,'GLICA'],:].bss_succ.plot()
#dfgroupby.mean().loc[idx[:,4,4096,'GLICA'],:].total_corr.plot()
#dfgroupby.mean().loc[idx[3,4,:,'GLICA'],:].trial_time.plot()
#dfgroupby.mean().loc[idx[3,4,1024,:],:].total_corr.plot.barh(title='hi')
 
# dfgroupby.count().loc[idx[5,8,:,:],:] # it must be 40 for each one.
 
#dfgroupby.mean().loc[idx[2,3,16384,'GLICA','random_pmf'],'total_corr']
 
#This gave a plot with the algorithms on each line for K varying!
#dfgroupby.mean().loc[idx[2,:,16384,:,'random_pmf'],'total_corr'].unstack(level=3).plot()
 
 
 #this gave a plot with K-varying with labels as dimension,distribution
 #dfgroupby.mean().loc[idx[2,:,16384,:,'random_pmf'],'total_corr'].unstack(level=[0,2,3]).plot()
 
 # This made many graphs with varying K! and many distributions scenario
 #for dist in df.distribution.unique():
    #dfgroupby.mean().loc[idx[2,:,16384,:,dist],'total_corr'].unstack(level=[0,2,3]).plot(title='total_corr')
    
    
#a more refined version:

dfgroupby_errors = dfgroupby.std()/np.sqrt(dfgroupby.count())

total_corr_dir = 'total_corr_plots'
trial_dir = 'trial_time_plots'

T_var_dir = 'T_var'
K_var_dir = 'K_var'
P_var_dir = 'P_var'
dist_var_dir = 'dist_var'

marker = {'america':'x','GLICA':'o','sa4ica':'*','QICA':'^'}

from time import sleep

confidence = 0.95
p = 1-confidence
def plot_K_var_total_corr(slack_time=0):
    
    for dist in df.distribution.unique():
        for prime in df.prime.unique():
            for sample in df.n_samples.unique():
               fig = plt.figure()
               
               for algorithm in df.algorithm_name.unique():
               
                   title= 'Correlação Total com K variando, P = {}, T = {} fixos\n Distribuição {}, Confiança {}%'.format(prime,sample,dist,confidence*100)
                   # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico ou desaparecer do eixo X, portanto os níveis 0 (prime),2(n_samples) e 3(algorithm_name) e 'distribution'. Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
                   x = df.dimension.unique()
                   
                   y = np.array(dfgroupby.mean().loc[idx[prime,:,sample,algorithm,dist],'total_corr'].unstack(level=['prime','n_samples','algorithm_name','distribution']))
                   y = y.flatten()
                   
                   yerr = np.array(dfgroupby_errors.loc[idx[prime,:,sample,algorithm,dist],'total_corr'].unstack(level=['prime','n_samples','algorithm_name','distribution']))
                   
                   c_interval = t_inv_2tail(p,len(y))
                   yerr = c_interval*yerr.flatten()
                   
                   plt.errorbar(x,y,label = algorithm,linewidth=3,marker=marker[algorithm],markersize=15,linestyle='--',yerr=yerr,markerfacecolor='none')
                   plt.title(title)
                   plt.ylabel('Correlação Total [bits]')
                   plt.xlabel('K')
                   plt.xticks(x)
                   # https://stackoverflow.com/a/36646298/1644727, this may be useful
                   plt.legend()

    
                   
               fig = plt.gcf()
               fig.savefig('{}\{}\P{}T{}{}.pdf'.format(total_corr_dir,K_var_dir,prime,sample,dist))
              
               
               plt.show()
               sleep(slack_time)

def plot_K_var_trial_time(slack_time=0):
    for dist in df.distribution.unique():
        for prime in df.prime.unique():
            for sample in df.n_samples.unique():
               fig = plt.figure()
               
               for algorithm in df.algorithm_name.unique():
               
                   title= 'Tempo da trial com K variando, P = {}, T = {} fixos\n Distribuição {}, Confiança {}%'.format(prime,sample,dist,confidence*100)
                   # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico ou desaparecer do eixo X, portanto os níveis 0 (prime),2(n_samples) e 3(algorithm_name) e 'distribution'. Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
                   x = df.dimension.unique()
                   
                   y = np.array(dfgroupby.mean().loc[idx[prime,:,sample,algorithm,dist],'trial_time'].unstack(level=['prime','n_samples','algorithm_name','distribution']))
                   y = y.flatten()
                   
                   yerr = np.array(dfgroupby_errors.loc[idx[prime,:,sample,algorithm,dist],'trial_time'].unstack(level=['prime','n_samples','algorithm_name','distribution']))
                  
                   c_interval = t_inv_2tail(p,len(y))
                   yerr = c_interval*yerr.flatten()
                   
                   plt.errorbar(x,y,label = algorithm,linewidth=3,marker=marker[algorithm],markersize=15,linestyle='--',yerr=yerr,markerfacecolor='none')
                   plt.title(title)
                   plt.ylabel('Tempo da trial [s]')
                   plt.yscale('log')
                   plt.xlabel('K')
                   plt.xticks(x)
                   # https://stackoverflow.com/a/36646298/1644727, this may be useful
                   plt.legend()
    
    
                   
               fig = plt.gcf()
               fig.savefig('{}\{}\P{}T{}{}.pdf'.format(trial_dir,K_var_dir,prime,sample,dist))
              
               
               plt.show()
               sleep(slack_time)
               
               
def plot_P_var_total_corr(slack_time=0):
    for dist in df.distribution.unique():
        for dim in df.dimension.unique():
            for sample in df.n_samples.unique():
               fig = plt.figure()
               
               for algorithm in df.algorithm_name.unique():
               
                   title= 'Correlação Total com P variando, K = {}, T = {} fixos\n Distribuição {}, Confiança {}%'.format(dim,sample,dist,confidence*100)
                   # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico ou desaparecer do eixo X, portanto os níveis 0 (prime),2(n_samples) e 3(algorithm_name) e 'distribution'. Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
                   x = df.prime.unique()
                   
                   
                   # Some strange behaviour here.
                   #It was saying the y and yerr did not have same len. 
                   #So...we did flatten, Also for all other function
                   y = np.array(dfgroupby.mean().loc[idx[:,dim,sample,algorithm,dist],'total_corr'].unstack(level=['dimension','n_samples','algorithm_name','distribution']))
                   y = y.flatten()
                   yerr = np.array(dfgroupby_errors.loc[idx[:,dim,sample,algorithm,dist],'total_corr'].unstack(level=['dimension','n_samples','algorithm_name','distribution']))
                   
                                     
                   c_interval = t_inv_2tail(p,len(y))
                   yerr = c_interval*yerr.flatten()
                   
                   
                   plt.errorbar(x,y,label = algorithm,linewidth=3,marker=marker[algorithm],markersize=15,linestyle='--',yerr=yerr,markerfacecolor='none')
                   plt.title(title)
                   plt.ylabel('Correlação Total [bits]')
                   plt.xlabel('P')
                   plt.xticks(x)
                   # https://stackoverflow.com/a/36646298/1644727, this may be useful
                   plt.legend()
    
    
                   
               fig = plt.gcf()
               fig.savefig('{}\{}\K{}T{}{}.pdf'.format(total_corr_dir,P_var_dir,dim,sample,dist))
              
           
               plt.show()
               sleep(slack_time)
               
def plot_P_var_trial_time(slack_time=0):
    for dist in df.distribution.unique():
        for dim in df.dimension.unique():
            for sample in df.n_samples.unique():
               fig = plt.figure()
               
               for algorithm in df.algorithm_name.unique():
               
                   title= 'Tempo da trial com P variando, K = {}, T = {} fixos\n Distribuição {}, Confiança {}%'.format(dim,sample,dist,confidence*100)
                   # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico ou desaparecer do eixo X, portanto os níveis 0 (prime),2(n_samples) e 3(algorithm_name) e 'distribution'. Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
                   x = df.prime.unique()
                   
                   
                   # Some strange behaviour here.
                   #It was saying the y and yerr did not have same len. 
                   #So...we did flatten, Also for all other function
                   y = np.array(dfgroupby.mean().loc[idx[:,dim,sample,algorithm,dist],'trial_time'].unstack(level=['dimension','n_samples','algorithm_name','distribution']))
                   y = y.flatten()
                   yerr = np.array(dfgroupby_errors.loc[idx[:,dim,sample,algorithm,dist],'trial_time'].unstack(level=['dimension','n_samples','algorithm_name','distribution']))
                                    
                   c_interval = t_inv_2tail(p,len(y))
                   yerr = c_interval*yerr.flatten()
                   
                   
                   plt.errorbar(x,y,label = algorithm,linewidth=3,marker=marker[algorithm],markersize=15,linestyle='--',yerr=yerr,markerfacecolor='none')
                   plt.title(title)
                   plt.ylabel('Tempo da trial [s]')
                   plt.yscale('log')
                   plt.xlabel('P')
                   plt.xticks(x)
                   # https://stackoverflow.com/a/36646298/1644727, this may be useful
                   plt.legend()
    
    
                   
               fig = plt.gcf()
               fig.savefig('{}\{}\K{}T{}{}.pdf'.format(trial_dir,P_var_dir,dim,sample,dist))
             
           
               plt.show()
               sleep(slack_time)

def plot_T_var_total_corr(slack_time=0):
    for dist in df.distribution.unique():
        for dim in df.dimension.unique():
            for prime in df.prime.unique():
               fig = plt.figure()
               
               for algorithm in df.algorithm_name.unique():
               
                   title= 'Correlação Total com T variando, P = {}, K = {} fixos\n Distribuição {}, Confiança {}%'.format(prime,dim,dist,confidence*100)
                   # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico ou desaparecer do eixo X, portanto os níveis 0 (prime),2(n_samples) e 3(algorithm_name) e 'distribution'. Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
                   x = df.n_samples.unique()
                   
                   
                   # Some strange behaviour here.
                   #It was saying the y and yerr did not have same len. 
                   #So...we did flatten, Also for all other function
                   y = np.array(dfgroupby.mean().loc[idx[prime,dim,:,algorithm,dist],'total_corr'].unstack(level=['prime','dimension','algorithm_name','distribution']))
                   y = y.flatten()
                   yerr = np.array(dfgroupby_errors.loc[idx[prime,dim,:,algorithm,dist],'total_corr'].unstack(level=['prime','dimension','algorithm_name','distribution']))
                                   
                   c_interval = t_inv_2tail(p,len(y))
                   yerr = c_interval*yerr.flatten()
                   
                   
                   plt.errorbar(x,y,label = algorithm,linewidth=3,marker=marker[algorithm],markersize=15,linestyle='--',yerr=yerr,markerfacecolor='none')
                   plt.title(title)
                   plt.ylabel('Correlação Total [bits]')
                   plt.xlabel('T')
                   plt.xticks(x)
                   # https://stackoverflow.com/a/36646298/1644727, this may be useful
                   plt.legend()
    
    
                   
               fig = plt.gcf()
               fig.savefig('{}\{}\P{}K{}{}.pdf'.format(total_corr_dir,T_var_dir,prime,dim,dist))
               
           
               plt.show()
               sleep(slack_time)

def plot_T_var_trial_time(slack_time=0):    
    for dist in df.distribution.unique():
        for dim in df.dimension.unique():
            for prime in df.prime.unique():
               fig = plt.figure()
               
               for algorithm in df.algorithm_name.unique():
               
                   title= 'Tempo da trial com T variando,P = {}, K = {} fixos\n Distribuição {}, Confiança {}%'.format(prime,dim,dist,confidence*100)
                   # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico ou desaparecer do eixo X, portanto os níveis 0 (prime),2(n_samples) e 3(algorithm_name) e 'distribution'. Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
                   x = df.n_samples.unique()
                   
                   
                   # Some strange behaviour here.
                   #It was saying the y and yerr did not have same len. 
                   #So...we did flatten, Also for all other function
                   y = np.array(dfgroupby.mean().loc[idx[prime,dim,:,algorithm,dist],'trial_time'].unstack(level=['prime','dimension','algorithm_name','distribution']))
                   y = y.flatten()
                   yerr = np.array(dfgroupby_errors.loc[idx[prime,dim,:,algorithm,dist],'trial_time'].unstack(level=['prime','dimension','algorithm_name','distribution']))
                   
                                     
                   c_interval = t_inv_2tail(p,len(y))
                   yerr = c_interval*yerr.flatten()
                   
                   
                   plt.errorbar(x,y,label = algorithm,linewidth=3,marker=marker[algorithm],markersize=15,linestyle='--',yerr=yerr,markerfacecolor='none')
                   plt.title(title)
                   plt.ylabel('Tempo da trial [s]')
                   plt.yscale('log')
                   plt.xlabel('T')
                   plt.xticks(x)
                   # https://stackoverflow.com/a/36646298/1644727, this may be useful
                   plt.legend()
    
    
                   
               fig = plt.gcf()
               fig.savefig('{}\{}\P{}K{}{}.pdf'.format(trial_dir,T_var_dir,prime,dim,dist))
               
           
               plt.show()
               sleep(slack_time)
        
 #TODO
def plot_dist_var_total_corr(slack_time=0):
    # We add this, because when saving it was cutting the x_labels
    bbox_inches = 'tight'
    distributions = df.distribution.unique()
    distributions = np.sort(distributions)
    for sample in df.n_samples.unique():
        for dim in df.dimension.unique():
            for prime in df.prime.unique():
                fig = plt.figure()
                x_dif = 0
                for algorithm in df.algorithm_name.unique():
               
                   x_dif+=1
                   title= 'Correlação Total com distribuição variando\n P = {}, K = {}, T={} fixos'.format(prime,dim,sample)
                   # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico ou desaparecer do eixo X, portanto os níveis 0 (prime),2(n_samples) e 3(algorithm_name) e 'distribution'. Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
                   xlabels = distributions
                   x = np.linspace(1,20,num=10)
                   
     
                   # Some strange behaviour here.
                   #It was saying the y and yerr did not have same len. 
                   #So...we did flatten, Also for all other function
                   y = np.array(dfgroupby.mean().loc[idx[prime,dim,sample,algorithm,:],'total_corr'].unstack(level=['prime','dimension','n_samples','algorithm_name']))
                   y = y.flatten()
                   yerr = np.array(dfgroupby_errors.loc[idx[prime,dim,sample,algorithm,:],'total_corr'].unstack(level=['prime','dimension','n_samples','algorithm_name']))
                                     
                   c_interval = t_inv_2tail(p,len(y))
                   yerr = c_interval*yerr.flatten()
                   
                   
                   plt.errorbar(x,y,label = algorithm,linewidth=3,marker=marker[algorithm],markersize=15,linestyle='--',yerr=yerr,markerfacecolor='none')
                   #plt.bar(x+x_dif,y,yerr=yerr,align='edge',tick_label=xlabels)
                   plt.title(title)
                   plt.ylabel('Correlação Total [bits]')
                   
                   plt.xlabel('distribuição')
                   plt.xticks(x,labels = xlabels,rotation=60)
                   
                   # https://stackoverflow.com/a/36646298/1644727, this may be useful
                   plt.legend()
    
    
                   
                fig = plt.gcf()
                fig.savefig('{}\{}\P{}K{}T{}.pdf'.format(total_corr_dir,dist_var_dir,prime,dim,sample),bbox_inches=bbox_inches)
                
           
                plt.show()
                sleep(slack_time)

from collections import Counter
def show_where_algorithm_is_min_total_corr(algorithm):
    min_list = []
    for prime in df.prime.unique():
        for dim in df.dimension.unique():
            for sample in df.n_samples.unique():
                min_list.append(dfgroupby.mean().loc[idx[prime,dim,sample,algorithm,:]]['total_corr'].idxmin()[4])
                
    print(Counter(min_list))
    
def show_where_algorithm_is_max_total_corr(algorithm):
    max_list = []
    for prime in df.prime.unique():
        for dim in df.dimension.unique():
            for sample in df.n_samples.unique():
                max_list.append(dfgroupby.mean().loc[idx[prime,dim,sample,algorithm,:]]['total_corr'].idxmax()[4])
                
    print(Counter(max_list))

def product_mean_on_algorithms():
    total_corrmean = dfgroupby.mean()['total_corr']
    trial_timemean = dfgroupby.mean()['trial_time']
    product_mean = total_corrmean * trial_timemean
    
    print(product_mean.mean(level=3))
    
    
# https://medium.com/@SciencelyYours/errors-bars-standard-errors-and-confidence-intervals-on-line-and-bar-graphics-matlab-254d6aa32b76
# https://stackoverflow.com/questions/19339305/python-function-to-get-the-t-statistic
    
def t_inv_2tail(p,n):
    # p is live: p<0,05, so you would put if you want confidence of 
    # 0.95 = 95%
    # confidence = 1 - p
    return(t.ppf(1-p/2,n-1))
    
def t_inv_singletail(p,n):
    # p is live: p<0,05, so you would put if you want confidence of 
    # 0.95 = 95%
    # confidence = 1 - p
    return(t.ppf(1-p,n-1))


# slack_time seconds interval between plots
slack_time = 0
if do_plot == True:
    # K variando P e T fixos!
    # plots for total corr        
    plot_K_var_total_corr(slack_time)
            
    #plots for trial time
    plot_K_var_trial_time(slack_time)
         
    
    # P variando K e T fixos!     
    # plots for total corr
    plot_P_var_total_corr(slack_time)
   
    # plots for trial time
    plot_P_var_trial_time(slack_time)
    
    # T variando P e T fixos!
    # plots for total corr
    plot_T_var_total_corr(slack_time)
    


    #plots for trial time
    plot_T_var_trial_time(slack_time)
    
    plot_dist_var_total_corr(slack_time)
    

               
## 16/10/2019 starting the Wilcoxon tests, now.
               
def select_trials_from(df,prime,dimension,n_samples,algorithm,distribution,the_col):
    return df[(df.prime==prime)&\
              (df.dimension==dimension)&\
              (df.n_samples==n_samples)&\
              (df.algorithm_name==algorithm)&\
              (df.distribution==distribution)
              ][the_col]
 
    
def select_all_P_mean_cases_from(dfgroupbymean,prime,algorithm,column):
    return dfgroupbymean.loc[idx[prime,:,:,algorithm,:],column]
### 20/10/2019
    # important command to check stuff:
#dfgroupbymean_total_corr = dfgroupby.mean().loc[idx[:,:,:,:,:],'total_corr'].unstack(level='algorithm_name')

## Show the algorithms which gives the minimum values for total corr
#dfgroupbymean_total_corr.idxmin(axis='columns') 

# shows were QICA were the minimum 
    # Which is just 2.
#dfgroupbymean_total_corr[ dfgroupbymean_total_corr.idxmin(axis='columns') == 'QICA']

## Testing america vs QICA on total_corr test less

# 26/10/2019 Comentado, por enquanto
    
zero_method = 'wilcox'  
alternative = 'less'
algo1 = 'america'
algo2 = 'QICA'
target_value = 'total_corr'
print('wilcoxon test for {} vs {} on {}, test {}'.format(algo1,algo2,target_value,alternative))  
for prime in df.prime.unique():
    for dim in df.dimension.unique():
        for sample in df.n_samples.unique():
            for dist in df.distribution.unique():
                x = np.array(select_trials_from(df,prime,dim,sample,algo1,dist,target_value))
                y = np.array(select_trials_from(df,prime,dim,sample,algo2,dist,target_value))
                
                try:
                   
                    result = wilcoxon(x,y,zero_method=zero_method,alternative=alternative)
                    if(result[1] < 0.05):
                         print('wilcoxon test on P={},K={},T={},dist={}'.format(prime,dim,sample,dist))
                         print(result)
                except:
                    pass
                
                
                
print('\n wilcoxon for P_mean {} vs {} on {}, test {}'.format(algo1,algo2,target_value,alternative))  
for prime in df.prime.unique():
    x = select_all_P_mean_cases_from(dfgroupby.mean().dropna(),prime,algo1,target_value)
    y = select_all_P_mean_cases_from(dfgroupby.mean().dropna(),prime,algo2,target_value)
    try:
        result = wilcoxon(x,y,zero_method=zero_method,alternative=alternative)
        
        if(result[1] < 0.05):
            print('wilcoxon test on P={}'.format(prime))
            print(result)
        
    except:
        pass
