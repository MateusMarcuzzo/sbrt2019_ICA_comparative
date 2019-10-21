# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:30:07 2019

@author: MateusMarcuzzo
"""

import numpy as np
from scipy.io import loadmat
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# The colors were too close
#plt.style.use('ggplot')

plt.style.use('seaborn-bright')
# If True, generates many plots
do_plot = True

# It's really useful for getting the right indices on .loc usage
idx = pd.IndexSlice 

# 21/09/2019
# we are importing an .mat file from Octave.
# we must do rows with it, extracting the important variables.

important_variables = ['some_primes','n_sources','n_samples','algorithms_names','n_trials',]
                      
# to be used later to build a dataframe                     
rows_list = []


## Possible files names:
# diff_setup1_part1.mat
# diff_setup1_part2.mat
# diff_setup1_part3.mat
# diff_setup1_part4.mat << Daniel. Needs a different treatment
# diff2_setup1.mat
# setup1.mat
      
setups_names = ['diff_setup1_part1.mat',
                'diff_setup1_part2.mat',
                'diff_setup1_part3.mat',
                'diff_setup1_part4.mat',
                'diff2_setup1.mat',
                'setup1.mat']

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
    data_variables = ['bss_succ_rate','bss_trial_time','total_corr_results']
    
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
        temp_dict.update({'n_sources':row[1]})
    
        
        tnsai = the_n_samples_index = np.min(np.where(a_setup['n_samples'] == row[2]))
        temp_dict.update({'n_samples':row[2]})
    
        
        # It's not an np.array, its a list!
        talgi = the_algorithm_index = np.min(a_setup['algorithms_names'].index(row[3]))
        temp_dict.update({'algorithm_name': row[3]})
    
        if some_setup != 'diff_setup1_part4.mat':
            # I've put some .copy() at the end, and It seems that it worked
            # It was giving some problems without it
            for trial in range(n_trials):
                temp_dict['total_corr'] = a_setup['total_corr_results'][(tpi,tnsoi,tnsai,talgi,trial)].copy()
                temp_dict['bss_succ'] = a_setup['bss_succ_rate'][(tpi,tnsoi,tnsai,talgi,trial)].copy()
                temp_dict['trial_time'] = a_setup['bss_trial_time'][(tpi,tnsoi,tnsai,talgi,trial)].copy()
                
                rows_list.append(temp_dict.copy())
        else:
            for trial in range(n_trials):
                algorithm = a_setup['algorithms_names'][talgi]
                #The .lower() here is just because its low-case in Daniels code for the name
                temp_dict['total_corr'] = a_setup['total_corr_results_{}'.format(algorithm.lower())][(tpi,tnsoi,tnsai,trial)].copy()
                temp_dict['bss_succ'] = a_setup['bss_succ_rate_{}'.format(algorithm.lower())][(tpi,tnsoi,tnsai,trial)].copy()
                temp_dict['trial_time'] = a_setup['bss_trial_time_{}'.format(algorithm.lower())][(tpi,tnsoi,tnsai,trial)].copy()
        
                rows_list.append(temp_dict.copy())
            

        
#just checking    
#for row in rows_list:
#    print(row)


# Here the dataframe is created
df = pd.DataFrame(rows_list)

# we change this column to  category
df['algorithm_name'] = df['algorithm_name'].astype('category')


# and these to integer values (since they are), they come as float
int_variables = ['n_samples','n_sources','prime']
for a_variable in int_variables:
    df[a_variable] = df[a_variable].astype('int')

#print(df)
#
### Above we have some examples.
#
## it shows some stuff we already know on matlab/Octave:
#print(df.groupby(['prime','n_sources','n_samples','algorithm_name']).mean())
#print(df.groupby(['prime','n_sources','n_samples','algorithm_name']).count())
#
#print(df.dtypes)
#    
#
### Some interesting commands
dfgroupby = df.groupby(['prime','n_sources','n_samples','algorithm_name']) 

#drops the cases with NaN values.


dfgroupby_bss = dfgroupby['bss_succ']
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
print('n_sources evaluated')
print(df.n_sources.unique())
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
 
 
 
 # inspirado no extrator de dados de ICA:
 # Tive de fazer algumas adaptações, claramente
 
 
 
 
 
 
 
 # Faltam os plots de BSS
 
dfgroupby_errors = dfgroupby.std()/np.sqrt(dfgroupby.count())
    
bss_dir = 'bss_plots'
total_corr_dir = 'total_corr_plots'
trial_dir = 'trial_time_plots'

T_var_dir = 'T_var'
K_var_dir = 'K_var'
P_var_dir = 'P_var'

if do_plot == True:
    # K variando P e T fixos!
    # plots for total corr
    
    for prime in df.prime.unique():
        for sample in df.n_samples.unique():
           fig = plt.figure()
           # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico, portanto os níveis 0 (prime),2(n_samples) e 3(algorithm_name). Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
           ax = dfgroupby.mean().loc[idx[prime,:,sample,:],'total_corr'].unstack(level=['prime','n_samples','algorithm_name']).plot(title='Correlação Total com K variando \n P = {}, T = {} fixos'.format(prime,sample),linewidth=3,marker='o',markersize=10,linestyle='--',yerr=dfgroupby_errors.loc[idx[prime,:,sample,:],'total_corr'].unstack(level=['prime','n_samples','algorithm_name']))
           ax.set(ylabel = 'Correlação Total [bits]')
           
           fig = ax.get_figure()
           fig.savefig('{}\{}\P{}T{}'.format(total_corr_dir,K_var_dir,prime,sample))
           fig.savefig('{}\{}\P{}T{}.eps'.format(total_corr_dir,K_var_dir,prime,sample))
           plt.show()
           
    # bss_succ  rate
    for prime in df.prime.unique():
        for sample in df.n_samples.unique():
           plt.figure()
           # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico, portanto os níveis 0 (prime),2(n_samples) e 3(algorithm_name). Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
           ax = dfgroupby.mean().loc[idx[prime,:,sample,:],'bss_succ'].unstack(level=['prime','n_samples','algorithm_name']).plot(title='Taxa de Sucesso Separação Total com K variando \n P = {}, T = {} fixos'.format(prime,sample),linewidth=3,marker='o',markersize=10,linestyle='--',yerr=dfgroupby_errors.loc[idx[prime,:,sample,:],'bss_succ'].unstack(level=['prime','n_samples','algorithm_name']))
           ax.set(ylabel = 'Taxa de Sucesso de Separação [0 - 1]')
           
           fig = ax.get_figure()
           fig.savefig('{}\{}\P{}T{}'.format(bss_dir,K_var_dir,prime,sample))
           fig.savefig('{}\{}\P{}T{}.eps'.format(bss_dir,K_var_dir,prime,sample))
           
           plt.show()
        
    #plots for trial time
    
    for prime in df.prime.unique():
        for sample in df.n_samples.unique():
           plt.figure()
           ax = dfgroupby.mean().loc[idx[prime,:,sample,:],'trial_time'].unstack(level=['prime','n_samples','algorithm_name']).plot(title='Tempo da trial com K variando \n P = {}, T = {} fixos'.format(prime,sample),linewidth=3, marker='o',markersize=10,linestyle='--',yerr=dfgroupby_errors.loc[idx[prime,:,sample,:],'trial_time'].unstack(level=['prime','n_samples','algorithm_name']))
           ax.set(ylabel = 'Tempo da trial [s]')
           ax.set(yscale='log')
           
           fig = ax.get_figure()
           fig.savefig('{}\{}\P{}T{}'.format(trial_dir,K_var_dir,prime,sample))
           fig.savefig('{}\{}\P{}T{}.eps'.format(trial_dir,K_var_dir,prime,sample))
           
           plt.show()
           
    # P variando K e T fixos!
    # plots for total corr
    
    for dim in df.n_sources.unique():
        for sample in df.n_samples.unique():
           plt.figure()
           # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico, portanto os níveis 1 (n_sources),2(n_samples) e 3(algorithm_name). Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
           ax = dfgroupby.mean().loc[idx[:,dim,sample,:],'total_corr'].unstack(level=['n_sources','n_samples','algorithm_name']).plot(title='Correlação Total com P variando \n K = {}, T = {} fixos'.format(dim,sample),linewidth=3,marker='o',markersize=10,linestyle='--',yerr=dfgroupby_errors.loc[idx[:,dim,sample,:],'total_corr'].unstack(level=['n_sources','n_samples','algorithm_name']))
           ax.set(ylabel = 'Correlação Total [bits]')
           ax.set_xticks(df.prime.unique())
           
           fig = ax.get_figure()
           fig.savefig('{}\{}\K{}T{}'.format(total_corr_dir,P_var_dir,dim,sample))
           fig.savefig('{}\{}\K{}T{}.eps'.format(total_corr_dir,P_var_dir,dim,sample))
           
           plt.show()
          
    # bss succ rate
    for dim in df.n_sources.unique():
        for sample in df.n_samples.unique():
           plt.figure()
           # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico, portanto os níveis 1 (n_sources),2(n_samples) e 3(algorithm_name). Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
           ax = dfgroupby.mean().loc[idx[:,dim,sample,:],'bss_succ'].unstack(level=['n_sources','n_samples','algorithm_name']).plot(title='Taxa de Sucesso de Separação Total com P variando \n K = {}, T = {} fixos'.format(dim,sample),linewidth=3,marker='o',markersize=10,linestyle='--',yerr=dfgroupby_errors.loc[idx[:,dim,sample,:],'bss_succ'].unstack(level=['n_sources','n_samples','algorithm_name']))
           ax.set(ylabel = 'Taxa de Sucesso de Separação [0 - 1]')
           ax.set_xticks(df.prime.unique())
           
           fig = ax.get_figure()
           fig.savefig('{}\{}\K{}T{}'.format(bss_dir,P_var_dir,dim,sample))
           fig.savefig('{}\{}\K{}T{}.eps'.format(bss_dir,P_var_dir,dim,sample))
           
           plt.show()
        
    #plots for trial time
    
    for dim in df.n_sources.unique():
        for sample in df.n_samples.unique():
           plt.figure()
           ax = dfgroupby.mean().loc[idx[:,dim,sample,:],'trial_time'].unstack(level=['n_sources','n_samples','algorithm_name']).plot(title='Tempo da trial com P variando \n K = {}, T = {} fixos'.format(dim,sample),linewidth=3, marker='o',markersize=10,linestyle='--',yerr=dfgroupby_errors.loc[idx[:,dim,sample,:],'trial_time'].unstack(level=['n_sources','n_samples','algorithm_name']))
           ax.set(ylabel = 'Tempo da trial [s]')
           ax.set(yscale='log')
           ax.set_xticks(df.prime.unique())
           
           fig = ax.get_figure()
           fig.savefig('{}\{}\K{}T{}'.format(trial_dir,P_var_dir,dim,sample))
           fig.savefig('{}\{}\K{}T{}.eps'.format(trial_dir,P_var_dir,dim,sample))
           
           plt.show()
           
    # T variando P e T fixos!
    # plots for total corr
    
    for dim in df.n_sources.unique():
        for prime in df.prime.unique():
           plt.figure()
           # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico, portanto os níveis 1 (n_sources),0(prime) e 3(algorithm_name). Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
           ax = dfgroupby.mean().loc[idx[prime,dim,:,:],'total_corr'].unstack(level=['prime','n_sources','algorithm_name']).plot(title='Correlação Total com T variando \n P = {}, K = {} fixos'.format(prime,dim),linewidth=3,marker='o',markersize=10,linestyle='--',yerr=dfgroupby_errors.loc[idx[prime,dim,:,:],'total_corr'].unstack(level=['prime','n_sources','algorithm_name']))
           ax.set(ylabel = 'Correlação Total [bits]')
           ax.set_xticks(df.n_samples.unique())
           
           fig = ax.get_figure()
           fig.savefig('{}\{}\K{}P{}'.format(total_corr_dir,T_var_dir,dim,prime))
           fig.savefig('{}\{}\K{}P{}.eps'.format(total_corr_dir,T_var_dir,dim,prime))
           
           plt.show()
           
    # bss succ rate       
    for dim in df.n_sources.unique():
        for prime in df.prime.unique():
           plt.figure()
           # A gente só dá unstack no que é fixo ou deve aparecer na legenda do gráfico, portanto os níveis 1 (n_sources),0(prime) e 3(algorithm_name). Deixei na forma de string para claridade. Desculpa por ter uma linha tão grande, mas é a vida
           ax = dfgroupby.mean().loc[idx[prime,dim,:,:],'bss_succ'].unstack(level=['prime','n_sources','algorithm_name']).plot(title='Taxa de Sucesso de Separação Total com T variando \n P = {}, K = {} fixos'.format(prime,dim),linewidth=3,marker='o',markersize=10,linestyle='--',yerr=dfgroupby_errors.loc[idx[prime,dim,:,:],'bss_succ'].unstack(level=['prime','n_sources','algorithm_name']))
           ax.set(ylabel = 'Taxa de Sucesso de Separação [0 - 1]')
           ax.set_xticks(df.n_samples.unique())
           
           fig = ax.get_figure()
           fig.savefig('{}\{}\K{}P{}'.format(bss_dir,T_var_dir,dim,prime))
           fig.savefig('{}\{}\K{}P{}.eps'.format(bss_dir,T_var_dir,dim,prime))
           plt.show()
        
    #plots for trial time
    
    for dim in df.n_sources.unique():
        for prime in df.prime.unique():
           plt.figure()
           ax = dfgroupby.mean().loc[idx[prime,dim,:,:],'trial_time'].unstack(level=['prime','n_sources','algorithm_name']).plot(title='Tempo da trial com T variando \n P = {}, K = {} fixos'.format(prime,dim),linewidth=3, marker='o',markersize=10,linestyle='--',yerr=dfgroupby_errors.loc[idx[prime,dim,:,:],'trial_time'].unstack(level=['prime','n_sources','algorithm_name']))
           ax.set(ylabel = 'Tempo da trial [s]')
           ax.set(yscale='log')
           ax.set_xticks(df.n_samples.unique())
           
           
           fig = ax.get_figure()
           fig.savefig('{}\{}\K{}P{}'.format(trial_dir,T_var_dir,dim,prime))
           fig.savefig('{}\{}\K{}P{}.eps'.format(trial_dir,T_var_dir,dim,prime))
           
           plt.show()
           
           
           
def select_trials_from(df,prime,n_sources,n_samples,algorithm,the_col):
    return df[(df.prime==prime)&\
              (df.n_sources==n_sources)&\
              (df.n_samples==n_samples)&\
              (df.algorithm_name==algorithm)\
              ][the_col]
    
    
def get_non_perfect_nor_0_bss():
    return dfgroupby.mean().loc[(dfgroupby.mean()['bss_succ'] > 0) &\
                        (dfgroupby.mean()['bss_succ'] < 1)]
    
def get_perfect_bss():
    return dfgroupby.mean().loc[(dfgroupby.mean()['bss_succ'] == 1)]

def get_0_bss():
    return dfgroupby.mean().loc[(dfgroupby.mean()['bss_succ'] == 0)]


### 15/10/2019
#  trying wilcoxon test part for bss

zero_method = 'wilcox'
alternative = 'greater'
algo1 = 'america'
algo2 = 'GLICA'
print('wilcoxon for {} vs {} on bss_succ, test {}'.format(algo1,algo2,alternative))    
for prime in df.prime.unique():
    for source in df.n_sources.unique():
        for sample in df.n_samples.unique():
            x = select_trials_from(df,prime,source,sample,algo1,'bss_succ')
            y = select_trials_from(df,prime,source,sample,algo2,'bss_succ')
            try:
                result = wilcoxon(x,y,zero_method=zero_method,alternative=alternative)
                
                if(result[1] < 0.05):
                    print('wilcoxon test on P={},K={},T={}'.format(prime,source,sample))
                    print(result)
                
            except:
                pass
            
#  trying wilcoxon test part for total_corr
    
zero_method = 'wilcox'
alternative = 'less'
algo1 = 'america'
algo2 = 'GLICA'
print('\n wilcoxon for {} vs {} on total_corr, test {}'.format(algo1,algo2,alternative))  
for prime in df.prime.unique():
    for source in df.n_sources.unique():
        for sample in df.n_samples.unique():
            x = select_trials_from(df,prime,source,sample,algo1,'total_corr')
            y = select_trials_from(df,prime,source,sample,algo2,'total_corr')
            try:
                result = wilcoxon(x,y,zero_method=zero_method,alternative=alternative)
                
                if(result[1] < 0.05):
                    print('wilcoxon test on P={},K={},T={}'.format(prime,source,sample))
                    print(result)
                
            except:
                pass
   
# 16/10/2019
# trying wilcoxon test regarding the means 
                
def select_all_P_mean_cases_from(dfgroupbymean,prime,algorithm,column):
    return dfgroupbymean.loc[idx[prime,:,:,algorithm],column]


## Test for bss_succ, america vs GLICA,
zero_method = 'wilcox'
alternative = 'greater'      

algo1 = 'america'
algo2 = 'GLICA' 
print('wilcoxon on GF(P) means for {} vs {} on bss_succ, test {}'.format(algo1,algo2,alternative))  

for prime in df.prime.unique():

    x = np.array(select_all_P_mean_cases_from(dfgroupby.mean().dropna(),prime,algo1,'bss_succ'))
    y = np.array(select_all_P_mean_cases_from(dfgroupby.mean().dropna(),prime,algo2,'bss_succ'))
    try:
        
        print('wilcoxon test on P={} means'.format(prime))
        result = wilcoxon(x,y,zero_method=zero_method,alternative=alternative)
        
        
        print(result)
        
    except:
        pass 


## test for total_corr america vs GLICA, less
zero_method = 'wilcox'
alternative = 'less'       

algo1 = 'america'
algo2 = 'GLICA' 
print('wilcoxon on GF(P) means for {} vs {} on total_corr, test {}'.format(algo1,algo2,alternative))  

for prime in df.prime.unique():

    x = np.array(select_all_P_mean_cases_from(dfgroupby.mean().dropna(),prime,algo1,'total_corr'))
    y = np.array(select_all_P_mean_cases_from(dfgroupby.mean().dropna(),prime,algo2,'total_corr'))
    try:
        
        print('wilcoxon test on P={} means'.format(prime))
        result = wilcoxon(x,y,zero_method=zero_method,alternative=alternative)
        
        
        print(result)
        
    except:
        pass 
  
    
