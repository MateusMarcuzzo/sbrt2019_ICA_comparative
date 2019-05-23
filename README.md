# sbrt2019_ICA_comparative
# Comparação de algoritmos de ICA para o caso de BSS e ICA pura

Este projeto envolve principalmente dois cenários de simulação:

Utilizamos no primeiro cenário que chamaremos de setup 1, três algoritmos diferentes de ICA linear que resolvem o problem de separação cega de fontes (blind source separation - BSS).
Mediremos a taxa de sucesso de separação perfeita (ou seja, separação parciais não conta aqui, apenas totais).
Bem como guardaremos o tempo de cada trial e sua correlação total (informação mútua generalizada - Watanabe) para que possamos
cruzar dados no futuro, caso necessário.

O setup 2 é um experimento de ICA pura, e testamos quatro algoritmos diferentes, onde três deles são os três lineares do anterior e um deles
, novo, não-linear, chamado QICA.

Dada a pmf conjunta Zipf, com os mesmos parâmetros encontrados no artigo do criador do algoritmo, Amichai Painsky, realizamos o mesmo experimento
de medir a função objetivo de correlação total para ver quem executa ICA melhor, isto é, encontra componentes independentes o melhor quanto possível.

Neste experimento medimos o tempo de cada algoritmo bem como sua correlação total.

________________

Os números primos,
A quantidade de fontes
A quantidade de observações
E os algoritmos

São dispostos nos scripts de simulação:

runs_sim_bss_and_total_corr_17_04_2019
Note que o nome, ainda que pedante, informa o que o script faz e qual o experimento.
A data é uma mera nota mental de quando esse arquivo começou inicialmente a ser trabalhado.

O segundo arquivo de simulação, o do setup2:

runs_sim_total_corr_pure_ICA_09_05_2019
Segue as mesmas orientações do primeiro.

___________

Cada um destes arquivos geram arquivo de save('nomearquivo') para que possamos executar load('nomearquivo')
em um momento futuro, quando terminadas as simulações.

Necessitamos destes arquivos para que possamos plotar gráficos das simulações e ter visualização do que ocorreu no experimento
sob diversas perspectivas e projeções dos dados.

Antes de executarmos algum plotter, pretendemos dar o load em alguns desses arquivos.


________________________

Os arquivos de simulação geram os nomes destes saves com a data e a hora que a simulação começou bem como a que terminou.
No final do nome aparece de qual script se trata, do setup1 ou do setup2.

Por exemplo:
sim_data_start_12_05_2019_23_50_end_13_05_2019_12_58_sim_bss_and_total_corr_17_04_2019
e
sim_data_start_13_05_2019_12_58_end_13_05_2019_23_21_sim_total_corr_pure_ICA_09_05_2019

_______________________

Os scripts também retornam, ao final de sua execução, o tempo total decorrido, para teste de ensaio em dimensões mais longas ou
uma pré-estimativa do tempo necessário se quisermos mudar os parâmetros de simulações e/ou colocar mais trials, para uma medida mais
fidedigna de estatística dos algoritmos.













