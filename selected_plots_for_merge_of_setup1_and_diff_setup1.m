load('sim_data_start_12_05_2019_23_50_end_13_05_2019_12_58_sim_bss_and_total_corr_17_04_2019')

setup1.bss = mean_bss_succ_rate;
setup1.trial_time = mean_bss_trial_time;
setup1.total_corr_results = mean_total_corr_results;

setup1.some_primes = some_primes;
setup1.n_sources = n_sources;
setup1.n_samples = n_samples;
setup1.algorithms = algorithms_names;
setup1.n_trials = n_trials;

 
load('sim_data_start_18_05_2019_19_21_end_20_05_2019_01_16_sim_bss_and_total_corr_17_04_2019_diff_setup1')

size_to_merge = [ 4 2 5 3];

diff_setup1.bss = mean_bss_succ_rate;
diff_setup1.bss = resize(diff_setup1.bss, size_to_merge );

diff_setup1.trial_time = mean_bss_trial_time;
diff_setup1.trial_time = resize(diff_setup1.trial_time, size_to_merge );


diff_setup1.total_corr_results = mean_total_corr_results;
diff_setup1.total_corr_results = resize(diff_setup1.total_corr_results, size_to_merge);

diff_setup1.some_primes = some_primes;
diff_setup1.n_sources = n_sources;
diff_setup1.n_samples = n_samples;
diff_setup1.algorithms = algorithms_names;
diff_setup1.n_trials = n_trials;


%% ok ...we have data from both.
%% now we'll try to do the following:


bss_overall = [setup1.bss, diff_setup1.bss];
trial_time_overall = [setup1.trial_time,diff_setup1.trial_time];
total_corr_results_overall = [setup1.total_corr_results, diff_setup1.total_corr_results];

n_sources_overall = [setup1.n_sources diff_setup1.n_sources];
some_primes_overall = setup1.some_primes;
n_samples_overall = n_samples;




%%% we need to set the 0 values to NaN, because they didn't were measured.
%% which is the case for P= 7 and K=6 or 7
bss_overall(bss_overall == 0 ) = NaN;
trial_time_overall(trial_time_overall == 0) = NaN;
total_corr_results_overall(total_corr_results_overall ==0) = NaN;

for i=1:length(algorithms_names);
	algorithms_str{i} = sprintf(";%s;",algorithms_names{i});
end

%% Here we'll do something close to the do_plots_of_sims_17_04_2019
%% I'm having problems with the plots and utf-8...
%% gonna try my best

%%%%%%%%%%%%%% PLOT OF MEAN TIME VARYING K %%%%%%%%%%%%%% 
%%%%%%%%%%% FIGURA 2 %%%%%%%%%%%%
figure('visible','on','name','varying K, mean time');


sim_data.bss = bss_overall;
sim_data.trial_time = trial_time_overall;
sim_data.total_corr = total_corr_results_overall;
sim_data.n_samples = n_samples_overall;
sim_data.n_sources = n_sources_overall;
sim_data.some_primes = some_primes_overall;


x = n_sources_overall
p_i =  2 %% prime = 3
t_i = 5 %% n_samples = 4096
k_i = 1:length(n_sources_overall)

for i=1:length(algorithms_names);
	algorithms_str{i} = sprintf(";%s",algorithms_names{i});
	%% we will add something after the names
	%% this is meant to be used as a legend
end

xlabel_str = 'número de fontes';

markers_list = {'o','+','x'}; 
color_list= {'blue','red','green'};

for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s P = %d;",algorithms_str{algo_i},some_primes_overall(p_i))
	the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','-')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)
	legend ("location", "northeastoutside");
	hold on

end


ylabel("média temporal do algoritmo [s]");
xlabel(xlabel_str);
% the_str = sprintf("mean time: P = %d and T= %d",some_primes_overall(p_i),n_samples_overall(t_i));
the_str = sprintf("média temporal, P e T = %d fixos",n_samples_overall(t_i))
title(the_str);
    
set(gca,"xtick",x);
set(gca,"xticklabel",x);


p_i = 3 %% now 5

for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s P = %d;",algorithms_str{algo_i},some_primes_overall(p_i))
	the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','--')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)
	legend ("location", "northeastoutside");
	hold on

end

p_i = 4 %% now it's 7

for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s P = %d;",algorithms_str{algo_i},some_primes_overall(p_i))
	the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','-.')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)
	legend ("location", "northeastoutside");
	hold on
end


filename = sprintf('varying_K_mean_time_merge_setup1.svg')
print(filename,'-color')


%%%%%%%%%%%%% PLOT OF MEAN TIME VARYING P %%%%%%%%%%%%
%%%%%%% FIGURA 3%%%%%%%%

figure('visible','on','name','varying P, mean time');



sim_data.bss = bss_overall;
sim_data.trial_time = trial_time_overall;
sim_data.total_corr = total_corr_results_overall;
sim_data.n_samples = n_samples_overall;
sim_data.n_sources = n_sources_overall;
sim_data.some_primes = some_primes_overall;


x = some_primes_overall
p_i =  1:length(some_primes_overall) %% 
t_i = 5 %% n_samples = 4096
k_i = 4 % 5 sources

for i=1:length(algorithms_names);
	algorithms_str{i} = sprintf(";%s",algorithms_names{i});
	%% we will add something after the names
	%% this is meant to be used as a legend
end

xlabel_str = 'números primos';

markers_list = {'o','+','x'}; 
color_list= {'blue','red','green'};

for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s K = %d;",algorithms_str{algo_i},n_sources_overall(k_i))
	the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','-')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)
	%legend ("location", "northeastoutside");
	hold on

end


ylabel("média temporal do algoritmo [s]");
xlabel(xlabel_str);
% the_str = sprintf("mean time: P = %d and T= %d",some_primes_overall(p_i),n_samples_overall(t_i));
the_str = sprintf("média temporal, K e T = %d fixos",n_samples_overall(t_i))
title(the_str);
    
set(gca,"xtick",x);
set(gca,"xticklabel",x);


k_i = 5 %% now it's six

for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s K = %d;",algorithms_str{algo_i},n_sources_overall(k_i))
	the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','--')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)
	legend ("location", "northeastoutside");
	hold on

end

k_i = 6 %% now it's seven

for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s K = %d;",algorithms_str{algo_i},n_sources_overall(k_i))
	the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','-.')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)
	legend ("location", "northeastoutside");
	hold on

end

filename = sprintf('varying_P_mean_time_merge_setup1.svg')
print(filename,'-color')

%%%%%%%%%%%%% PLOT OF MEAN TIME VARYING SAMPLES %%%%%%%%%%%%
%%%%%%% FIGURA 1 %%%%%%%%



figure('visible','on','name','varying n_samples, mean time');


sim_data.bss = bss_overall;
sim_data.trial_time = trial_time_overall;
sim_data.total_corr = total_corr_results_overall;
sim_data.n_samples = n_samples_overall;
sim_data.n_sources = n_sources_overall;
sim_data.some_primes = some_primes_overall;


x = n_samples_overall
p_i =  3 %% P = 7 
t_i = 1:length(n_samples_overall) %% n_samples = 4096
k_i = 4 % 5 sources

for i=1:length(algorithms_names);
	algorithms_str{i} = sprintf(";%s",algorithms_names{i});
	%% we will add something after the names
	%% this is meant to be used as a legend
end

xlabel_str = 'log2(número de amostras)';

markers_list = {'o','+','x'}; 
color_list= {'blue','red','green'};




for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s P = %d, K = %d;",algorithms_str{algo_i},some_primes_overall(p_i),n_sources_overall(k_i))
	the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','-')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)
	legend ("location", "northeastoutside");
	hold on

end




ylabel("média temporal do algoritmo [s]");
xlabel(xlabel_str);
% the_str = sprintf("mean time: P = %d and T= %d",some_primes_overall(p_i),n_samples_overall(t_i));
the_str = sprintf("média temporal, K e P = %d fixos",some_primes_overall(p_i))
title(the_str);
    
set(gca,"xtick",x);
set(gca,"xticklabel",log2(x));
%set(gca,'xticklabelrotation',30); %not implemented yet

p_i = 3 %% P=7 does not have this data, so we choose for 5
k_i = 5 %% now it's six

for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s P = %d, K = %d;",algorithms_str{algo_i},some_primes_overall(p_i),n_sources_overall(k_i))
	the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','--')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)
	legend ("location", "northeastoutside");
	hold on

end


k_i = 6 %% now it's seven

for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s P = %d, K = %d;",algorithms_str{algo_i},some_primes_overall(p_i),n_sources_overall(k_i))
	the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','-.')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)

	legend ("location", "northeastoutside");


	hold on

end

filename = sprintf('varying_T_mean_time_merge_setup1.svg')
print(filename,'-color')

%%%%%%%%%% GRÁFICO DE BSS AQUI %%%%%%%%%

%%% ESCOLHI VARIAÇÃO DAS AMOSTRAS

plot_index = 1



sim_data.bss = bss_overall;
sim_data.trial_time = trial_time_overall;
sim_data.total_corr = total_corr_results_overall;
sim_data.n_samples = n_samples_overall;
sim_data.n_sources = n_sources_overall;
sim_data.some_primes = some_primes_overall;


x = n_sources_overall
%% p_i is in loop
t_i = 1 %% n_samples = 256
k_i = 1:length(n_sources_overall)

xlabel_str = 'número de fontes';

for i=1:length(algorithms_names);
	algorithms_str{i} = sprintf(";%s",algorithms_names{i});
	%% we will add something after the names
	%% this is meant to be used as a legend
end
markers_list = {'o','+','x'}; 
color_list= {'blue','red','green'};

figure('visible','on','name','varying K, many plots, BSS_succ');

t_i = 1
for plot_index=1:4

	subplot(2,2,plot_index)


	for algo_i = 1:length(algorithms_str)
		for p_i = 1:length(some_primes_overall)
			if p_i == 1 %%% I just want one legend...
				the_legend = sprintf("%s;",algorithms_str{algo_i})
				the_plot = plot(x,sim_data.bss(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
			else
				the_plot = plot(x,sim_data.bss(p_i,k_i,t_i,algo_i),"marker",markers_list{algo_i},"color",color_list{algo_i});			
			end
			set(the_plot,'markersize',13)
			set(the_plot,'linewidth',2);
			legend("show")
			legend("boxoff")
			set(legend,'fontsize',15)

			if t_i ~= 5
				legend("hide")
				
			end

			legend ("location", "northeastoutside");
			
			hold on
		end

	end


	ylabel("Taxa de sucesso, BSS");
	xlabel(xlabel_str);
	% the_str = sprintf("mean time: P = %d and T= %d",some_primes_overall(p_i),n_samples_overall(t_i));
	the_str = sprintf("cada P e T = %d fixos",n_samples_overall(t_i))
	title(the_str);


	% legend("america","sa4ica","GLICA")
	% legend ("location", "eastoutside");
    
	set(gca,"xtick",x);
	set(gca,"xticklabel",x);
	t_i +=1
	if t_i == 3 %% we dont want 1024 samples to show.
		t_i+=1
	end
end

filename = sprintf('varying_K_bss_succ_merge_setup1.svg')
print(filename,'-color')


%%% Este é o gráfico de TOTAL CORRELATION %%%
figure("visible","on")

p_i = 3 % prime is 5, seven wasn't measured
t_i = 5 % 4096 samples
x = n_sources_overall
k_i = 1:length(n_sources_overall)
for algo_i=1:length(algorithms_str)
	the_legend = sprintf("%s;",algorithms_str{algo_i})
	the_plot = plot(x, sim_data.total_corr(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color", color_list{algo_i} );
	
	set(the_plot,'markersize',13)
	set(the_plot,'linewidth',2);
	set(gca,'fontsize',15)

	hold on
end

the_str = sprintf("correlação total, P = %d, T = %d",some_primes_overall(p_i),n_samples_overall(t_i))
title(the_str)
ylabel("correlação total [bits]");
xlabel(xlabel_str)

set(legend,'fontsize',15)
legend("location","northeastoutside")
%%legend("boxoff")


% legend("location","west")
filename = sprintf('varying_K_total_corr_merge_setup1.svg')
print(filename,'-color')

disp("OBS: IT WILL NOT GENERATE NON-ASCII FONTS, THE EDIT WAS MANUAL ON .SVG FILE")