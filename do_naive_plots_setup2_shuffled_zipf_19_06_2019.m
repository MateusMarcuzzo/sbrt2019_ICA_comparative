%% THIS IS THE LOAD OF SECOND SETUP with shuffled Zipf, THE PURE ICA EXPERIMENT
%% 15/07/2019
load('sim_data_start_09_07_2019_20_58_end_10_07_2019_09_44_sim_total_corr_pure_ICA_19_06_2019');

for i=1:length(algorithms_names);
	algorithms_str{i} = sprintf(";%s",algorithms_names{i});
end





sim_data.trial_time = mean_trial_time;
sim_data.total_corr = mean_total_corr_results;
sim_data.n_samples = n_samples;
sim_data.n_sources = n_sources;
sim_data.some_primes = some_primes;



 
%% I'm having problems with the plots and utf-8...
%% gonna try my best

%%%%%%%% TRIAL TIME %%%%%%
figure("visible","on")

xlabel_str = 'K (dimensão)';

%% maintaining the order of the markers, QICA is the third
markers_list = {'o','+','*','x'}; 

%% we will maintain the colors like before, QICA is the third
color_list= {'blue','red','magenta','green'} 

p_i = 1 % prime = 2
t_i = 1 % 16k samples
k_i = 1:length(n_sources)

x = n_sources

for algo_i = 1:length(algorithms_str)
		the_legend = sprintf("%s P = %d;",algorithms_str{algo_i},some_primes(p_i))
		the_plot = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
		set(the_plot,'linewidth',2)
		set(the_plot,'linestyle','-')
		set(the_plot,'markersize',13)
		set(gca,'fontsize',15)
		set(legend,'fontsize',15)
		legend ("location", "northeastoutside");
		hold on

end

ylabel("média temporal do algoritmo[s]");
xlabel(xlabel_str);
the_str = sprintf("média temporal: T = %d", n_samples(t_i));
title(the_str);
    
set(gca,"xtick",x);
set(gca,"xticklabel",x);

p_i = 2

for algo_i = 1:length(algorithms_str)
	the_legend = sprintf("%s P = %d;",algorithms_str{algo_i},some_primes(p_i))
	the_plot  = semilogy(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
	set(the_plot,'linewidth',2)
	set(the_plot,'linestyle','--')
	set(the_plot,'markersize',13)
	set(gca,'fontsize',15)
	set(legend,'fontsize',15)
	legend ("location", "northeastoutside");
	hold on

end

%filename = sprintf("varying_K_mean_time_setup2.svg")
%print(filename,"-color")



%%%%%%%%%%% TOTAL CORRELATION %%%%%%%%
p_i = 1 % prime = 2
t_i = 1 % 16k samples
k_i = 1:length(n_sources)

x = n_sources

	figure("visible","on");

	for algo_i = 1:length(algorithms_str)
		the_legend = sprintf("%s P = %d;",algorithms_str{algo_i},some_primes(p_i))
		the_plot = plot(x,sim_data.total_corr(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
		set(the_plot,'linewidth',2);
		set(the_plot,'linestyle','-')
		set(the_plot,'markersize',13)
		set(gca,'fontsize',15)
		set(legend,'fontsize',15)
		legend ("location", "northeastoutside");
		hold on

	end

	ylabel("correlação total [bits]");
	xlabel(xlabel_str);
	the_str = sprintf("correlação total média T = %d", n_samples(t_i));
	title(the_str);
        
	set(gca,"xtick",x);
	set(gca,"xticklabel",x);

	p_i = 2 % prime = 3
	for algo_i = 1:length(algorithms_str)
		the_legend = sprintf("%s P = %d;",algorithms_str{algo_i},some_primes(p_i))
		the_plot = plot(x,sim_data.total_corr(p_i,k_i,t_i,algo_i),the_legend,"marker",markers_list{algo_i},"color",color_list{algo_i});
		set(the_plot,'linewidth',2);
		set(the_plot,'linestyle','--')
		set(the_plot,'markersize',13)
		set(gca,'fontsize',15)
		set(legend,'fontsize',15)
		legend ("location", "northeastoutside");
		hold on

	end

	%filename = sprintf("varying_K_total_corr_setup2.svg")
	%print(filename, "-color")

	disp("OBS: IT WILL NOT GENERATE NON-ASCII FONTS, THE EDIT WAS MANUAL ON .SVG FILE")