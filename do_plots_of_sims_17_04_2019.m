%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% PLOT GENERATION  %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% THIS IS THE LOAD OF FIRST SETUP, THE BSS EXPERIMENT
load('sim_data_start_12_05_2019_23_50_end_13_05_2019_12_58_sim_bss_and_total_corr_17_04_2019');

for i=1:length(algorithms_names);
	algorithms_str{i} = sprintf(";%s;",algorithms_names{i});
end

plot_time = tic;


sim_data.bss = mean_bss_succ_rate;
sim_data.trial_time = mean_bss_trial_time;
sim_data.total_corr = mean_total_corr_results;
sim_data.n_samples = n_samples;
sim_data.n_sources = n_sources;
sim_data.some_primes = some_primes;

% That's the heart of the loops:
function one_plot = plot_P_case(x,sim_data,p_i,k_i,t_i,algorithms_str)


	xlabel_str = 'the primes';
	first_save_folder = 'sim_plots_setup1';
	bss_save_folder = 'bss';
	trial_time_save_folder = 'trial_time';
	total_corr_save_folder = 'total_corr';
	this_file_date = '15_05_2019';
	markers_list = {'o','d','x','*'}; 

	
	
%%%%%%%%%%%%%%%  BSS SUCC RATE %%%%%%%%%%%%%%%%
	figure("visible","off");

	n_samples = sim_data.n_samples;
	n_sources = sim_data.n_sources;
	some_primes = sim_data.some_primes;

	for algo_i = 1:length(algorithms_str)

		plot(x,sim_data.bss(p_i,k_i,t_i,algo_i),algorithms_str{algo_i},"marker",markers_list{algo_i});
		hold on

	end

	ylabel("total separation rate ");
	xlabel(xlabel_str);
	the_str = sprintf("sucess rate comparison: T = %d and K= %d",n_samples(t_i),n_sources(k_i));
	title(the_str);


	set(gca,"xtick",x);
	set(gca,"xticklabel",x);


	filename = sprintf("%s/%s/P_var/%s_K%d_T%d_suc_rate.eps",first_save_folder,bss_save_folder,...
	 	this_file_date,n_sources(k_i),n_samples(t_i));
		
	print(filename,'-color');
            
%%%%%%%%%%%%%%% TRIAL TIME %%%%%%%%%%%%%%%%%%

	figure("visible","off");

	for algo_i = 1:length(algorithms_str)

		plot(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),algorithms_str{algo_i},"marker",markers_list{algo_i});
		hold on

	end

	ylabel("algorithm mean time [s]");
	xlabel(xlabel_str);
	the_str = sprintf("mean time: T = %d and K= %d",n_samples(t_i),n_sources(k_i));
	title(the_str);
        
	set(gca,"xtick",x);
	set(gca,"xticklabel",x);


	filename = sprintf("%s/%s/P_var/%s_K%d_T%d_trial_time.eps",first_save_folder,...
	 	trial_time_save_folder,this_file_date...
	 	,n_sources(k_i),n_samples(t_i));

    print(filename,'-color')


%%%%%%%%%%%% %%% Total Correlation %%%%%%%%%%%%%%%%%
	figure("visible","off");

	for algo_i = 1:length(algorithms_str)

		plot(x,sim_data.total_corr(p_i,k_i,t_i,algo_i),algorithms_str{algo_i},"marker",markers_list{algo_i});
		hold on

	end

	ylabel("total correlation [bits]");
	xlabel(xlabel_str);
	the_str = sprintf("mean total correlation: T = %d and K= %d",n_samples(t_i),n_sources(k_i));
	title(the_str);
        
	set(gca,"xtick",x);
	set(gca,"xticklabel",x);

	filename = sprintf("%s/%s/P_var/%s_K%d_T%d_total_corr.eps",first_save_folder,...
	 	total_corr_save_folder,this_file_date...
	 	,n_sources(k_i),n_samples(t_i));


    print(filename,'-color')



end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function one_plot = plot_T_case(x,sim_data,p_i,k_i,t_i,algorithms_str)


	xlabel_str = 'number of samples';
	first_save_folder = 'sim_plots_setup1';
	bss_save_folder = 'bss';
	trial_time_save_folder = 'trial_time';
	total_corr_save_folder = 'total_corr';
	this_file_date = '15_05_2019';
	markers_list = {'o','d','x','*'}; 

	
	
%%%%%%%%%%%%%%%  BSS SUCC RATE %%%%%%%%%%%%%%%%
	figure("visible","off");

	n_samples = sim_data.n_samples;
	n_sources = sim_data.n_sources;
	some_primes = sim_data.some_primes;

	for algo_i = 1:length(algorithms_str)

		plot(x,sim_data.bss(p_i,k_i,t_i,algo_i),algorithms_str{algo_i},"marker",markers_list{algo_i});
		hold on

	end

	ylabel("total separation rate ");
	xlabel(xlabel_str);
	the_str = sprintf("sucess rate comparison: P = %d and K= %d",some_primes(p_i),n_sources(k_i));
	title(the_str);


	set(gca,"xtick",x);
	set(gca,"xticklabel",x);


	filename = sprintf("%s/%s/T_var/%s_K%d_P%d_suc_rate.eps",first_save_folder,bss_save_folder,...
	 	this_file_date,n_sources(k_i),some_primes(p_i));
		
	print(filename,'-color');
            
%%%%%%%%%%%%%%% TRIAL TIME %%%%%%%%%%%%%%%%%%

	figure("visible","off");

	for algo_i = 1:length(algorithms_str)

		plot(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),algorithms_str{algo_i},"marker",markers_list{algo_i});
		hold on

	end

	ylabel("algorithm mean time [s]");
	xlabel(xlabel_str);
	the_str = sprintf("mean time: P = %d and K= %d",some_primes(p_i),n_sources(k_i));
	title(the_str);
        
	set(gca,"xtick",x);
	set(gca,"xticklabel",x);


	filename = sprintf("%s/%s/T_var/%s_K%d_P%d_trial_time.eps",first_save_folder,...
	 	trial_time_save_folder,this_file_date...
	 	,n_sources(k_i),some_primes(p_i));

    print(filename,'-color')


%%%%%%%%%%%% %%% Total Correlation %%%%%%%%%%%%%%%%%
	figure("visible","off");

	for algo_i = 1:length(algorithms_str)

		plot(x,sim_data.total_corr(p_i,k_i,t_i,algo_i),algorithms_str{algo_i},"marker",markers_list{algo_i});
		hold on

	end

	ylabel("total correlation [bits]");
	xlabel(xlabel_str);
	the_str = sprintf("mean total correlation: P = %d and K= %d",some_primes(p_i),n_sources(k_i));
	title(the_str);
        
	set(gca,"xtick",x);
	set(gca,"xticklabel",x);

	filename = sprintf("%s/%s/T_var/%s_K%d_P%d_total_corr.eps",first_save_folder,...
	 	total_corr_save_folder,this_file_date...
	 	,n_sources(k_i),some_primes(p_i));


    print(filename,'-color')



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function one_plot = plot_K_case(x,sim_data,p_i,k_i,t_i,algorithms_str)


	xlabel_str = 'number of sources';
	first_save_folder = 'sim_plots_setup1';
	bss_save_folder = 'bss';
	trial_time_save_folder = 'trial_time';
	total_corr_save_folder = 'total_corr';
	this_file_date = '15_05_2019';
	markers_list = {'o','d','x','*'}; 

	
	
%%%%%%%%%%%%%%%  BSS SUCC RATE %%%%%%%%%%%%%%%%
	figure("visible","off");

	n_samples = sim_data.n_samples;
	n_sources = sim_data.n_sources;
	some_primes = sim_data.some_primes;

	for algo_i = 1:length(algorithms_str)

		plot(x,sim_data.bss(p_i,k_i,t_i,algo_i),algorithms_str{algo_i},"marker",markers_list{algo_i});
		hold on

	end

	ylabel("total separation rate ");
	xlabel(xlabel_str);
	the_str = sprintf("sucess rate comparison: P = %d and T= %d",some_primes(p_i),n_samples(t_i));
	title(the_str);


	set(gca,"xtick",x);
	set(gca,"xticklabel",x);


	filename = sprintf("%s/%s/K_var/%s_T%d_P%d_suc_rate.eps",first_save_folder,bss_save_folder,...
	 	this_file_date,n_samples(t_i),some_primes(p_i));
		
	print(filename,'-color');
            
%%%%%%%%%%%%%%% TRIAL TIME %%%%%%%%%%%%%%%%%%

	figure("visible","off");

	for algo_i = 1:length(algorithms_str)

		plot(x,sim_data.trial_time(p_i,k_i,t_i,algo_i),algorithms_str{algo_i},"marker",markers_list{algo_i});
		hold on

	end

	ylabel("algorithm mean time [s]");
	xlabel(xlabel_str);
	the_str = sprintf("mean time: P = %d and T= %d",some_primes(p_i),n_samples(t_i));
	title(the_str);
        
	set(gca,"xtick",x);
	set(gca,"xticklabel",x);


	filename = sprintf("%s/%s/K_var/%s_T%d_P%d_trial_time.eps",first_save_folder,...
	 	trial_time_save_folder,this_file_date...
	 	,n_samples(t_i),some_primes(p_i));

    print(filename,'-color')


%%%%%%%%%%%% %%% Total Correlation %%%%%%%%%%%%%%%%%
	figure("visible","off");

	for algo_i = 1:length(algorithms_str)

		plot(x,sim_data.total_corr(p_i,k_i,t_i,algo_i),algorithms_str{algo_i},"marker",markers_list{algo_i});
		hold on

	end

	ylabel("total correlation [bits]");
	xlabel(xlabel_str);
	the_str = sprintf("mean total correlation: P = %d and T= %d",some_primes(p_i),n_samples(t_i));
	title(the_str);
        
	set(gca,"xtick",x);
	set(gca,"xticklabel",x);

	filename = sprintf("%s/%s/K_var/%s_T%d_P%d_total_corr.eps",first_save_folder,...
	 	total_corr_save_folder,this_file_date...
	 	,n_samples(t_i),some_primes(p_i));


    print(filename,'-color')



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





  for k_i = 1:length(n_sources)
  	for t_i = 1:length(n_samples)
            
         p_i=1:length(some_primes);
%          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          %%%%%% varying some_primes first %%%%%
%          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
  		x=some_primes;

  		plot_P_case(x,sim_data,p_i,k_i,t_i,algorithms_str);
 

     end
  end

 for p_i=1:length(some_primes)
 	for k_i =1:length(n_sources)

 		t_i = 1:length(n_samples);
% 		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        %%%%%%   varying n_samples now   %%%%%
%        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		x=n_samples;

		plot_T_case(x,sim_data,p_i,k_i,t_i,algorithms_str);


 	end

 end

for p_i=1:length(some_primes)
	for t_i =1:length(n_samples)

 		k_i = 1:length(n_sources);
% 		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 		%%%%%%   varying n_sources now   %%%%%
% 		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		x=n_sources;

		plot_K_case(x,sim_data,p_i,k_i,t_i,algorithms_str);

 	end

 end

 toc(plot_time)