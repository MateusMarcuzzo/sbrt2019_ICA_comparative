%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% PLOT GENERATION  %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% MUST FIRST LOAD THE FILE IN TERMINAL BEFORE EXECUTING THE CODE BELOW %%%
%%%%%%%%%%% MUST REFACTOR STILL, needs some modifications for EFECTIVE DRY APPLYING 
%%%%%%%%% ^^^^ TO-DO 09/05/2019

america_str = sprintf(";AMERICA;");
sa4ica_str = sprintf(";SA4ICA;");
qica_str = sprintf(";QICA;");
glica_str = sprintf(";GLICA;");


plot_time = tic;

plot_index=1;
 for k_i = 1:length(n_sources)
 	for t_i = 1:length(n_samples)
        
        p_i=1:length(some_primes);
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %%%%%% varying some_primes first %%%%%
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         %%% SUCC RATE %%% 
 		figure("visible","off");
 		subplot( 2,2, plot_index)
 		x=some_primes;

 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,1),america_str,"marker",'o');
 		hold on
 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,2),sa4ica_str,"marker",'d');
 		hold on
 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,3),qica_str,"marker",'x');
 		hold on
 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,4),glica_str,"marker",'*');
 		hold on

 		ylabel("total separation rate ");
 		xlabel("the primes");
 		the_str = sprintf("sucess rate comparison: T = %d and K= %d",n_samples(t_i),n_sources(k_i));
 		title(the_str);

 		set(gca,"xtick",x);
 		set(gca,"xticklabel",x);

 		% filename = sprintf("sim_plots/22_04_2019_K%d_T%d_suc_rate.pdf",n_sources(k_i),n_samples(t_i));
 		% print(filename);
	            
         %%% TRIAL TIME %%%%
        subplot( 2,2, plot_index+1)

 		plot(x,mean_bss_trial_time(p_i,k_i,t_i,1),america_str,"marker",'o');
 		hold on
 		plot(x,mean_bss_trial_time(p_i,k_i,t_i,2),sa4ica_str,"marker",'d');
 		hold on
 		plot(x,mean_bss_trial_time(p_i,k_i,t_i,3),qica_str,"marker",'x');
 		hold on
 		plot(x,mean_bss_trial_time(p_i,k_i,t_i,4),glica_str,"marker",'*');
 		hold on

 		ylabel("algorithm mean time [s]");
 		xlabel("the primes");
 		the_str = sprintf("mean time: T = %d and K= %d",n_samples(t_i),n_sources(k_i));
 		title(the_str);
                
 		set(gca,"xtick",x);
 		set(gca,"xticklabel",x);

 		% filename = sprintf("sim_plots/22_04_2019_K%d_T%d_trial_time.pdf",n_sources(k_i),n_samples(t_i));
 		% print(filename);
	    
         %%% Total Correlation %%%%
         subplot( 2,2, plot_index+2)

 		plot(x,mean_total_corr_results(p_i,k_i,t_i,1),america_str,"marker",'o');
 		hold on
 		plot(x,mean_total_corr_results(p_i,k_i,t_i,2),sa4ica_str,"marker",'d');
 		hold on
 		plot(x,mean_total_corr_results(p_i,k_i,t_i,3),qica_str,"marker",'x');
 		hold on
 		plot(x,mean_total_corr_results(p_i,k_i,t_i,4),glica_str,"marker",'*');
 		hold on

 		ylabel("total correlation [bits]");
 		xlabel("the primes");
 		the_str = sprintf("mean total correlation: T = %d and K= %d",n_samples(t_i),n_sources(k_i));
 		title(the_str);
                
 		set(gca,"xtick",x);
 		set(gca,"xticklabel",x);

 		filename = sprintf("sim_plots/K_T/22_04_2019_K%d_T%d_.pdf",n_sources(k_i),n_samples(t_i));
 		print(filename);



     end
 end

for p_i=1:length(some_primes)
	for k_i =1:length(n_sources)

		t_i = 1:length(n_samples);
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%%%%%   varying n_samples now   %%%%%
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       %%% SUCC RATE %%% 
		figure("visible","off");
		subplot( 2,2, plot_index)
		x=n_samples;

 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,1),america_str,"marker",'o');
 		hold on
 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,2),sa4ica_str,"marker",'d');
 		hold on
 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,3),qica_str,"marker",'x');
 		hold on
 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,4),glica_str,"marker",'*');
 		hold on

		ylabel("total separation rate ");
		xlabel("number of samples");
		the_str = sprintf("sucess rate comparison: P = %d and K= %d",some_primes(p_i),n_sources(k_i));
		title(the_str);

		set(gca,"xtick",x);
		set(gca,"xticklabel",x);

		% filename = sprintf("sim_plots/22_04_2019_K%d_P%d_suc_rate.pdf",n_sources(k_i),some_primes(p_i));
		% print(filename);
	            
       %%% TRIAL TIME %%%%
       subplot( 2,2, plot_index+1)

		plot(x,mean_bss_trial_time(p_i,k_i,t_i,1),america_str,"marker",'o');
		hold on
		plot(x,mean_bss_trial_time(p_i,k_i,t_i,2),sa4ica_str,"marker",'d');
		hold on
		plot(x,mean_bss_trial_time(p_i,k_i,t_i,3),qica_str,"marker",'x');
		hold on
		plot(x,mean_bss_trial_time(p_i,k_i,t_i,4),glica_str,"marker",'*');
		hold on

		ylabel("algorithm mean time [s]");
		xlabel("number of samples");
		the_str = sprintf("mean time: P = %d and K= %d",some_primes(p_i),n_sources(k_i));
		title(the_str);
               
		set(gca,"xtick",x);
		set(gca,"xticklabel",x);

		% filename = sprintf("sim_plots/22_04_2019_P%d_K%d_trial_time.pdf",some_primes(p_i),n_sources(k_i));
		% print(filename);
	    
       %%% Total Correlation  %%%%
       subplot( 2,2, plot_index+2)

		plot(x,mean_total_corr_results(p_i,k_i,t_i,1),america_str,"marker",'o');
		hold on
		plot(x,mean_total_corr_results(p_i,k_i,t_i,2),sa4ica_str,"marker",'d');
		hold on
		plot(x,mean_total_corr_results(p_i,k_i,t_i,3),qica_str,"marker",'x');
		hold on
		plot(x,mean_total_corr_results(p_i,k_i,t_i,4),glica_str,"marker",'*');
		hold on

		ylabel("total correlation [bits]");
		xlabel("number of samples");
		the_str = sprintf("mean total correlation: P = %d and K= %d",some_primes(p_i),n_sources(k_i));
		title(the_str);
               
		set(gca,"xtick",x);
		set(gca,"xticklabel",x);

		filename = sprintf("sim_plots/K_P/22_04_2019_P%d_K%d_.pdf",some_primes(p_i),n_sources(k_i));
		print(filename);



	end

end

for p_i=1:length(some_primes)
	for t_i =1:length(n_samples)

		k_i = 1:length(n_sources);
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%   varying n_sources now   %%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       	%%% SUCC RATE %%% 
		figure("visible","off");
		subplot( 2,2, plot_index)
		x=n_sources;

 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,1),america_str,"marker",'o');
 		hold on
 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,2),sa4ica_str,"marker",'d');
 		hold on
 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,3),qica_str,"marker",'x');
 		hold on
 		plot(x,mean_bss_succ_rate(p_i,k_i,t_i,4),glica_str,"marker",'*');
 		hold on

		ylabel("total separation rate ");
		xlabel("number of sources");
		the_str = sprintf("sucess rate comparison: P = %d and T= %d",some_primes(p_i),n_samples(t_i));
		title(the_str);

		set(gca,"xtick",x);
		set(gca,"xticklabel",x);

		% filename = sprintf("sim_plots/22_04_2019_P%d_T%d_suc_rate.pdf",some_primes(p_i),n_samples(t_i));
		% print(filename);
	            
       %%% TRIAL TIME %%%%
       	subplot( 2,2, plot_index+1)

		plot(x,mean_bss_trial_time(p_i,k_i,t_i,1),america_str,"marker",'o');
		hold on
		plot(x,mean_bss_trial_time(p_i,k_i,t_i,2),sa4ica_str,"marker",'d');
		hold on
		plot(x,mean_bss_trial_time(p_i,k_i,t_i,3),qica_str,"marker",'x');
		hold on
		plot(x,mean_bss_trial_time(p_i,k_i,t_i,4),glica_str,"marker",'*');
		hold on

		ylabel("algorithm mean time [s]");
		xlabel("number of sources");
		the_str = sprintf("mean time: P = %d and T= %d",some_primes(p_i),n_samples(t_i));
		title(the_str);
               
		set(gca,"xtick",x);
		set(gca,"xticklabel",x);

		% filename = sprintf("sim_plots/22_04_2019_P%d_T%d_trial_time.pdf",some_primes(p_i),n_samples(t_i));
		% print(filename);
	    
		%%% Total Correlation  %%%%
       	subplot( 2,2, plot_index+2)

		plot(x,mean_total_corr_results(p_i,k_i,t_i,1),america_str,"marker",'o');
		hold on
		plot(x,mean_total_corr_results(p_i,k_i,t_i,2),sa4ica_str,"marker",'d');
		hold on
		plot(x,mean_total_corr_results(p_i,k_i,t_i,3),qica_str,"marker",'x');
		hold on
		plot(x,mean_total_corr_results(p_i,k_i,t_i,4),glica_str,"marker",'*');
		hold on

		ylabel("total correlation [bits]");
		xlabel("number of sources");
		the_str = sprintf("mean total correlation: P = %d and T= %d",some_primes(p_i),n_samples(t_i));
		title(the_str);
               
		set(gca,"xtick",x);
		set(gca,"xticklabel",x);

		filename = sprintf("sim_plots/P_T/22_04_2019_P%d_T%d_.pdf",some_primes(p_i),n_samples(t_i));
		print(filename);



	end

end

toc(plot_time)