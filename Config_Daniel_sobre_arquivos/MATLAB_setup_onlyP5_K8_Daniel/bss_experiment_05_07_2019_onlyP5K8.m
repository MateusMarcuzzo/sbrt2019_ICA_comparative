% Este ?? um c??digo inspirado no supercomparativo.m do prof. Daniel Guerreiro.
% 17/04/2019

% This is the second different setup1. To see original setup, you must use
% runs_sim_bss_and_total_corr_17_04_2019.m

% onlyP5_K8_Daniel
% 05/07/2019

%16/09/2019
% some documentation needed. This is just the Matlab version for the setup1
% it registers each trial for algorithm.
% some setups in the past just registered success in means, not for each trial.


clear;
%if(isempty(gcp('nocreate')))
%    parpool(2);
%end


% sim_start_time = localtime(time());
sim_start_time = datetime('now');

%we wont do P=7 and K=6 and above anymore. It has frozen the computer
% starting from K=6. (on Mateus computer)

some_primes = [5];
n_sources = [8];
n_samples = 2.^(8:12);

% Must be a cel array, so we can do a strfind...
algorithms_names = {'america';'sa4ica';'GLICA'}

the_algorithms = 1:length(algorithms_names);


n_trials = 40;

space = [ length(some_primes), length(n_sources), length(n_samples), n_trials];

n_cases = prod(space);

bss_succ_rate_america = zeros(1,n_cases);
bss_succ_rate_sa4ica = zeros(1,n_cases);
bss_succ_rate_glica = zeros(1,n_cases);

bss_trial_time_america = zeros(1,n_cases);
bss_trial_time_sa4ica = zeros(1,n_cases);
bss_trial_time_glica = zeros(1,n_cases);

total_corr_results_america = zeros(1,n_cases);
total_corr_results_sa4ica = zeros(1,n_cases);
total_corr_results_glica = zeros(1,n_cases);

% this threshold is such that
% there is a gap of "threshold" in the kullback-leibler divergence
% it is because uniform distributions are unsolvable.
threshold = 0.2;


total_time = tic;

for id_case=1:n_cases
	[p_i, k_i, t_i, trial] = ind2sub(space,id_case);


	P = some_primes(p_i)
	K = n_sources(k_i)
	Nobs = n_samples(t_i)

	PK = P^K;
	%%%%%%% from super_comparativo.m %%%%%%
	%% Which was directly extracted from america code
	%% itself

	% construct a "Lexicon": Lex(:,n) is the (n-1)-th
	Lex=single(zeros(K,PK));
	nvec=single(0:PK-1);
	for k=1:K-1
		Lex(k,:)=mod(nvec,P);
		nvec=floor(nvec/P);
	end
	Lex(K,:)=nvec;
	%construct an "index-vector translation" vector:
	r=P.^(0:K-1); %AMERICA parameter

	% from Daniel code
	% gives the pmfs of each source and generate the Samples
	[the_pmfs, S] = geravetorsinais(P,1,K,threshold,Nobs);

	% generates the mixing_matrix/mix_matrix
	A = geramatrizmistura(P,1,[],K);

	% the miXed Samples
	X = produtomatrizGF(A,S,P,1,[]);


        % calculo tensor de probabilidades - AMERICA
        idx=r*X;
        Px = zeros(1,PK,'single');
        for t=1:Nobs
            Px(idx(t)+1) = Px(idx(t)+1) + 1;
        end

        Px=Px/Nobs;

        h_joint=-sum(Px(Px>0).*log2(Px(Px>0)));

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% AMERICA ALGORITHM EVALUATION %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                 algo_i = find(strcmp(algorithms_names,'america'));

        start_time = tic;
        [Wm] = america(Px, P, K, PK, Lex, r);
        bss_trial_time_america(id_case) = toc(start_time);

        % This is the demixing matrix, we are going to do U*A
        % and check the number of hits. If everyone hits, we count it
        U = produtomatrizGF(Wm,A,P,1,[]);
%                U_america = U

        % there's some legacy code happening here, we'll leave it here.
        % it resembles supercomparativo.m code
        if(1>1)
            Z = (U>-1); %null element in GF(P^1)
        else
            Z = (U>0);
        end

        hits = sum(sum(Z,2)==1);
        if(hits == K)
        	bss_succ_rate_america(id_case) = 1;
        end

        Y = produtomatrizGF(Wm,X,P,1,[]);
                
        h_marg=entropy_from_frequencies(estimate_marg_probs(Y,P)',2);
        h_marg = sum(h_marg(:));
        total_corr_results_america(id_case) = h_marg-h_joint;


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% SA4ICA ALGORITHM EVALUATION %%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% algo_i = find(strcmp(algorithms_names,'sa4ica'));

        start_time = tic;

        % the decode_ function, makes a confusion with a pre-existing
        % decode function. so I renamed it to decode_. It is part of sa4ica
        [Wsa] = sa4ica_decode(Px,r,P,K,Lex,0.995,5);
	bss_trial_time_sa4ica(id_case) = toc(start_time);

	%once again, we are going to test the number of hits
        U = produtomatrizGF(Wsa,A,P,1,[]);
	% Usa4ica = U

        if(1>1)
            Z = (U>-1); %null element in GF(q^m)
        else
            Z = (U>0);
        end

        hits = sum(sum(Z,2)==1);
        if(hits == K)
        	bss_succ_rate_sa4ica(id_case) = 1;
        end

        Y = produtomatrizGF(Wsa,X,P,1,[]);
%       Y_possible_tuples_sa4ica = produtomatrizGF(Usa4ica,all_possible_tuples,P,1,[])
	h_marg=entropy_from_frequencies(estimate_marg_probs(Y,P)',2);
	h_marg = sum(h_marg(:));
	total_corr_results_sa4ica(id_case) = h_marg-h_joint;

	        	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		        %%%%%  QICA ALGORITHM EVALUATION  %%%%%
		        %%%%% WE'VE SENT THIS TO EXPERIMENT 2 %
		        %%%%%        WHICH IS PURE ICA     %%%%
		        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                 start_time = tic;

%                  % this is the gradient descent approach. the exhaustive may need too much time
% 		       	[opt_est_vals,opt_appox_lin_min2,opt_appox_ent_with_appox_vals2,opt_perm,opt_v_vec]=QICA_function(K,P,Px',1,1,10,1000);
% %                opt_perm
% %		        [opt_p,opt_perm,est_vals,marginals_after]=BICA_function(Px',4,10);
% %                opt_perm'


% 		        bss_trial_time(p_i,k_i,t_i,3,trial) = toc(start_time);

% 		        [Yqica,maxhit,Y_possible_tuples_qica] = mapeiapermutacao(S, X, opt_perm,P,K);
%                 Y_possible_tuples_qica

% 		        if(maxhit == K)
% 		        	bss_succ_rate(p_i,k_i,t_i,3) += 1;
% 		        end


          %       fprintf('maxhit is %d where K = %d\n',maxhit,K);

		        % h_marg=entropy_from_frequencies(estimate_marg_probs(Yqica,P)');
		        % h_marg = sum(h_marg(:));

		        % total_corr_results(p_i,k_i,t_i,3,trial) = h_marg-h_joint;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  GLICA ALGORITHM EVALUATION  %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                 algo_i = find(strcmp(algorithms_names,'GLICA'));

        start_time = tic;
        [Wglica] = GLICA_function(X,P,K);
        bss_trial_time_glica(id_case) = toc(start_time);

        %once again, we are going to test the number of hits
        U = produtomatrizGF(Wglica,A,P,1,[]);
%       Uglica = U

        if(1>1)
            Z = (U>-1); %null element in GF(q^m)
        else
            Z = (U>0);
        end

        hits = sum(sum(Z,2)==1);
        if(hits == K)
        	bss_succ_rate_glica(id_case) = 1;
        end



	Y = produtomatrizGF(Wglica,X,P,1,[]);
%       Y_possible_tuples_glica = produtomatrizGF(Uglica,all_possible_tuples,P,1,[])
        h_marg=entropy_from_frequencies(estimate_marg_probs(Y,P)',2);
        h_marg = sum(h_marg(:));
        total_corr_results_glica(id_case) = h_marg-h_joint;

end
toc(total_time)

bss_succ_rate_america = reshape(bss_succ_rate_america,space);
bss_succ_rate_sa4ica = reshape(bss_succ_rate_sa4ica,space);
bss_succ_rate_glica = reshape(bss_succ_rate_glica,space);

bss_trial_time_america = reshape(bss_trial_time_america,space);
bss_trial_time_sa4ica = reshape(bss_trial_time_sa4ica,space);
bss_trial_time_glica = reshape(bss_trial_time_glica,space);

total_corr_results_america = reshape(total_corr_results_america,space);
total_corr_results_sa4ica = reshape(total_corr_results_sa4ica,space);
total_corr_results_glica = reshape(total_corr_results_glica,space);

mean_bss_succ_rate_america = mean(bss_succ_rate_america,4);
mean_bss_succ_rate_sa4ica = mean(bss_succ_rate_sa4ica,4);
mean_bss_succ_rate_glica = mean(bss_succ_rate_glica,4);

mean_bss_trial_time_america = mean(bss_trial_time_america, 4);
mean_bss_trial_time_sa4ica = mean(bss_trial_time_sa4ica, 4);
mean_bss_trial_time_glica = mean(bss_trial_time_glica, 4);

mean_total_corr_results_america = mean(total_corr_results_america, 4);
mean_total_corr_results_sa4ica = mean(total_corr_results_sa4ica, 4);
mean_total_corr_results_glica = mean(total_corr_results_glica, 4);

% saves with the date (day/month/year) and the hour: hh:mm
% start and ending times
start_time_str = datestr(sim_start_time, 'dd_mmm_yy_HH_MM');
saved_sim_str = datestr(datetime('now'), 'dd_mmm_yy_HH_MM');

saved_sim = sprintf('%s%s%s%s%s','sim_data_start_',start_time_str,'_end_',saved_sim_str,'_bss_experiment_05_07_2019');
save(saved_sim)
