% Este é um código inspirado no supercomparativo.m do prof. Daniel Guerreiro.
% 17/04/2019

clear;


sim_start_time = localtime(time());

some_primes = [2,3,5,7];
n_sources = [2,3,4,5];
n_samples = [2^8,2^9,2^10,2^11,2^12];

% Must be a cel array, so we can do a strfind...
algorithms_names = {'america';'sa4ica';'GLICA'}

the_algorithms = 1:length(algorithms_names);

n_trials = 40;

space = [ length(some_primes), length(n_sources), length(n_samples),...
 length(the_algorithms) ];

n_cases = prod(space);

bss_succ_rate = zeros( [ space ] );

bss_trial_time = zeros( [ size(bss_succ_rate), n_trials] );

total_corr_results = zeros( [size(bss_trial_time) ] );

% this threshold is such that
% there is a gap of "threshold" in the kullback-leibler divergence
% it is because uniform distributions are unsolvable.
threshold = 0.2;


total_time = tic;


for p_i = 1:length(some_primes)
	for k_i = 1:length(n_sources)
%        The code below was for debugging
%        all_possible_tuples = generate_pai_P(n_sources(k_i),some_primes(p_i))'
%        apt = all_possible_tuples;
		for t_i = 1:length(n_samples)

			P = some_primes(p_i)
			K = n_sources(k_i)
			Nobs = n_samples(t_i)
     		fprintf('\n');

            
			for trial=1:n_trials

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



				%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


				parameters.P = P;
				parameters.K = K;
				parameters.PK = PK;
				parameters.Lex = Lex;
				parameters.r = r;

				% from Daniel code
				% gives the pmfs of each source and generate the Samples
				[the_pmfs, S] = geravetorsinais(P,1,K,threshold,Nobs);

				% generates the mixing_matrix/mix_matrix
				A = geramatrizmistura(P,1,[],K);

				% the miXed Samples
				X = produtomatrizGF(A,S,P,1,[]);
               

		        % calculo tensor de probabilidades - AMERICA
		        idx=r*X;
		        Px = single(zeros(1,PK));
		        for t=1:Nobs
		            Px(idx(t)+1) = Px(idx(t)+1) + 1; 
		        end

		        Px=Px/Nobs;
		        
		        h_joint=-sum(Px(Px>0).*log2(Px(Px>0)));

	        	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		        %%%%% AMERICA ALGORITHM EVALUATION %%%%
		        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                algo_i = find(strcmp(algorithms_names,'america'));
                
		        start_time = tic;
		        [Wm] = america(Px,parameters);
		        bss_trial_time(p_i,k_i,t_i,algo_i,trial) = toc(start_time);
		        
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
		        	bss_succ_rate(p_i,k_i,t_i,algo_i) = bss_succ_rate(p_i,k_i,t_i,algo_i) +1;
		        end 

		        Y = produtomatrizGF(Wm,X,P,1,[]);
%                Y_possible_tuples_america = produtomatrizGF(U_america,all_possible_tuples,P,1,[])

                % estimate_marg_probs(Y,P)' must be alike with 'the_pmfs'
                
		        h_marg=entropy_from_frequencies(estimate_marg_probs(Y,P)');
		        h_marg = sum(h_marg(:));
		        total_corr_results(p_i,k_i,t_i,algo_i,trial) = h_marg-h_joint;


	        	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		        %%%%% SA4ICA ALGORITHM EVALUATION %%%%%
		        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                algo_i = find(strcmp(algorithms_names,'sa4ica'));
                
		        start_time = tic;
                
                % the decode_ function, makes a confusion with a pre-existing
                % decode function. so I renamed it to decode_. It is part of sa4ica
		        [Wsa] = sa4ica_decode(Px,parameters,0.995,5);
				bss_trial_time(p_i,k_i,t_i,algo_i,trial) = toc(start_time);

				%once again, we are going to test the number of hits
		        U = produtomatrizGF(Wsa,A,P,1,[]);
%                Usa4ica = U
                
		        if(1>1)
		            Z = (U>-1); %null element in GF(q^m) 
		        else
		            Z = (U>0); 
		        end

		        hits = sum(sum(Z,2)==1);    
		        if(hits == K)  
		        	bss_succ_rate(p_i,k_i,t_i,algo_i) = bss_succ_rate(p_i,k_i,t_i,algo_i) +1;
		        end 

                Y = produtomatrizGF(Wsa,X,P,1,[]);
%                Y_possible_tuples_sa4ica = produtomatrizGF(Usa4ica,all_possible_tuples,P,1,[])
		        h_marg=entropy_from_frequencies(estimate_marg_probs(Y,P)');
		        h_marg = sum(h_marg(:));
		        total_corr_results(p_i,k_i,t_i,algo_i,trial) = h_marg-h_joint;

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
            
                algo_i = find(strcmp(algorithms_names,'GLICA'));
            
                start_time = tic;
		        [Wglica] = GLICA_function(X,parameters);
		        bss_trial_time(p_i,k_i,t_i,algo_i,trial) = toc(start_time);
                
                %once again, we are going to test the number of hits
		        U = produtomatrizGF(Wglica,A,P,1,[]);
%                Uglica = U

		        if(1>1)
		            Z = (U>-1); %null element in GF(q^m) 
		        else
		            Z = (U>0); 
		        end

		        hits = sum(sum(Z,2)==1);    
		        if(hits == K)  
		        	bss_succ_rate(p_i,k_i,t_i,algo_i) = bss_succ_rate(p_i,k_i,t_i,algo_i) + 1;
		        end 
                
                
                
                Y = produtomatrizGF(Wglica,X,P,1,[]);
%                Y_possible_tuples_glica = produtomatrizGF(Uglica,all_possible_tuples,P,1,[])
		        h_marg=entropy_from_frequencies(estimate_marg_probs(Y,P)');
		        h_marg = sum(h_marg(:));
		        total_corr_results(p_i,k_i,t_i,algo_i,trial) = h_marg-h_joint;
                
                
		  	end 
	  		
            





		end
	end
end
toc(total_time)

mean_bss_succ_rate = bss_succ_rate/n_trials;
mean_bss_trial_time = mean(bss_trial_time, 5);
mean_total_corr_results = mean(total_corr_results, 5);

% saves with the date (day/month/year) and the hour: hh:mm 
% start and ending times
start_time_str = strftime('sim_data_start_%d_%m_%Y_%H_%M',sim_start_time);
saved_sim_str = strftime('_end_%d_%m_%Y_%H_%M_sim_bss_and_total_corr_17_04_2019',localtime(time()));

saved_sim = sprintf('%s%s',start_time_str,saved_sim_str);
save(saved_sim)





