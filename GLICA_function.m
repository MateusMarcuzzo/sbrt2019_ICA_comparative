function W = GLICA_function(X,param)
	% this fuction seems really closer to america
	% and was created by Painsky et.al
	% We'll do an implementation accordingly to
	% the original paper: 
	% "Linear Independent Component Analysis over
	% Finite Fields: Algorithms and Bounds"

	% X is the observations matrix. which is K x Nobs
	% output: the separation matrix W

	P = param.P;
	K = param.K;
	PK = param.PK;
	
	% Nothing is said about Lex in Painsky's work
	% Lex = param.Lex;

    
	eqepes = 1e-9;


	vT_matrix = generate_pai_P(K,P);
    
    %america does in this order and uses Lex to reference this elements
    % flipud(vT_matrix)
    
    
	U = produtomatrizGF(vT_matrix , X,P,1,[]);
    
    marg_probs = estimate_marg_probs(U,P)';
    U_entropies = entropy_from_frequencies(marg_probs)';
    U_entropies(end) = NaN;
    
    
    [U_entropies_sorted U_entropies_sorted_index] = sort(U_entropies);
    
    
    index_entropy = 1;
    W = [];
    k=1;
    while k<=K

        if isempty(W)
                W = [ vT_matrix(U_entropies_sorted_index,:)(index_entropy,:)];
                index_entropy += 1;
                k+=1;
                continue;
        end
        
        row_candidate = vT_matrix(U_entropies_sorted_index,:)(index_entropy,:);
        
        % checks if the row candidate is a combination of pre-existing lines
        existing_lines = size(W,1);
        
        
        vT_matrix_view = vT_matrix(1:P^existing_lines,1+K-existing_lines:end);
        
        %excluding zero
        vT_matrix_view = vT_matrix_view(1:end-1,:);

        %if the row_candidate is NOT a linear combination...
        % of the existing rows in W... add it to the W matrix
        if(~ any(sum(abs(mod(vT_matrix_view * W,P) - row_candidate),2) ==0) )
            W = [W ; row_candidate];
            k+=1;
        end
        index_entropy+=1;

    end

end


