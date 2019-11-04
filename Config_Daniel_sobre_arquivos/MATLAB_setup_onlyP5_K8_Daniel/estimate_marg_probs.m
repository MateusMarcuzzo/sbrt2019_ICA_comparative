function marg_probs = estimate_marg_probs(Y,P)
% This function estimates the marginal probabilities given the
% Y matrix, of the observations. It must known which prime P its
% working

	marg_probs= [];
	for symbol=0:(P-1)
		marg_probs = [marg_probs mean(Y==symbol,2)];
	end

end

