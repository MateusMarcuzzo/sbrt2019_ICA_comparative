%This routine generates a binomial pmf with cardinality N
% Given the probability of success P
% This routine is used to give a joint_pmf
% to ICA experiments

% Mateus Marcuzzo da Rosa 05/08/2019
% Which is actually not needed, since Octave gives support for
% a function does the same, and faster.
function [probs] = generate_binomial_pmf(N,P_succ)


    probs = zeros(N,1);

    N = N-1;
    % Note that we must use (N-1 k), not the N itself, to
    % do a binomial with support size N

    for i=0:(N)
    	j = i+1;
    	probs(j) = nchoosek(N,i)*(P_succ^i)*(1-P_succ)^(N-i);
	end

       

end