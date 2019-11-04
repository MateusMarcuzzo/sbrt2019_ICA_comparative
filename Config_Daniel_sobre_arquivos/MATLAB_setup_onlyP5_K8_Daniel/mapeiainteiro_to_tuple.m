


% Maps from range 1 to P^K it's respective integer to respective tuple
% in component form to effective BSS or ICA task
function [S] = mapeiainteiro_to_tuple(column_wise_matrix,P,K)

	% The -1 is necessary, since 0 is the first number and full element
	% is the last: P^K - 1

	% 48 is 0 in ascii form...
	S = dec2base(column_wise_matrix-1,P,K) - 48;

end