function v = allVL1nonrecurs(n, L1)
% function v=allVL1eq(n, L1);
% INPUT
%    n: length of the vector
%    L1: desired L1 norm
% OUTPUT:
%    if head is not defined
%      v: (m x n) array such as sum(v,2)==L1
%         all elements of v is naturel numbers {0,1,...}
%         v contains all (=m) possible combinations
%         v is (dictionnary) sorted
% Algorithm:
%    NonRecursive

% Chose (n-1) the splitting points of the array [0:(n+L1)]
s = nchoosek(1:n+L1-1,n-1);
m = size(s,1);

s1 = zeros(m,1,class(L1));
s2 = (n+L1)+s1;

v = diff([s1 s s2],1,2); % m x n
v = v-1;

end % allVL1nonrecurs
