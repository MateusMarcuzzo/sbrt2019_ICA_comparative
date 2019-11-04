%auxiliary routine to randomly generate a KxK mixing matrix over GF(q^m)
%INPUT
%field: non-uniformity threshold
%Nobs: number of observations

%OUTPUT
%probs: PxK probability matrix (pmf) for each source
%S: KxNobs generated sources matrix

%Daniel Guerreiro e Silva - 12/01/2015
function A = geramatrizmistura(q,m,field,K)

P = q^m;
A = round((P-1)*rand(K,K));
A = A + diag(diag(A)==0);
AL = tril(A);
AU = eye(K) + triu(A,1);

if(m>1)%adjustment if non-prime field
    AL = AL - 1;
    AU = AU - 1;
end

A = produtomatrizGF(AU,AL,q,m,field);

end
