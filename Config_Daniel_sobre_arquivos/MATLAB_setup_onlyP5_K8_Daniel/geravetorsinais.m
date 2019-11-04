%auxiliary routine to generate K non-uniform sources over GF(q^m)
%INPUT
%K: number of sources
%thre: non-uniformity threshold
%Nobs: number of observations

%OUTPUT
%probs: PxK probability matrix (pmf) for each source
%S: KxNobs generated sources matrix

%Daniel Guerreiro e Silva - 12/01/2015
function [probs,S] = geravetorsinais(q,m,K,thre,Nobs)

P = q^m;

probs = zeros(P,K);
for k=1:K
    prob=rand(P,1);
    sprob=sum(prob);
    prob=prob./sprob(ones(P,1),:);
    KLD = 1 + prob'*(log(prob)./log(P)); %Kullback-Leibler Divergence
    while(KLD<thre || max(prob)>.98 || min(prob)==0) %non-uniformity and non-degenerate requirement
        prob=rand(P,1);
        sprob=sum(prob);
        prob=prob./sprob(ones(P,1),:);
        KLD = 1 + prob'*(log(prob)./log(P));
    end
    probs(:,k)=prob;
end
cprobs=cumsum(probs);
RND=rand(K,Nobs);
S=single(zeros(K,Nobs));
for k=1:K
    for cp=1:P-1
        S(k,:)=S(k,:)+(RND(k,:)>cprobs(cp,k));
    end
end

if(m>1)
    S = S - 1;
end

end
