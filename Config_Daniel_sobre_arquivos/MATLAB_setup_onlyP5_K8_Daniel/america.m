function B=america(PPx,P, K, PK, Lex, r)
%The Ascending Minimization of EntRopies for ICA
%(AMERICA) algorithm
%input: PPx - the (estimated) probabilities tensor
%output: B - the estimated separating matrix

% global P K PK r Lex

% P = param.P;
% K = param.K;
% PK = param.PK;
% Lex = param.Lex;
% r = param.r;

eqeps=1e-9; %a threshold for deciding
            %equal entropies


%K-D fft for obtaining the characteristic tensor
fPPyn=fftn(reshape(PPx,P*ones(1,K)));
fPPy=fPPyn(:);
        
%obtain the characteristic vectors of 
%the linear combinations
qf=ones(P,PK);
qf(2,:)=fPPy;
if P>2
    qf(P,:)=conj(fPPy);
    for m=2:P/2
        mLex=mod(m*Lex,P);
        qf(m+1,:)=fPPy(r*mLex+1);
        qf(P+1-m,:)=conj(qf(m+1,:));
    end
end

%translate characteristic vectors into probabilities
%vectors and then into entropies
ffq=ifft(qf);
ffq=max(ffq,eps);
h=-sum(ffq.*log2(ffq+eps),1);
%mark irrelevant entropies (such as the one related
%to the all-zeros (trivial) combination, and subsequent
%"used" entropies - with a NaN
h(1)=NaN;

B=[];
k=1;
%sorted entropies (ascending order)
[sh shix]=sort(h);
inh=1;
while k<=K
    vh=sh(inh);
    mix=shix(inh);
    for itry=inh+1:PK
        if abs(sh(itry)-vh)>eqeps, break; end
    end
    %randomized selection in case of a tie
    neq=itry-inh;
    if neq>1
        ipick=floor(rand*neq);
        pinh=inh+ipick;
        tmph=sh(inh);
        tmpi=shix(inh);
        sh(inh)=sh(pinh);
        shix(inh)=shix(pinh);
        sh(pinh)=tmph;
        shix(pinh)=tmpi;
    end
    %test if the selected is not a linear combination
    %of the previous ones
    mix=shix(inh);
    b=Lex(:,mix);
    Bb=[B b];
    TLex=Lex(1:k,2:P^k);
    test0=mod(Bb*TLex,P);
    if ~any(sum(test0,1)==0)    %not a linear combination
        B=Bb;   
        k=k+1;
    end
    inh=inh+1;
end

B=B';