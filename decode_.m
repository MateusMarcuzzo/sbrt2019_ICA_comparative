%decode function
%PPx: probabilities tensor 
function [h,PPy] = decode_(W,PPx,parameters)

    PPy = PPx;
    r = parameters.r;
    q = parameters.P;
    K = parameters.K;
    lex = parameters.Lex;
%     global r q K lex;
%     K = length(W);
    lg_cte = log2(q); %correction factor to calculate always logP entropies
%     r=q.^(0:K-1);
    
    %K-D fft for obtaining the characteristic tensor
    fPPyn=fftn(reshape(PPx,q*ones(1,K)));
    fPPy=fPPyn(:);
        
    %obtain the characteristic vectors of 
    %the linear combinations
    qf=ones(q,q^K);
    qf(2,:)=fPPy;
    if q>2
        qf(q,:)=conj(fPPy);
        for m=2:q/2
            mLex=rem(m*lex,q);
            qf(m+1,:)=fPPy(r*mLex+1);
            qf(q+1-m,:)=conj(qf(m+1,:));
        end
    end
    
    %translate characteristic vectors into probabilities
    %vectors and then into entropies
    ffq=ifft(qf);
    ffq=max(ffq,eps);
    h=-sum(ffq.*log2(ffq+eps),1);
    h = h(W*r'+1)./lg_cte;
    
    WLex = rem(W*lex, q);
    PPy(r*WLex+1) = PPx;%new prob tensor after W transform.
    
%     lgP = log(p)./lg_cte;
%     Pzero = (p==0);
%     PlgP = p.*lgP;
%     PlgP(Pzero) = 0;
%     h = sum(PlgP,2);
    
%     h = -p.*log2(p) - (1-p).*log2(1-p); %binary marginal entropies
%     h = sum(h);
          
%     v(it) = 1-(h/M);    


end