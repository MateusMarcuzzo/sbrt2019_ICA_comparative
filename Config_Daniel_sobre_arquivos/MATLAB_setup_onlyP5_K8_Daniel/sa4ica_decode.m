

%SA4ICA Probabilities Tensor version - Prime fields only!
%Good for large number of samples - 2^11 or larger!!!
function [B] = sa4ica_decode(Px,r,q,K,lex,beta,k)

epsilon = 1e-3;

% N = K;
% q = parameters.P;

B = eye(K,K); %initial solution
[h, ~] = decode_(eye(K),Px,r,q,K,lex); %initial entropies


T = 1; %Kirckpatrick

while T>epsilon
    ij = round((K-1)*rand(2,k)) + 1;
    c = round((q-2)*rand(1,k)) + 1;
    for it=1:k
        %generate a random moveZ
        V = eye(K);
        V(ij(1,it),ij(2,it)) = c(it);  %switch Xi by the combination Xi + c.Xj

        [hnew, Pxnew] = decode_(V, Px, r,q,K,lex);
        hnew = hnew(ij(1,it));
        delta_H = hnew - h(ij(1,it));
        %update candidate solution
        if(delta_H<0 || exp(-delta_H/T) > rand())
            B = produtomatrizGF(V,B,q,1,[]);
            Px = Pxnew;
            h(ij(1,it)) = hnew;
        end
    end
    T = beta * T;
%     fprintf(1,'%d %.4f\n', Nit, T);
end
end
