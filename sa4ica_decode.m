

%SA4ICA Probabilities Tensor version - Prime fields only!
%Good for large number of samples - 2^11 or larger!!!
function [B] = sa4ica_decode(Px,parameters,beta,k)

% global count;

% histH = [];
epsilon = 1e-3;

N = parameters.K;
q = parameters.P;
% Nfits = 0;
%flag_parada = 0;
%Ntotal_comb = 0;
B = eye(N,N); %initial solution
[h, ~] = decode_(eye(N),Px,parameters); %initial entropies
% if m > 1
%     B = B - 1;
% end

T = 1; %Kirckpatrick

while T>epsilon

    for it=1:k
        %generate a random move
        i =randi(N);
        j = randi(N);
        V = eye(N);
%         if m > 1            
%             V = V - 1;
%             c = randsrc(1,1,0:q^m-2);
%         else
            c = randi(q-1);
%         end
        V(i,j) = c;  %switch Xi by the combination Xi + c.Xj
        
%         Xnew = produtomatrizGF(V, X, q, m, field);
        [hnew, Pxnew] = decode_(V, Px, parameters);
        hnew = hnew(i);
%         H = entrp([combined_signal; X(i,:)],q,m);
%         [hnew, count] = entrp(Xnew(i,:),q,m);
%         Nfits = Nfits + count;
        delta_H = hnew - h(i);
        %update candidate solution
        if(delta_H<0 || exp(-delta_H/T) > rand())                    
%             B = produtomatrizGF(V,B,q,m,field);
            B = produtomatrizGF(V,B,q,1,[]);
            Px = Pxnew;
            h(i) = hnew;
        end       
    end
%     Nfits = Nfits + k;    
    T = beta * T;    
%     fprintf(1,'%d %.4f\n', Nit, T);
end
% Nfits = count;
end
    

% function [v,count] = entrp(Y,q,m)
% 
% % global count;
% 
% P = q^m;
% 
% [n, Nobs]= size(Y);
% 
% lg_cte = log(P); %correction factor to calculate always logP entropies
% v = ((P-1)/(2*Nobs)).*ones(n,1);
% 
% if(m>1)%non-prime field
%     Py = histc(Y,-1:P-2,2)./Nobs; %pmf estimation for all q symbols
% else
%     Py = histc(Y,0:P-1,2)./Nobs; %pmf estimation for all q symbols
% end
% lgPy = log(Py)./lg_cte;
% Pzero = (Py==0);
% PlgP = Py.*lgPy;
% PlgP(Pzero) = 0;
% v = v - sum(PlgP,2);
% 
% count = n;
% 
% end


% function y = somavetGF(x,z,field)
% y = zeros(1,length(x));
% 
% for it=1:length(x)
%     y(it) = gfadd(x(it),z(it),field);
% end
% 
% y(y==-Inf) = -1;
% 
% end
% 
% function y = prodescalarGF(c,x,field)
% y = zeros(1,length(x));
% for it=1:length(x)    
%     y(it) = gfmul(c,x(it),field);    
% end
% 
% y(y==-Inf) = -1;
%    
% end







