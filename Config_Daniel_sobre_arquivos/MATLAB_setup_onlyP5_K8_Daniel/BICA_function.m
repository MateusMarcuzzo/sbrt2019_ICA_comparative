function [permuted_p,opt_perm,est_vals,min_ent]=BICA_function(p,min_k,max_k)

min_ent=inf;
permuted_p=inf;
opt_perm=inf;
est_vals=inf;
n=log2(length(p));


[sorted_p sorted_p_ind]=sort(p);
%sorted_p=p(sorted_p_ind);
[pai]=generate_pai(n); 


ent_with_lin_params=zeros(max_k-min_k+1,1);
lin_min_vec=zeros(max_k-min_k+1,1);


for k=min_k:max_k
    k;
    [slopes, consts,range]=calc_k_params(k);

    lin_min=inf;
    opt_v_vec=NaN;

    %v = allVL1(k, n);
    v = allVL1nonrecurs(k,n); %returns all combinations of n balls in k boxes
    v=fliplr(v);

    for iter=1:size(v,1)
        coef=zeros(2^n,1);
        calculated_k=0;
        v_vec=v(iter,:);
        for l=1:size(v,2)
            a=slopes(l)*pai(:,calculated_k+1:calculated_k+v_vec(l));
            coef=coef+sum(a,2);
            calculated_k=calculated_k+v_vec(l);
        end
        sorted_coef=flipud(sort(coef));
        lin_min_iter=sorted_coef'*sorted_p+v_vec*consts;
        if lin_min_iter<lin_min
            lin_min=lin_min_iter;
            opt_v_vec=v_vec;
        end
    end


    %calc est_vals
    coef=zeros(2^n,1);
    calculated_k=0;
    for l=1:size(v,2)
        a=slopes(l)*pai(:,calculated_k+1:calculated_k+opt_v_vec(l));
        coef=coef+sum(a,2);
        calculated_k=calculated_k+opt_v_vec(l);
    end

    
    %rank p according to the order of the coef vector 
    vector=coef';
    [sorted_values order]=sort(vector);
    rank=zeros(1,length(vector));
    rank(order) = 1:length(vector);
    
    
    ind=max(rank)-rank+1;
    opt_p=sorted_p(ind);
    est_vals=sum(pai.*repmat(opt_p,1,n));
    lin_min_vec(k-min_k+1)=lin_min;
    ent_with_lin_params(k-min_k+1)=sum(-est_vals.*log2(est_vals)-(1-est_vals).*log2(1-est_vals));
    if k==min_k
        opt_perm= sorted_p_ind(ind');
        %sorted_p=p(sorted_p_ind);
        permuted_p=opt_p;
        min_ent=ent_with_lin_params(k-min_k+1);
        
    else
        if  ent_with_lin_params(k-min_k+1)<min_ent
            opt_perm= sorted_p_ind(ind');
            permuted_p=opt_p;
            min_ent=ent_with_lin_params(k-min_k+1);  
        end
    end
end



end


