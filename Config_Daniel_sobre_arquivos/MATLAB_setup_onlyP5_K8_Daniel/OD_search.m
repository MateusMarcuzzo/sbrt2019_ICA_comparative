function [lin_min ent_with_appox_vals est_vals opt_v_vec iter terminated ind]=OD_search(P,n,p,slopes,consts,pai,v_vec)


    lin_min=inf;
    ent_with_appox_vals=inf;
    est_vals=NaN;
    opt_v_vec=NaN;
    terminated=0;
    
    
    iter=0;
    flag=1;
    sorted_p=sort(p);
    while flag
        if iter>1
           working=1; 
        end
        %extract coef
        l=find(v_vec, 1, 'first');
        calculated_k=0;
        coef=zeros(P^n,1);
        while calculated_k<n
            a=assign_slopes(pai(:,calculated_k+1:calculated_k+v_vec(l)),slopes(l,:));
            coef=coef+sum(a,2);
            calculated_k=calculated_k+v_vec(l);
            l=l+1;
        end

        sorted_coef=sort(coef,'descend');
        lin_min_iter=sorted_coef'*sorted_p+v_vec*consts;

        %extact est vals
        
        %rank p according to the order of the coef vector 
        vector=coef';
        [sorted_values order]=sort(vector);
        rank=zeros(1,length(vector));
        rank(order) = 1:length(vector);


        ind=max(rank)-rank+1;        
        
        opt_perm=sorted_p(ind);
        
        est_vals=zeros(P,n);
        %est_vals_in_ranges=zeros(P,n);
        for m1=1:P
            for m2=1:n
                est_vals(m1,m2)=sum((pai(:,m2)==(m1-1)).*opt_perm);
                %est_vals_in_ranges(m1,m2)=size(range,1)-sum(est_vals(m1,m2)<range);
            end
        end
        
        %We want to find for each set of parameters at which cell it lies.
        %We use the concavity of the linear approximation to find the minmal objective which determines the minimizing cell 
        sorted_est_vals=sort(est_vals);
        est_vals_tag=est_vals(1:end-1,:);
        [minimum occupied_cells]=min(slopes*est_vals_tag+repmat(consts,1,size(est_vals_tag,2)));
        
        %extract est_v_vec
        %sorted_est_vals_in_ranges=sort(est_vals_in_ranges);
        %sorted_est_vals_in_ranges=sorted_est_vals_in_ranges(1:end-1,:);
        %weights=repmat(P.^(P-2:-1:0)',1,n);
        %occupied_cells=sum((sorted_est_vals_in_ranges-1).*weights,1)+1;
        est_v_vec=zeros(1,size(v_vec,2));
        for m=1:n
              est_v_vec(occupied_cells(1,m))=est_v_vec(occupied_cells(1,m))+1;
        end
        
        if var(est_v_vec-v_vec)>0
            %test if we are not in decent direction
            if lin_min_iter-10^-3>lin_min
                prob=1;
            end
            v_vec=est_v_vec;
            lin_min=lin_min_iter;
            ent_with_appox_vals=sum(sum(-est_vals.*log2(est_vals)));
            opt_v_vec=v_vec;
            iter=iter+1;
        else
            flag=0;
            %test if we are not in decent direction
            if lin_min_iter-10^-3>lin_min
                prob=1;
            end
            lin_min=lin_min_iter;
            ent_with_appox_vals=sum(sum(-est_vals.*log2(est_vals)));
            opt_v_vec=v_vec;
        end
        if iter>10000
            flag=0;
            terminated=1;
            ent_with_appox_vals=inf;
            lin_min=inf;
        end
    end
    
    
end