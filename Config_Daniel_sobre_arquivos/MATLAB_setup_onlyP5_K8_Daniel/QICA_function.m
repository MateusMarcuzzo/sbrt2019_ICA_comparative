function [opt_est_vals,opt_appox_lin_min2,opt_appox_ent_with_appox_vals2,overall_opt_perm,opt_v_vec]=QICA_function(n,P,p,exhaustive,min_k,max_k,I)


sorted_p=sort(p);

[pai]=generate_pai_P(n,P); 
tot_ent=-sum(p(p>0).*log2(p(p>0)));

ent_vec=tot_ent*ones(max_k-min_k+1,1);

lin_min_vec=zeros(max_k-min_k+1,1);
appox_lin_min_vec=zeros(max_k-min_k+1,1);
ent_with_lin_params=zeros(max_k-min_k+1,1);
appox_ent_with_lin_params=zeros(max_k-min_k+1,1);
est_var=zeros(max_k-min_k+1,1);

appox_lin_min_vec2=zeros(max_k-min_k+1,1);
appox_ent_with_lin_params2=zeros(max_k-min_k+1,1);

num_of_steps_to_find_solution=zeros(max_k-min_k+1,1);
num_of_initializations_to_find_solution=zeros(max_k-min_k+1,1);

for k=min_k:max_k
    [slopes, consts]=calc_k_params_P(k,P);

    if exhaustive

        %exhustive algorithm:

        % I'm using the non_mex form
        % 25/04/2019
        % I'm wondering if the k^(P-1) here is correct for just k. Like BICA_function
        %
        v = allVL1nonrecurs(k^(P-1), n); %we have n entropies to calculate, each one of them may fall in each of the k^(p-1) cells we defined. 
        v=fliplr(v);

        lin_min=inf;
        opt_v_vec=NaN;

        for iter=1:size(v,1)
            broken=0;
            coef=zeros(P^n,1);
            calculated_k=0;
            v_vec=v(iter,:);
            l=find(v_vec, 1, 'first');
            while calculated_k<n
                if (max(slopes(l,:))==inf)
                    broken=1;
                    break;
                end
                a=assign_slopes(pai(:,calculated_k+1:calculated_k+v_vec(l)),slopes(l,:));
                coef=coef+sum(a,2);
                calculated_k=calculated_k+v_vec(l);
                l=l+1;
            end
            if ~broken

                [sorted_coef ind]=sort(coef,'descend');
                lin_min_iter=sorted_coef'*sorted_p+v_vec*consts;


                 if (lin_min_iter<lin_min)
                    lin_min=lin_min_iter;
                    if lin_min_iter<tot_ent
                        problem=1;
                    end
                    opt_v_vec=v_vec;
                 end
            end
         end


        %calc est_vals
        coef=zeros(P^n,1);
        calculated_k=0;
        for l=1:size(v,2)
            a=assign_slopes(pai(:,calculated_k+1:calculated_k+opt_v_vec(l)),slopes(l,:));
            coef=coef+sum(a,2);
            calculated_k=calculated_k+opt_v_vec(l);
        end


        %rank the vector p according to the order of the coef vector 
        vector=coef';
        [sorted_values order]=sort(vector);
        rank=zeros(1,length(vector));
        rank(order) = 1:length(vector);


        ind=max(rank)-rank+1;
        opt_perm=sorted_p(ind);
        est_vals=zeros(P,n);
        for m1=1:P
            for m2=1:n
                est_vals(m1,m2)=sum((pai(:,m2)==(m1-1)).*opt_perm);
            end
        end
        opt_est_vals=est_vals;
        lin_min_vec(k-min_k+1)=lin_min;
        ent_with_est_vals=sum(sum(-est_vals.*log2(est_vals)));
        ent_with_lin_params(k-min_k+1)=ent_with_est_vals;
        opt_appox_ent_with_appox_vals2=ent_with_est_vals;
        overall_opt_perm=ind;
        opt_appox_lin_min2=lin_min;
    end
    
    
    %%%%% appox. algorithm
    if ~exhaustive
        finish=0;
        counter=1;
        %opt_appox_lin_min - holds the value of the upper bound piecewise objective
        %opt_appox_ent_with_appox_vals - holds the value of the true objective, when applying to it the params we found
        
        %For debugging purposes, we have two "minimizations" we perform - in the first we keep track 
        % of both variables, according to the lowset value of
        % opt_appox_ent_with_appox_vals we have seen so far:
        opt_appox_lin_min=inf;    
        opt_appox_ent_with_appox_vals=inf;

        %In the second we keep track of both variables, according to the 
        % lowset value of opt_appox_lin_min we have seen so far:
        opt_appox_lin_min2=inf;
        opt_appox_ent_with_appox_vals2=inf;

        opt_iter=inf;
        %we run the appox alg. enough times and keep the optimal vals we find
        while ~finish
            s = sort(randperm(size(consts,1)+n-1,size(consts,1)-1));
            v_vec = diff([0 s size(consts,1)+n]) - 1;

            [appox_lin_min appox_ent_with_appox_vals appox_est_vals appox_v_vec iter terminated opt_perm]=OD_search(P,n,p,slopes,consts,pai,v_vec);

            if terminated==1
                test_it=1;
            end
            if appox_ent_with_appox_vals<opt_appox_ent_with_appox_vals
                opt_appox_lin_min=appox_lin_min;
                opt_appox_ent_with_appox_vals=appox_ent_with_appox_vals;
                opt_iter=iter;
            end
            if appox_lin_min<opt_appox_lin_min2
                opt_appox_lin_min2=appox_lin_min;
                opt_appox_ent_with_appox_vals2=appox_ent_with_appox_vals;
                opt_v_vec=appox_v_vec;
                opt_iter=iter;
                opt_est_vals=appox_est_vals;
                overall_opt_perm=opt_perm;
            end

            if counter==I-1
                finish=1;
            end
            counter=counter+1;
        end
    

        appox_lin_min_vec(k-min_k+1)=opt_appox_lin_min;
        appox_ent_with_lin_params(k-min_k+1)=opt_appox_ent_with_appox_vals;

        appox_lin_min_vec2(k-min_k+1)=opt_appox_lin_min2;
        appox_ent_with_lin_params2(k-min_k+1)=opt_appox_ent_with_appox_vals2;

        num_of_steps_to_find_solution(k-min_k+1)=opt_iter;
        num_of_initializations_to_find_solution(k-min_k+1)=counter;
    
    end
    
    
  
end


end
