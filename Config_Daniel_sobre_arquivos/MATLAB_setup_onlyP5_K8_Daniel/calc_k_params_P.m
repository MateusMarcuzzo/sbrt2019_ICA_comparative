function [slopes, consts]=calc_k_params_P(k,P)

    range_length=0.5/k;
    range=[0:range_length:0.5]';
    mid_range_points=[range_length/2:range_length:0.5-range_length/2]';
    mid_range_points_in_all_dims= repmat(mid_range_points,1,P-1);
    %number_of_points=size(mid_range_points,1);
    slopes=inf*ones(k^(P-1),P-1);   %in total we have k^(P-1) cells, each one contains P-1 slope parameters (gradients)
    consts=zeros(k^(P-1),1);   %in total we have k^(P-1) cells, each contains a single value.
    
    counter=ones(1,P-1);  %this counter indicates the desired point at each axis 
    for m1=1:size(slopes,1)
       p= mid_range_points_in_all_dims(counter); %p holds the mid range points at the current counter. These are actually the probabilities according to which we calculate the entropy. 
       if sum(p)<1
            all_p=[p 1-sum(p)];
            ent_at_p=sum(-all_p.*log2(all_p));
            for m2=1:P-1
                mid_range_point=p(m2);
                slopes(m1,m2)=log2(all_p(end))-log2(mid_range_point);
                
       
            end
            consts(m1)=ent_at_p-p*slopes(m1,:)';
       end
       counter=add_to_counter(counter,k,P);
    end
    %calc correct range (where the gradient intersect)
%     slopes_of_single_cell=slopes(1:size(range,1)-1,end);
%     consts_of_single_cell=consts(1:size(range,1)-1,end);
%     slopes_difference=slopes_of_single_cell(1:end-1)-slopes_of_single_cell(2:end);
%     consts_difference=consts_of_single_cell(2:end)-consts_of_single_cell(1:end-1);
%     correct_range=consts_difference./slopes_difference;
%     range=[0; correct_range; 0.5;];
   
end

function [counter]=add_to_counter(counter,k,P)
    n=size(counter,2);
    flag=1;
    while flag
       t=counter(1,n)+1;
       if t<=k
           counter(1,n)=t;
           flag=0;
       else
           counter(1,n)=1;
           if n-1>0
                n=n-1;
           else
               flag=0;
           end
       end
    end
               
end
        
        

