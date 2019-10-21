%This routine generates a random pmf with cardinality Q
% This routine is used to give a joint_pmf
% to ICA experiments

% Mateus Marcuzzo da Rosa 15/07/2019
function [probs] = generate_random_pmf(Q)


    probs =rand(Q,1);
    sprob=sum(probs);
    probs=probs./sprob(ones(Q,1),:);


       

end