function [pai]=generate_pai(n)
    pai=zeros(2^n,n);
    for m=0:n-1
        len=2^(n-m-1);
        vec_instance=[ones(len,1); zeros(len,1)];
        vec=repmat(vec_instance,2^m,1);
        pai(:,m+1)=vec;
    end
                       
end
