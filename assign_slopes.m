function [M]=assign_slopes(A,probs)
%     tic
%     M=zeros(size(A,1),size(A,2)); 
%     for m1=1:size(A,1)
%         for m2=1:size(A,2)
%             if A(m1,m2)+1<=size(probs,2)
%                 M(m1,m2)=probs(1,A(m1,m2)+1);
%             end
%         end
%     end
%     toc
%     tic
    M=zeros(size(A,1),size(A,2));
    for m1=1:size(probs,2)
        M=M+probs(1,m1)*(A==(m1-1));
    end
%     toc
    
                 
end
