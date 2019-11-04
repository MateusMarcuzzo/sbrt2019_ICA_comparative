function [pai]=generate_pai_P(n,P)
    counter=zeros(1,n,'single');
    %pai=counter;
    number_of_words=P^n;
    pai=zeros(number_of_words,n,'single');
    for m=1:number_of_words
        pai(m,:)=counter;
        counter=add_to_counter(counter,P-1);
    end
    %pai=pai(1:end-1,:);
    pai=flipud(pai);
end


function [counter]=add_to_counter(counter,P)
    n=size(counter,2);
    flag=1;
    while flag
       t=counter(1,n)+1;
       if t<=P
           counter(1,n)=t;
           flag=0;
       else
           counter(1,n)=0;
           if n-1>0
                n=n-1;
           else
               flag=0;
           end
       end
    end

end
