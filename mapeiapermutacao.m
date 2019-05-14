function [Y] = mapeiapermutacao(S,X, permut,q,K)
    % S são as amostras originais
    % X são as observações, após serem misturadas
    % permut é um array (neste caso coluna (NO QICA está sendo linha..)) que indica
    % como se permutaram as posições
    % q é o primo de (q^K)
    % K é a quantidade de fontes.
    
    % The below line creates an array like this:
    % [q^(K-1) q^(K-2) ... q^2 q^1 1]
    % example for q=2 and K = 5
    % 16 8 4 2 1
    base = q.^(K-1:-1:0);



    Y = permut(base*X+1)-1;

    % 17/04/2019
    % pq 48? 48 is the '0', character zero in ASCII

    %THis is the original one
    % Y = double(dec2bin(Y,K)-48)';

    %this is my modification:
    % 25/04/2019
    Y = double(dec2base(Y,q,K)-48)';
    Nobs = size(Y,2);        


% 10/05/2019
% Comentei várias coisas para de fato realizar o que o nome da função diz que ela faz.
% visto que não pretendemos usar essa função para o caso de BSS-lineear, 
% e os algoritmos não-lineares não aparentam conseguir resolver o problema.

    % obs:

    % not and ~ does the same thing in this context.
    % Stemp = S; % ain't need a copy.
    % found = zeros(K,1);
%     for ity=1:K;        
%         for its=1:K;
%             if (~found(its) && sum(Stemp(its,:)==Y(ity,:))==Nobs ...
%                     || sum(Stemp(its,:)==not(Y(ity,:)))==Nobs)                
%                 found(its) = 1;
% %                 Stemp(its,:) = []; %tira da busca pq já encontrou
%                 break;
%             end
%         end        
%     end

    % This part is a pseudocode which will be implemented inside the loop above
    % started in 29/04/2019 at 00:10
    % sums_on_S = [];
    % sums_on_Y = [];
    
    % for symbol=0:(q-1)
    %   sums_on_S = [sums_on_S sum(S==symbol,2)];
    %   sums_on_Y = [sums_on_Y sum(Y==symbol,2)];
    % end
    
    %   sums_on_S = sort(sums_on_S,2);
    %   sums_on_Y = sort(sums_on_Y,2);

    % % we could use all((line1==line2),2) to check if there's a matching line
    % % the 2 means along --->
    % % if it happens, we have found one source.
    % %


    % for i=1:K;
    %   for j=1:K;
    %       if(~found(i))
    %           if( all( (sums_on_S(i,:) == sums_on_Y(j,:) ),2 ) )
    %             found(j) = 1;
    %             break;
    %           end
    %       end
    %   end
    % end



    % WE WILL TEST THE ABOVE CODE LATER (29/04/2019) 00:37
    % 06/05/2019: recuperado da dengue, vamos testar.
    % Parece que ele tá mapeando corretamente as tuplas, mas a maneira de testar
    % se cada fonte foi devidamente separada parece meio...errada.


    % maxhit = sum(found);


    
    % if (maxhit == K)
    %     fprintf('fully separated. Check Y and S for the first 10 elements\n');
    %     Y(:,1:10);
    %     S(:,1:10);
    % else
    %     fprintf('NOT fully separated. Check Y and S for the first 10 elements\n');
    %     Y(:,1:10);
    %     S(:,1:10);
    % end

    % fprintf('Checking for generate_pai_P how he maps every possible tuple\n');
    % all_tuples = generate_pai_P(K,q)';

    % Y_tuples = permut( base * all_tuples +1 ) -1;
    % Y_tuples =  double(dec2base(Y_tuples,q,K)-48)';
    % fprintf('All possible tuples\n');
    % all_tuples
    % fprintf('The mapping found\n');
    % Y_tuples

end

