% Matriz de entrada da RNA
% Cada coluna da matriz representa um caractere hexadecimal vetorizado em 49
% posi��es, ou seja, a matriz de entrada � uma matriz 49x16

X(:,1) = [0 1 1 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 1 0];
%   0  = [0 1 1 1 1 1 0    
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 1 0];

X(:,2) = [0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 1 1 1 1 0];
%    1 =[0 0 0 1 0 0 0    
%        0 0 1 1 0 0 0
%        0 1 0 1 0 0 0
%        0 0 0 1 0 0 0
%        0 0 0 1 0 0 0
%        0 0 0 1 0 0 0
%        0 1 1 1 1 1 0];

X(:,3)=[0 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0];
%     2 =[0 1 1 1 1 1 0    
%         0 0 0 0 0 1 0
%         0 0 0 0 0 1 0
%         0 1 1 1 1 1 0
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 1 1 1 1 0];

X(:,4)=[0 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 1 1 1 1 0];
%     3 =[0 1 1 1 1 1 0    
%         0 0 0 0 0 1 0
%         0 0 0 0 0 1 0
%         0 1 1 1 1 1 0
%         0 0 0 0 0 1 0
%         0 0 0 0 0 1 0
%         0 1 1 1 1 1 0];

X(:,5)=[0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0];
%     4 =[0 1 0 0 0 1 0    
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 1 0
%         0 0 0 0 0 1 0
%         0 0 0 0 0 1 0
%         0 0 0 0 0 1 0];

X(:,6)=[0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 1 1 1 1 0];
%     5 =[0 1 1 1 1 1 0    
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 1 1 1 1 0
%         0 0 0 0 0 1 0
%         0 0 0 0 0 1 0
%         0 1 1 1 1 1 0];

X(:,7)=[0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 1 0];
%     6 =[0 1 1 1 1 1 0    
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 1 1 1 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 1 0];

X(:,8)=[0 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0];
%     7 =[0 1 1 1 1 1 0    
%         0 0 0 0 0 1 0
%         0 0 0 0 0 1 0
%         0 0 0 0 1 0 0
%         0 0 0 1 0 0 0
%         0 0 1 0 0 0 0
%         0 1 0 0 0 0 0];

X(:,9)=[0 1 1 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 1 0];
%     8 =[0 1 1 1 1 1 0    
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 1 0];

X(:,10)=[0 1 1 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 1 1 1 1 0];
%     9 =[0 1 1 1 1 1 0    
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 1 0
%         0 0 0 0 0 1 0
%         0 0 0 0 0 1 0
%         0 1 1 1 1 1 0];

X(:,11)=[0 1 1 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0];
%     A =[0 1 1 1 1 1 0    
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0];

X(:,12)=[0 1 1 1 1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 0 0];
%     B =[0 1 1 1 1 0 0    
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 0 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 0 0];

X(:,13)=[0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0];
%     C =[0 1 1 1 1 1 0    
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 1 1 1 1 0];

X(:,14)=[0 1 1 1 1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1 0 0];
%     D =[0 1 1 1 1 0 0    
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 0 0 0 1 0
%         0 1 1 1 1 0 0];

X(:,15)=[0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0];
%     E =[0 1 1 1 1 1 0    
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 1 1 1 1 0
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 1 1 1 1 0];

X(:,16)=[0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0];
%     F =[0 1 1 1 1 1 0    
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 1 1 1 1 0
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0
%         0 1 0 0 0 0 0];
