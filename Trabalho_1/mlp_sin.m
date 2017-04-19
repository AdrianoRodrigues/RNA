clc

more off

% MATLAB Neural Network Multi-Layer Perceptron Backpropagation
% Predicting function y = sin(x)
% author Adriano Silva
% date 2017/04/12

I = 1;          % neuronios na camada de entrada
H = 5;         % neuronios na camada escondida
O = 1;          % neuronios na camada de saida
ETA = 0.001;    % taxa de aprendizado
E_MAX = 0.0025; % erro maximo
EPOCHS = 250;   % quantidade de epocas para o treinamento da rede
TRAIN = 0.7;    % percentual dos dados usados para treinar a rede

% dados de entrada
% X = 0 : 2 * pi / 99 : 2 * pi;

%X = linspace(0, 2*pi, 100);
%X_train = X(1:floor(TRAIN.*end));
%X_test = X(floor(TRAIN.*end)+1:end);
X_train = sort(rand(1, 1000)*2*pi); %linspace(0, 2*pi, 1000);
X_test = sort(rand(1, 300)*2*pi); %linspace(0, 2*pi, 300);

% resultado esperado
%D = sin(X);
%D_train = D(1:floor(TRAIN.*end));
%D_test = D(floor(TRAIN.*end)+1:end);
D_train = sin(X_train);
D_test = sin(X_test);

% matriz que ira armazenar o erro quadratico medio
Eav = [];

vars = load ('data.mat'); 
if exist('INIT_WEIGHTS', 'var') == 1 && INIT_WEIGHTS == 1
  printf('Inicializando pesos...\n');
  % inicializacao dos pesos entre camada de entrada e camada escondida
  Whi = rand(H, I) - 0.5; 
  bias_hi = rand(H, 1) - 0.5;

  % inicializacao dos pesos entre camada escondida e camada de saida
  Woh = rand(O, H) - 0.5; 
  bias_oh = rand(O, 1) - 0.5;  
else  
  printf('Carregando pesos...\n');
  Whi = vars.Whi;
  Woh = vars.Woh;
  bias_hi = vars.bias_hi; 
  bias_oh = vars.bias_oh;
end

for epoch = 1 : EPOCHS
  % entrada da camada escondida com bias
  net_h = Whi * X_train + bias_hi * ones(1, size(X_train, 2)); 
  
  % entrada da camada escondida
  % net_h = Whi * X; 

  % saida da camada escondida 
  Yh = logsig(net_h); 

  % entrada da camada de saida com bias
  net_o = Woh * Yh + bias_oh * ones(1, size(Yh, 2));
  
  % entrada da camada de saida 
  % net_o = Woh * Yh; 
  
  % calcula o erro na saida da rede
  E = (D_train - net_o);
  
  % E = (saida - desejado) * 1/desejado; % calcula erro e considerar E < 0.01
  
  % calcula variacao dos pesos entre camada de saida e camada escondida
  delta_bias_oh = ETA * sum(E')';
  delta_Woh = ETA * E * Yh';
  
  df = logsig(net_h) - (logsig(net_h).^2);
  
  % calcula erro retropropagado para camada de entrada
  Eh = -(Woh' * E);

  % calcula variacao dos pesos entre camada escondida e camada de entrada
  
  delta_bias_hi = -ETA * sum((Eh .* df)')';
  delta_Whi = -ETA * (Eh .* df) * X_train';
  
  % calcula novos pesos
  Woh = Woh + delta_Woh;
  bias_oh = bias_oh + delta_bias_oh;
  
  Whi = Whi + delta_Whi;
  bias_hi = bias_hi + delta_bias_hi;
  
  % calcula erro quadratico medio
  %Eav(epoch) = sum(sum(E.^2)) / size(X_train, 2) / H;
  
  Eav = [Eav (sum(E.^2) / H)];
  
  figure(1);
  plot(Eav)
  refresh(); 
    
  if rem(epoch, 50) == 0
    fprintf('epoch: %d, erro: %f\n', epoch, Eav(epoch));
  end
  
  % parar quando erro < 0.005
  if Eav(epoch) < E_MAX
    fprintf('converged at epoch: %d\n', epoch);
    fprintf('epoch: %d, erro: %f\n', epoch, Eav(epoch));
    break 
  end
 
end

% ir� salvar novos pesos somente se houve altera��o na quantidade de epocas
% ou se o �ltimo erro for menor do que o �ltimo erro anterior ao inicio do 
% treinamento.
if size(vars.Eav, 2) != EPOCHS || Eav(EPOCHS) < vars.Eav(EPOCHS) || INIT_WEIGHTS == 1
  save data.mat Whi Woh bias_hi bias_oh Eav
end

INIT_WEIGHTS = 0;

figure(2)
plot(X_train, net_o, 'color', 'red', X_train, D_train, 'color', 'green');
title ("Trainning sin(x)");

%% Testando
% entrada da camada escondida com bias
net_h = Whi * X_test + bias_hi * ones(1, size(X_test, 2)); 

% saida da camada escondida 
Yh = logsig(net_h); 

% entrada da camada de saida com bias
net_o = Woh * Yh + bias_oh * ones(1, size(Yh, 2)); 

figure(3)
plot(X_test, net_o, 'color', 'red', X_test, D_test, 'color', 'green');
title ("Test sin(x)");
