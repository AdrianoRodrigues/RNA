function [net] = MLP(numInputs, numHiddens, numOutputs, pGoal, pEpochs, pEta)
  
  % Parametros de treinamento da rede
  net.trainParam.numInputs = numInputs;
  net.trainParam.numHiddens = numHiddens;
  net.trainParam.numOutputs = numOutputs;
  net.trainParam.goal = pGoal;
  net.trainParam.epochs = pEpochs;
  net.trainParam.eta = pEta;
  net.trainParam.show = 50;
  
  % inicializacao dos pesos entre camada de entrada e camada escondida
  net.Whi = rand(numHiddens, numInputs) - 0.5; 
  net.bias_hi = rand(numHiddens, 1) - 0.5;

  % inicializacao dos pesos entre camada escondida e camada de saida
  net.Woh = rand(numOutputs, numHiddens) - 0.5; 
  net.bias_oh = rand(numOutputs, 1) - 0.5;
  
end