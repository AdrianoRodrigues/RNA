function [perf, Er, net_h, Yh] = MLP_Perf(net, input, output)
  
  % get parameters from net
  Whi     = net.Whi;
  bias_hi = net.bias_hi;
  Woh     = net.Woh;
  bias_oh = net.bias_oh;
  H       = net.trainParam.numHiddens;
  
    % entrada da camada escondida com bias
    net_h = Whi * input + bias_hi * ones(1, size(input, 2));

    % saida da camada escondida 
    Yh = logsig(net_h);

    % entrada da camada de saida com bias
    net_o = Woh * Yh + bias_oh * ones(1, size(Yh, 2));
    
    net_o = round(net_o);
   
    % calcula o erro na saida da rede
    Er = (output - net_o);

  % calcula o erro quadratico medio
  perf = sum(sum(Er.^2)) / H;  
  
end