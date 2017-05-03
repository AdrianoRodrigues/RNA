function [net_h, Yh, net_o] = mlp_test(X, Whi, bias_hi, Woh, bias_oh)
  % entrada da camada escondida com bias
  net_h = Whi * X + bias_hi * ones(1, size(X, 2));  

  % saida da camada escondida 
  Yh = logsig(net_h); 

  % entrada da camada de saida com bias
  net_o = Woh * Yh + bias_oh * ones(1, size(Yh, 2));
end