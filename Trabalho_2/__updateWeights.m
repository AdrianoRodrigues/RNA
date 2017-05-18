function outNet = __updateWeights(net, weights)
  
  outNet = net;
  outNet.Whi = weights.Whi;
  outNet.bias_hi = weights.bias_hi;
  outNet.Woh = weights.Woh;
  outNet.bias_oh = weights.bias_oh;
  
end