function y = tansig(x)
  ## see MATLAB(TM) online help
  y = 2 ./ (1 + exp(-2*x)) - 1;
  ## attention with critical values ==> infinite values
  ## must be set to 1
  i = find(!finite(y));
  y(i) = sign(x(i));
end