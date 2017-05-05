function [net] = MLP_Train(net, pInput, pOutput)

  # get trainning parameter for net
  epochs = net.trainParam.epochs
  goal = net.trainParam.goal
  show = net.trainParam.show

  # motivo do termino do treinamento
  stop = ""
    
  for epoch = 0 : epochs
  
        
    stop = stopifnecessary(stop, epoch, epochs, perf, goal);
    
    showtrainprogress(epoch, epochs, show, perf, goal, stop);
    
    if length(stop)
      break;
    endif
    
  end

end

function [stop] = stopifnecessary(stop, epoch, epochs, perf, goal)
  
  ## check number of inputs
  error(nargchk(5, 5, nargin));

  if (perf <= goal)
    stop = "Performance goal met.";
  elseif (epoch == epochs)
    stop = "Maximum epoch reached, performance goal was not met.";
  endif  
    
end

function showtrainprogress(epoch, epochs, show, perf, goal, stop)

  ## check number of inputs
  error(nargchk(6, 6, nargin));

  ## show progress
  if isfinite(show) & (!rem(epoch, show) | length(stop))
    if isfinite(epochs)
      fprintf("Epoch %g/%g", epoch, epochs);
    endif
    if isfinite(goal)
      fprintf(", Error %g/%g", perf, goal); # outputs the performance function
    endif
    fprintf("\n")
    if length(stop)
      fprintf("%s\n\n", stop);
    endif
    fflush(stdout); # writes output to stdout as soon as output messages are available
  endif
end