function [Whi, Bhi, Woh, Boh] = mlp_trainning(input_data, num_hidden, num_out, max_epochs, eta, target)

    mean_error = [];

    eta_inc = 10;
    eta_dec = .1;
    eta_max = 1e-10;
    err_max = 2e-10;
    
    k = 1;

    % inicializando os pesos entre a camanda escondida e a camada de entrada
    Whi = rand(num_hidden, size(input_data, 1)) - 0.5;
    Bhi = rand(num_hidden, 1) - 0.5;

    % inicializando os pesos entre a camada de saida e a camanda escondida
    Woh = rand(num_out, num_hidden) - 0.5;
    Boh = rand(num_out, 1) - 0.5;
    
    for epoch = 1:max_epochs
        % Para cada entrada de treinamento
        %for i = 1:size(input_data, 1)
            % Calcula entrada da camada escondida
            net_h = Whi * input_data + Bhi;
            
            % Calcula saida da camada escondida / entrada da camada de saida
            Yh = logsig(net_h);

            % Calcula a saida da rede
            net_o = Woh * Yh + Boh;
            
            Y = k * net_o;                   

            % Calcula o erro
            E = target - Y;
            
            df = k*ones(size(net_o));

            % Calcula os deltas para ajutes nos pesos entre camada escondida e camada de saida
            delta_bias_oh = eta * sum(E')';
            delta_Woh = eta * (E.*df) * Yh';

            % Ajusta os pesos entre a camada escondida e a camada de saida
            Woh = Woh + delta_Woh;
            Boh = Boh + delta_bias_oh;

            % Calcula o erro retropropagado
            Eh = Woh' * (E.*df);

            % Derivada da funcao sigmoid
            df = logsig(net_h) - (logsig(net_h).^2);
            
            % Calcula os deltas para ajutes nos pesos entre camda de entrada e camada escondida
            delta_bias_hi = -eta * sum(Eh .* df);
            delta_Whi = eta * (Eh .* df) * input_data';

            % Ajusta os pesos entre a camada de entrada e a camada escondida
            Whi = Whi + delta_Whi;
            Bhi = Bhi + delta_bias_hi;

            % Calcula e salva o erro medio
            mean_error = [ mean_error sum((E.^2)) / size(target, 2) ];

        %end
        
        if rem(epoch, 50) == 0
            fprintf('Epoch: %d; Mean Error: %.10f\n', epoch, mean_error(size(mean_error, 2)));
            if exist('OCTAVE_VERSION') fflush(stdout); end;
        end
        
    end    
    
end