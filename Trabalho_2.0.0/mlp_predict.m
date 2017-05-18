function output = mlp_predict(input_data, Whi, Bhi, Woh, Boh)
    output = zeros(size(input_data, 1), 16);
    % Para cada entrada de treinamento
    for i = 1:size(input_data, 1)
        % Calcula entrada da camada escondida
        net_h = Whi * input_data(i, :)' + Bhi;
        
        % Calcula saida da camada escondida / entrada da camada de saida
        Yh = logsig(net_h);

        % Calcula a saida da rede
        net_o = Woh * Yh + Boh;
        
        output(i,:) = logsig(net_o)';
    end
end