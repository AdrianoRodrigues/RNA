function new_V = generate_noise(V, p)
    new_V = V;
    % vetor com as posicoes
    positions = 1:length(V);

    % quantidade de mundancas que serao feitas
    num_changes = round(p * length(V));

    % obtem uma posicao aleatoria
    position_change = randsample(positions, num_changes);

    % muda o valor 1 -> 0 ou 0 -> 1
    new_V(position_change) = ~V(position_change);    
end