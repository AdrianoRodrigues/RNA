function samples = generate_samples(samples, num_extra)
    for i = 1:size(samples, 2)
        for j = 1:num_extra
            % pega um dos exemplos de treinamento
            sample = samples(:, i)';
            
            % obtem aleatoriamente uma taxa de ruido entre 5 e 10%.
            noise = 0.05 + (rand(1) * 0.05);

            % aplica o ruido no exemplo de treinamento
            new_sample = generate_noise(sample, noise);

            % adiciona o novo exemplo 
            samples = [ samples new_sample' ];
        end
    end
end