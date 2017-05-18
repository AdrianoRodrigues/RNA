hidden_neurons = 112;
output_neurons = 16;
learning_rate = 5e-4;
max_epochs = 30000;

% quantidade de novos exemplos com ruido que serao adicionados
samples_extra = 5;

% carrega os exemplos de treinamento
input_data = load_data();

% adiciona exemplos com ruido, para evitar overfitting
new_input_data = generate_samples(input_data, samples_extra);

% normaliza os dados de entrada e calcula a saida desejada
[ normalized_input, target ] = normalize_input(new_input_data, samples_extra);

% treinamento da rede
[ Whi, Bhi, Woh, Boh ] = mlp_trainning(normalized_input, hidden_neurons, output_neurons, max_epochs, learning_rate, target);

input_data_noise5 = input_data;
input_data_noise10 = input_data;
input_data_noise20 = input_data;
input_data_noise30 = input_data;

for i = 1:size(input_data, 2) 
    input_data_noise5(:,i) = generate_noise(input_data(:, i)', .05)';
    input_data_noise10(:,i) = generate_noise(input_data(:, i)', .10)';
    input_data_noise20(:,i) = generate_noise(input_data(:, i)', .20)';
    input_data_noise30(:,i) = generate_noise(input_data(:, i)', .30)';
end

% verificacao da rede
[ xx, yy ] = normalize_input(input_data, 0);

output = mlp_predict(xx, Whi, Bhi, Woh, Boh)

errr

% verificacao da rede
[ xx5, yy ] = normalize_input(input_data_noise5, 0);

output5 = mlp_predict(xx5, Whi, Bhi, Woh, Boh)

% verificacao da rede
[ xx10, yy ] = normalize_input(input_data_noise10, 0);

output10 = mlp_predict(xx10, Whi, Bhi, Woh, Boh)

% verificacao da rede
[ xx20, yy ] = normalize_input(input_data_noise20, 0);

output20 = mlp_predict(xx20, Whi, Bhi, Woh, Boh)

% verificacao da rede
[ xx30, yy ] = normalize_input(input_data_noise30, 0);

output30 = mlp_predict(xx30, Whi, Bhi, Woh, Boh)