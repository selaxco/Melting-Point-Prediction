from collections import Counter
import pandas as pd
import requests

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler



def get_df(file):
    df = pd.read_excel(file)
    return df



def get_filtered_df(file, length, filters=['PROTEIN', 'LENGTH', 'UNIPROT_ID', 'Tm_(C)', 'ADDITIVES']):
    df_unfiltered = get_df(file)
    df = df_unfiltered.loc[df_unfiltered['LENGTH'] <= length][filters]
    df = df.reset_index()
    return df



def get_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.ok:
        lines = response.text.split("\n")
        sequence = "".join(lines[1:]) # Ignorar a primeira linha (cabeçalho) e concatenar as linhas restantes
        return sequence
    else:
        print(f"Erro ao obter a sequência para o ID: {uniprot_id}")
        return None



def sequence_to_amino(sequence):
    z = Counter(sequence)
    
    amino = 'ACDEFGHIKLMNPQRSTVWY'
    dic = {i: z[i] for i in amino}
    
    return dic



def get_amino(uniprot_id):
    sequence = get_protein_sequence(uniprot_id)
    amino = sequence_to_amino(sequence)
    return amino



def get_additive_input_type_1(additive):
    if not isinstance(additive, str):
        return 0
    else:
        separated = additive.split(', ')
        a = 0
        if '30 μM Pervanadate' in separated: a += 1
        if '0.67% NP40 (octylphenoxypoly(ethyleneoxy)ethanol)' in separated: a += 2
        return a/3
    


def get_additive_input_type_2(additive):
    if not isinstance(additive, str):
        return [0, 0]
    else:
        separated = additive.split(', ')
        a = [0, 0]
        if '30 μM Pervanadate' in separated: a[0] = 1
        if '0.67% NP40 (octylphenoxypoly(ethyleneoxy)ethanol)' in separated: a[1] = 1
        return a



def cria_mlp(funcao_ativacao, x_treino, x_teste, y_treino, y_teste, normalizador_y, NEURONIOS_C1=30, NEURONIOS_C2=20, NEURONIOS_C3=10, TAXA_DE_APRENDIZADO=0.001, NUM_EPOCAS=20000):
    class MLP(nn.Module): # Leaky ReLU
        def __init__(
            self, num_dados_entrada, neuronios_c1, neuronios_c2, neuronios_c3, num_targets
        ):
            # Temos que inicializar a classe mãe
            super().__init__()

            # Definindo as camadas da rede
            self.camadas = nn.Sequential(
                        nn.Linear(num_dados_entrada, neuronios_c1),
                        funcao_ativacao(),
                        nn.Linear(neuronios_c1, neuronios_c2),
                        funcao_ativacao(),
                        nn.Linear(neuronios_c2, neuronios_c3),
                        funcao_ativacao(),
                        nn.Linear(neuronios_c3, num_targets),
                    )

        def forward(self, x):
            """Esse é o método que executa a rede do pytorch."""
            x = self.camadas(x)
            return x

    NUM_DADOS_DE_ENTRADA = x_treino.shape[1]
    NUM_DADOS_DE_SAIDA = y_treino.shape[1]

    mlp = MLP(NUM_DADOS_DE_ENTRADA, NEURONIOS_C1, NEURONIOS_C2, NEURONIOS_C3, NUM_DADOS_DE_SAIDA)

    # função perda será o erro quadrático médio
    fn_perda = nn.MSELoss()

    # otimizador será o Adam, um tipo de descida do gradiente
    otimizador = optim.Adam(mlp.parameters(), lr=TAXA_DE_APRENDIZADO)

    mlp.train()

    desc = 'Progresso: '
    loop = trange(NUM_EPOCAS, ncols=150, desc=desc)
    for epoca in loop:
        # forward pass
        y_pred = mlp(x_treino)

        # zero grad
        otimizador.zero_grad()

        # loss
        loss = fn_perda(y_pred, y_treino)
        loop.set_postfix_str(f'Perda: {loss.data:<20.10f} / Neurons: {NEURONIOS_C1, NEURONIOS_C2, NEURONIOS_C3} / LR: {TAXA_DE_APRENDIZADO}')
        loop.refresh()
        
        # backpropagation
        loss.backward()

        # atualiza parâmetros
        otimizador.step()
    
    mlp.eval()
    
    with torch.no_grad():
        y_true = normalizador_y.inverse_transform(y_teste)
        y_pred = mlp(x_teste)
        y_pred = normalizador_y.inverse_transform(y_pred)

    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    print(f'Loss do teste: {RMSE}')

    return mlp, RMSE


