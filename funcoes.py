'''
    Código desenvolvido para a matéria de Redes Neurais e Algoritmos Genéticos
    da Ilum - Escola de Ciência ministrada pelo professor doutor Daniel Roberto
    Cassar

    Autores do código:
    Gustavo Duarte Verçosa
    João Pedro Aroucha de Brito
    Thaynara Beatriz Selasco de Matos
    Vitória Yumi Uetuki Nicoleti
'''



################################################################################
#                                Importações                                   #
################################################################################



# Importação do objeto de criação de dicionários Counter
from collections import Counter

# Importação da biblioteca de DataFrames Pandas
import pandas as pd

# Importação da biblioteca de requisições de APIs
import requests


# Pytorch
# Biblioteca completa do pytorch
import torch

# Submódulo de Redes Neurais
import torch.nn as nn

# Submódulo de otimizadores
import torch.optim as optim


# IMódulo de progressão de iterações do python
from tqdm import trange

# Sklearn
# Submódulo de métricas de avaliação do sklear
from sklearn.metrics import mean_squared_error



################################################################################
#                                 Funções                                      #
################################################################################



def get_df(file):
    '''Cria um pandas.DataFrame a partir de um arquivo de excel.

    Args:
        file: string, o caminho relativo ou absoluto para o arquivo de excel.

    Returns:
        pandas.DataFrame: DataFrame com os dados do arquivo passado.
    '''
    df = pd.read_excel(file)
    return df



def get_filtered_df(
        file, length, filters=['PROTEIN', 'LENGTH', 'UNIPROT_ID', 'Tm_(C)',
        'ADDITIVES']
    ):
    '''Cria um pandas.DataFrame a partir do arquivo especificado aplicando um
    filtro de tamanho nas proteínas (quantidade de aminoácidos) e retorna
    apenas as colunas especificadas.

    Args:
        file: string, caminho relativo ou absoluto para o arquivo de excel;

        lenght: int, número de aminoácidos limite, número máximo de aminoácidos
        na cadeia da proteína;

        filters (default: ['PROTEIN', 'LENGTH', 'UNIPROT_ID', 'Tm_(C)',
        'ADDITIVES']): list, nome das colunas a serem adicionadas no DataFrame.
    
    Returns:
        pandas.DataFrame: DataFrame do arquivo de excel passado com os filtros.
    '''
    df_unfiltered = get_df(file)
    df = df_unfiltered.loc[df_unfiltered['LENGTH'] <= length][filters]
    df = df.reset_index()
    return df



def get_protein_sequence(uniprot_id):
    '''Pega a sequência de aminoácidos de uma proteina a partir de seu ID no
    banco de dados aberto UniProt.

    Args:
        uniprot_id: string, id da proteína no banco de dados.

    Returns:
        string: Sequência de aminoácidos da proteína.
    '''
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.ok:
        lines = response.text.split("\n")
        # Ignorar a primeira linha (cabeçalho) e concatenar as linhas restantes
        sequence = "".join(lines[1:])
        return sequence
    else:
        print(f"Erro ao obter a sequência para o ID: {uniprot_id}")
        return None



def sequence_to_amino(sequence):
    '''Pega a sequência de aminoácidos de uma proteína e retorna a quantidade de
    cada um em um dicionário ordenado por ordem alfabética.

    Args:
        sequence: string, sequência dos aminoácidos da proteína.
    
    Returns:
        dict: Com todos com a quantidade de cada aminoácido, incluindo não
        presentes.
    '''
    z = Counter(sequence)
    
    amino = 'ACDEFGHIKLMNPQRSTVWY'
    dic = {i: z[i] for i in amino}
    
    return dic



def get_amino(uniprot_id):
    '''Retorna diretamente a quantidade de cada aminoácido da proteína a partir
    do ID desta no UniProt.

    Args:
        uniprot_id: string, id da proteína no banco de dados.

    Returns:
        dict: Com todos com a quantidade de cada aminoácido, incluindo não
        presentes.
    '''
    sequence = get_protein_sequence(uniprot_id)
    amino = sequence_to_amino(sequence)
    return amino



def get_additive_input_type_1(additive):
    '''Retorna o dado de input de aditivos de tipo 1 para a rede, a partir do
    valor passado.

    Note:
        Para mais informações sobre o tipo de input dos aditivos e oque cada
        um representa, além dos possíveis valores aceitos na rede, acesso o
        README.md presente no github:
        https://github.com/selaxco/Melting-Point-Prediction

    Args:
        additive: string, representando os aditivos encontrados em conjunto com
        a proteína de interesse.
    
    Returns:
        float: Valor a ser utilizado na rede para contabilizar a presença ou não
        dos aditivos.
    '''
    if not isinstance(additive, str):
        return 0
    
    separated = additive.split(', ')
    a = 0
    if '30 μM Pervanadate' in separated: a += 1
    if '0.67% NP40 (octylphenoxypoly(ethyleneoxy)ethanol)' in separated: a += 2
    return a/3
    


def get_additive_input_type_2(additive):
    '''Retorna o dado de input de aditivos de tipo 2 para a rede, a partir do
    valor passado.

    Note:
        Para mais informações sobre o tipo de input dos aditivos e oque cada
        um representa, além dos possíveis valores aceitos na rede, acesso o
        README.md presente no github:
        https://github.com/selaxco/Melting-Point-Prediction

    Args:
        additive: string, representando os aditivos encontrados em conjunto com
        a proteína de interesse.
    
    Returns:
        list: Valores a serem utilizados na rede para contabilizar a presença ou
        não dos aditivos.
    '''
    if not isinstance(additive, str):
        return [0, 0]
    
    separated = additive.split(', ')
    a = [0, 0]
    if '30 μM Pervanadate' in separated: a[0] = 1
    if '0.67% NP40 (octylphenoxypoly(ethyleneoxy)ethanol)' in separated:
        a[1] = 1
    return a



class MLP(nn.Module):
    '''Classe que será utilizada como rede neural para o problema, podendo
    ser criada de maneira personalizável e podendo até mesma ser utilizada
    de outras formas ou aplicações.

    Attributes:
        camadas: torch.nn.Sequential, estrutura das camadas sequenciais da
        rede criada.
    
    Methods:
        __init__: inicializa a rede neural e cria suas camadas estruturadas;
        forward: executa a rede a partir do pytorch.
    '''
    
    def __init__(
        self, funcao_ativacao, num_features, neuronios_c1, neuronios_c2,
        neuronios_c3, num_targets
    ):
        '''Inicializa a classe e cria as camadas da rede.

        Args:
            self: MLP, instância própria da classe;

            funcao_ativacao: torch.nn.Module, a função de ativação a ser
            utilizada na rede neural como um todo;

            num_features: int, número de dados de entrada da rede;
            
            neuronios_c1: int, número de neurônios da camada 1 (hidden
            layer 1);
            
            neuronios_c2: int, número de neurônios da camada 2 (hidden
            layer 2);
            
            neuronios_c3: int, número de neurônios da camada 3 (hidden
            layer 3);
            
            num_targets: int, número de dados de saida da rede;
        '''

        super().__init__()

        # Definindo as camadas da rede
        self.camadas = nn.Sequential(
            nn.Linear(num_features, neuronios_c1),
            funcao_ativacao(),
            nn.Linear(neuronios_c1, neuronios_c2),
            funcao_ativacao(),
            nn.Linear(neuronios_c2, neuronios_c3),
            funcao_ativacao(),
            nn.Linear(neuronios_c3, num_targets),
        )

    def forward(self, x):
        '''Executa a rede a partir do pytorch.

        Args:
            self: MLP, instância própria da classe;
            x: torch.Tensor, tensores dos dados a serem passados pela reda.
        
        Returns:
            x: torch.Tensor, tensores com os valores previstos pela rede.
        '''
        x = self.camadas(x)
        return x



def cria_mlp(
        funcao_ativacao, x_treino, x_teste, y_treino, y_teste, normalizador_y,
        NEURONIOS_C1=50, NEURONIOS_C2=30, NEURONIOS_C3=25,
        TAXA_DE_APRENDIZADO=0.001, NUM_EPOCAS=5000
    ):

    '''Função que cria uma MLP (Multilayer perceptron) de 3 camadas e a treina
    com base nos dados passados, considerando uma determinada função de
    ativação, o normalizador utilizado nos dados, a quantidade de Neurônios das
    camadas 1, 2 e 3, a taxa de aprendizado a ser utilizada e o número de
    "épocas" de treino.

    Note:
        Para saber mais sobre a rede neural criada, qual seu objetivo e como ela
        é aplicada ao problema proposto, acesse o github:
        https://github.com/selaxco/Melting-Point-Prediction

    Args:
        funcao_ativacao: torch.nn.Module, a função de ativação a ser utilizada
        na rede neural como um todo;

        x_treino: torch.Tensor, valor das features de treino;

        x_teste: torch.Tensor, valores das features de teste;

        y_treino: torch.Tensor, valores dos targets de treino;

        y_teste: torch.Tensor, valores dos targets de teste;

        normalizador_y: sklearn.preprocessing.OneToOneFeatureMixin / 
        TransformerMixin / BaseEstimator, normalizador utilizado nos dados;

        NEURONIOS_C1 (default: 50): int, quantidade de neurônios na camada 1
        (hidden layer 1);

        NEURONIOS_C2 (default: 30): int, quantidade de neurônios na camada 2
        (hidden layer 2);
        
        NEURONIOS_C3 (default: 25): int, quantidade de neurônios na camada 3
        (hidden layer 3);

        TAXA_DE_APRENDIZADO (default: 0.001): float, taxa de aprendizado da
        rede;

        NUM_EPOCAS (default: 5000): int, número de "epocas" de treino da rede.

    Returns:
        mlp: torch.nn.Module, rede neural (MLP) criada e treinada;
        RMSE: float, erro médio da rede nas estimativas / predições.
    '''

    NUM_DADOS_DE_ENTRADA = x_treino.shape[1]
    NUM_DADOS_DE_SAIDA = y_treino.shape[1]

    mlp = MLP(
        funcao_ativacao, NUM_DADOS_DE_ENTRADA, NEURONIOS_C1, NEURONIOS_C2,
        NEURONIOS_C3, NUM_DADOS_DE_SAIDA
    )
    fn_perda = nn.MSELoss()
    otimizador = optim.Adam(mlp.parameters(), lr=TAXA_DE_APRENDIZADO)

    mlp.train()

    tipo = 1 if NUM_DADOS_DE_ENTRADA == 21 else 2
    desc = f'Progresso {funcao_ativacao.__name__} {tipo}: '
    loop = trange(NUM_EPOCAS, ncols=170, desc=desc, miniters=1)
    for _ in loop:
        y_pred = mlp(x_treino)

        otimizador.zero_grad()

        loss = fn_perda(y_pred, y_treino)
        loop.set_postfix_str(
            f'''Perda: {loss.data:<20.10f} / Neurons: {NEURONIOS_C1,
            NEURONIOS_C2, NEURONIOS_C3} / LR: {TAXA_DE_APRENDIZADO}'''
        )
        loop.refresh()
        
        loss.backward()

        otimizador.step()
    
    mlp.eval()
    
    with torch.no_grad():
        y_true = normalizador_y.inverse_transform(y_teste)
        y_pred = mlp(x_teste)
        y_pred = normalizador_y.inverse_transform(y_pred)

    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    print(f'Loss do teste: {RMSE}')

    return mlp, RMSE
