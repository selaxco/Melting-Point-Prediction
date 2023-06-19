from collections import Counter
import pandas as pd
import requests



def get_df(file):
    df = pd.read_excel(file)
    return df



def get_filtered_df(file, length, filters):
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