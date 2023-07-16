import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import numpy as np
import re
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('rslp', quiet=True) # Removedor de Sufixos da Língua Portuguesa

from nltk.tokenize import sent_tokenize # divide o comentário em frases
from unidecode import unidecode # remove acentos

# Importar variáveis das configurações (config.py)

from modules import config

ignorar = config.ignorar
dictionary = config.dictionary # substituições por regex para simplificar o léxico
remove_from_stopword_list = config.remove_from_stopword_list # remover da lista de stopwords
add_to_stopword_list = config.add_to_stopword_list
useless_terms = config.useless_terms # termos removidos por regex
blacklist = config.blacklist # frases com essas palavras são desconsideradas
lower_limit = config.lower_limit
upper_limit = config.upper_limit # máximo de palavras

def message_cleaner(df, remove_blacklisted=True, tokenize=False):
    '''Recebe o dataframe dos dados e trata a coluna "message", retornando um dataframe tratado e outro com os comentários removidos do tratamento.'''

    df = df.copy(deep=True)

    print('Iniciando limpeza de texto')
    print('Removendo vazios')
    df = df.dropna(subset=['message']) # alguns comentários vêm vazios


    # Remover duplicados
    print('Proporção de duplicados removidos:', (100 * df.duplicated(subset=['message']).sum()/df.duplicated(subset=['message']).count()).round(2), '%')
    df = df.drop_duplicates(subset=['message'], keep='first').reset_index(drop=True)


    # Salvar original
    print('Salvando texto original')
    df['original_message'] = df.message


    # Remover comentários que iniciam com nomes (duas palavras capitalizadas)
    print('Removendo comentários que iniciam com nomes (duas palavras capitalizadas)')
    df_ = df.loc[(~df.original_message.str.contains(r'^[A-ZÀ-Ú][a-zà-ú]+(\s|,)[A-ZÀ-Ú][a-zà-ú]{1,15}', regex=True, na=False))]
    df_holdout_1 = df.loc[~df.index.isin(df_.index)]
    df = df_


    # ---------------------------------- modificações de texto


    # Lower case
    print('Transformando em caixa baixa')
    df.message = df.message.apply(str.lower)


    if tokenize:
        print('Dividindo comentários em frases')
        # Tokenizar os comentários, dividindo frases por pontos
        df.message = df.message.str.replace(r'\.\.+', '.', regex=True) # substituir "mais de 1 ponto" por 1 só
        df.message = df.message.str.replace(r'\s([?.!"](?:\s|$))', r'\1', regex=True) # remover espaços antes de pontuação
        df.message = df.message.str.replace(',', '.') # trocar vírgula por ponto ------------------- teste
        nltk.download('punkt', quiet=True) # lista de pontos para dividir frases
        df['tokenized'] = df.message.apply(sent_tokenize)


        # Expandir dataframe, onde cada frase do comentário se torna uma linha 
        df = df.explode(column='tokenized').reset_index(drop=True)
        df.message = df.tokenized.astype(str)
        df.drop(columns='tokenized', inplace=True)


    # Stopwords
    print('Removendo stopwords')
    stop_words = nltk.corpus.stopwords.words('english')

    for i in remove_from_stopword_list:
        stop_words.remove(i)

    removed = ', '.join(remove_from_stopword_list)
    print(f'Removidas palavras da lista de stopwords: {removed}')

    for i in add_to_stopword_list:
        stop_words.append(i)

    added = ', '.join(add_to_stopword_list)
    print(f'Adicionadas palavras na lista de stopwords: {added}')

    def stopword_remover(message):
        return ' '.join([word for word in re.split('\W+', message) if word not in stop_words and word.isnumeric() is False])

    df.message = df.message.apply(stopword_remover)
    

    # Duplicated spaces and special characters 
    print('Removendo excesso de espaços em branco, letras repetidas, símbolos e resquícios HTML')
    df.message = df.message.str.replace(r' +', ' ', regex=True) # remover excesso de espaços em branco
    df.message = df.message.str.replace(r'^\s+', '', regex=True) # remover espaços em branco do início de frases
    df.message = df.message.str.replace(r'kk+', '', regex=True) # remover kkkk
    df.message = df.message.str.replace(r'r\$', '', regex=True) # remover "R$"
    df.message = df.message.str.replace(r'(?:\\n)+', '. ', regex=True) # resquícios HTML
    df.message = df.message.str.replace(r'(?:\\t)+', '. ', regex=True) # quebras de texto são substituídas por pontos

    print('Removendo o que não é letra ou espaço em branco')
    df.message = df.message.str.replace(r'[^a-zA-ZÀ-ÿ ]+', '', regex=True) # remove what's left (not a letter or white space)
    

    # Outliers de comprimento: textos muito grandes ou muito pequenos prejudicam o desempenho do algoritmo (ver config.py)
    print('Removendo comentários muito grandes ou pequenos')
    df['message_size'] = df.message.str.split().apply(len)
    len_df = df.shape[0] # total sentences before filtering

    df.loc[df['message_size'] < lower_limit, 'str_len'] = 'small'
    df.loc[df['message_size'].between(lower_limit, upper_limit), 'str_len'] = 'medium'
    df.loc[df['message_size'] > upper_limit, 'str_len'] = 'large'
    
    size_upper = df.loc[df.str_len == 'large'].shape[0]
    size_lower = df.loc[df.str_len == 'small'].shape[0]

    print(f'{size_lower} frases com menos de {lower_limit} palavras foram removidas.')
    print(f'{size_upper} frases com mais de {upper_limit} palavras foram removidas.')

    # Com base no tamanho das frases, é utilizado o tamanho médio  

    df_ = df.loc[df.str_len == 'medium']
    df_ = df_.drop(columns=['message_size', 'str_len'])
    print(f'Restaram {df_.shape[0]} frases de um total de {len_df}.')

    df_holdout_5 = df.loc[~df.index.isin(df_.index)].drop(columns=['message_size', 'str_len']) # salvar frases que não serão consideradas para retornar ao final
    df = df_.reset_index(drop=True)


    # Substituições de alguns termos para melhorar o resultado (ver config.py > dictionary)
    print('Realizando substituições de sinônimos')
    def subs(message):
        for key in dictionary.keys():
            message = message.replace(key, dictionary[key])

        return message

    df.message = df.message.apply(subs)


    # Remover frases que não contenham as palavras obrigatórias
    print('Removendo frases com palavras da blacklist ou que não contenham as palavras obrigatórias')
    if remove_blacklisted:
        mandatory = config.mandatory
        contains_mandatory = df.message.str.contains('|'.join(mandatory)) if len(mandatory) > 0 else len(df.message) * [True]
        contains_blacklist = df.message.str.contains('|'.join(blacklist))

        df_ = df.loc[~contains_blacklist & contains_mandatory] # remover frases contendo termos da blacklist e não contendo termos obrigatórios
        df_holdout_2 = df.loc[~df.index.isin(df_.index)] # salvar dataframe de frases excluídas para concatenar ao final
        df = df_.reset_index(drop=True)
        total_removed = round(100 * df_holdout_2.shape[0]/(df_holdout_2.shape[0] + df.shape[0]), 1)

        print(f'{total_removed}% das frases foram removidas.')


    # Termos inúteis (não utilizar antes da remoção de stopwords)
    print('Removendo termos inúteis')
    def useless_terms_remover(message): # operação lenta para lista extensa de termos (ver config.py > useless_terms)

        for term in useless_terms:
            message = re.sub(r'{}'.format(term), ' ', message)

        return message

    df.message = df.message.apply(useless_terms_remover)


    # Stemming
    print('Stemizando')

    stem_total = len(set(' '.join(df.message.tolist()).split()))
    print(f'Total de palavras únicas antes do stemming: {stem_total}')

    df = df.reset_index(drop=True)

    docs = df.message.to_list()
    stemmer = nltk.stem.RSLPStemmer()
    stemmed_docs = []
    for doc in docs:
        new_docs = []
        for word in doc.split():
            new_docs.append(stemmer.stem(word))
        stemmed_docs.append(' '.join(new_docs))

    df.message = pd.Series(stemmed_docs) # back to series

    stem_total = len(set(' '.join(df.message.tolist()).split()))
    print(f'Total de palavras únicas após stemming: {stem_total}')


    print('Removendo excesso de espaço em branco, vazios e nulos')
    df.message =  df.message.str.replace(r' +', ' ', regex=True) # remover excesso de espaços em branco


    df = df.dropna(subset=['message']).reset_index(drop=True) # remover nulos

    print('Pré-processamento concluído')

    return df