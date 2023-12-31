import os
import joblib
import argparse
import pandas as pd

from sklearn.pipeline import Pipeline

from modules.GensimEstimator import SentenceVectorizer
from sklearn.cluster import KMeans

from modules.cleaner import message_cleaner
from modules import config

import nltk
from collections import Counter

clusters = config.clusters

def bigram_counts(kmeans, dataframe):
    '''
    Coletar frequências de bigramas para entender o nome apropriado para o cluster.
    '''

    df_clustered = pd.concat([dataframe, pd.Series(kmeans.labels_, name='subclusters_id')], axis=1)
    df_clustered['split_message'] = df_clustered.message.apply(lambda x: list(nltk.bigrams(x.split())))
    df_subclusters = df_clustered.groupby('subclusters_id', as_index=False).agg({'split_message': 'sum'})

    df_bigram_counts = df_subclusters.explode('split_message').reset_index(drop=True)
    df_bigram_counts = df_bigram_counts.groupby('subclusters_id', as_index=False).agg({'split_message': Counter})
    df_bigram_counts.split_message = df_bigram_counts.split_message.apply(Counter.most_common)

    # checar consistência das atribuições de subclusters aos clusters

    list_of_terms = []
    for k, v in clusters.items():
        for i in v:
            list_of_terms.append(i)
    if len(set(list_of_terms)) < len(list_of_terms):
        print('Há termos duplicados na atribuição dos clusters (config.clusters)')
    else:
        print('Sem termos duplicados na atribuição de clusters (config.clusters)')

    return df_bigram_counts


def naming(df_bigram_counts):
    df_cluster_names = df_bigram_counts.copy()

    df_cluster_names.split_message = df_cluster_names.split_message.apply(lambda x: [' '.join(x[0][0]), ' '.join(x[1][0]), ' '.join(x[2 if len(x) > 2 else 1][0]), ' '.join(x[3 if len(x) > 3 else 2][0])])

    df_cluster_names['cluster'] = df_cluster_names.split_message.apply(cluster_name)

    print('Subclusters não agregados a um cluster:')
    print(df_cluster_names.loc[df_cluster_names.cluster.isna()]['split_message'].reset_index(drop=True))

    return df_cluster_names


def cluster_name(x): # clusters
    '''Olha para cada cluster (keys), e caso o subcluster esteja contido nos values, retorna o nome do cluster.'''
    for i in x:
        for k, v in clusters.items():
            if i in v:
                return k

    
def save_json(df_cluster_names):
    with open(os.path.join(os.environ.get('SM_MODEL_DIR'), 'clusters.json'), 'w', encoding='utf-8') as file:
        dataframe = df_cluster_names[['subclusters_id', 'cluster']].set_index('subclusters_id')
        dataframe.to_json(file, force_ascii=False)

if __name__ == '__main__':

    '''
    Esta etapa recebe o arquivo de treino em formato csv, executa o tratamento de linguagem natural e aplica a pipeline de vetorização e clusterização.
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--social-network', default=None)
    args, _ = parser.parse_known_args()
    social_network = args.social_network
    
    input_data_path = os.path.join(os.environ.get('SM_CHANNEL_DATA'), 'data.csv')

    df = pd.read_csv(input_data_path)

    df, _ = message_cleaner(df, n_processors=36, stemming=True, social_network=social_network)

    split_X = df.message.str.split()

    model = Pipeline(steps=[('preprocessor', SentenceVectorizer(min_count=config.min_count, window=config.window, vector_size=config.n_dim, workers=16, sg=True)), 
                            ('clusterer', KMeans(n_clusters=config.n_clusters))
                     ]
    )
    
    print('Fitting model...')
    model.fit(df.message)

    kmeans = model['clusterer']
    
    joblib.dump(model, os.path.join(os.environ.get('SM_MODEL_DIR'), 'model.joblib'))
    
    # nomear os clusters
    df_bigram_counts = bigram_counts(kmeans, df)

    # coletar os 4 termos mais numerosos de cada k-means subcluster para atribuir a um cluster
    df_cluster_names = naming(df_bigram_counts)

    # salvar json com as relações entre label numérica do subcluster e nome do cluster
    save_json(df_cluster_names)