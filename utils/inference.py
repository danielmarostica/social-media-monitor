# -*- coding: utf-8 -*-
'''
A inferência trará do Redshift Spectrum dados referentes a comentários nos últimos 7 dias, e retornará "output.csv" com as classificações dos mesmos.
'''

import pandas as pd
import tarfile
import joblib
import csv
import numpy as np
import sys
sys.path.append('modules')

import re
import boto3
import sagemaker

from scipy.spatial.distance import cdist

from modules.cleaner import message_cleaner
from modules import config

def connect_aws(social_network):
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    prefix = f'social-media-cx-monitor/dafiti_br/comentarios_{social_network}'

    return bucket, prefix

def retrieve_data(sequential: bool, bucket, prefix, social_network):
    '''Permite execução normal (últimos 30 dias) ou sequencial, para simular execuções em dias consecutivos (ver modules/inference_sequential).'''    

    # role = iam.get_role(RoleName='datascience-sagemaker-s3-redshift')['Role']['Arn']
    s3_client = boto3.client('s3')


    # load training job name (from sagemaker models list)
    client = boto3.client(service_name='sagemaker')
    training_job_name = client.list_models(NameContains=f'cx-social-comments-dafiti-br-{social_network}', SortBy='CreationTime', MaxResults=100)
    training_job_name = training_job_name['Models'][0]['ModelName']
    print('Modelo encontrado: ', training_job_name)

    if sequential:
        root = '../temp' # inference_sequential.py is inside the "extra" folder
    else:
        root = 'temp'


    # get sagemaker model attributes/weights
    s3_client.download_file(bucket, f'{prefix}/training-jobs/{training_job_name}/output/model.tar.gz', f'{root}/model.tar.gz')
    tar = tarfile.open(f'{root}/model.tar.gz')
    tar.extractall(f'{root}/')
    model = joblib.load(f'{root}/model.joblib')


    # get cluster names
    clusters = pd.read_json(f'{root}/clusters.json').reset_index().rename({'index': 'subclusters_id'}, axis=1)

    # get data for inference
    if sequential:
        original_data = pd.read_csv(f'data_for_inference_{social_network}.csv', delimiter=';', quotechar='"')
    else:
        original_data = pd.read_csv(f's3://{bucket}/{prefix}/data_for_inference.csv', delimiter=';', escapechar='\\')

    clean_data, removed = message_cleaner(original_data, inference=True, stemming=True, social_network=social_network)

    return model, clusters, clean_data, removed


def outliers(X, model, clusters, percent_outliers):

    X = X.values

    # obtaining the centers of the clusters
    centroids = model.cluster_centers_

    # points array will be used to reach the index easy
    points = np.empty((0, len(X[0])), float)

    # distances will be used to calculate outliers
    distances = np.empty((0, len(X[0])), float)

    # getting points and distances
    for i, center_elem in enumerate(centroids):
        # cdist is used to calculate the distance between center and other points
        distances = np.append(distances, cdist([center_elem], X[clusters == i], 'euclidean')) 
        points = np.append(points, X[clusters == i], axis=0)

    # getting outliers whose distances are greater than some percentile
    outliers = np.where(distances > np.percentile(distances, 100-percent_outliers), 1, 0)
    
    return outliers


def get_cluster_size(kmeans, clusters):

    cluster_size = pd.Series(kmeans.labels_, name='size').value_counts().sort_index()
    clusters = pd.concat([clusters, cluster_size], axis=1)

    return clusters

def get_mode(df_in):
    mode = list(pd.Series.mode(df_in.cluster)) # compute mode

    if len(mode) > 1: # if no mode, choose the biggest cluster
        biggest_cluster = max(df_in['size'])
        biggest_cluster_name = df_in.loc[df_in['size'] == biggest_cluster]['cluster'].tolist()[0]
        return biggest_cluster_name
    elif mode == []:
        return None
    else:
        return mode[0]


def infer(model, clean_data):

    # predict
    predictions = model.predict(clean_data.message)

    return predictions

def post_process(predictions, clusters, clean_data):

    # Adicionar output do predict ao dataframe
    df_clustered = pd.concat([clean_data, pd.Series(predictions, name='subclusters_id')], axis=1)
    df_clustered = pd.merge(df_clustered, clusters)


    # O cluster mais incerto é ignorado
    df_clustered.cluster = df_clustered.cluster.replace('Cluster não identificado', np.nan)

    return df_clustered


def prepare_for_upload(df_clustered):

    # Eliminar colunas desnecessárias
    df_clustered.drop(columns=['message', 'clusters', 'size'], inplace=True, errors='ignore')


    # Coletar números de pedidos
    df_clustered.original_message = df_clustered.original_message.fillna('') # re.findall não funciona com nans
    df_clustered['order_number'] = df_clustered.original_message.apply(lambda x: re.findall(r'(\d{8,10})', x) or None).str[0]


    df_clustered.interaction_date = pd.to_datetime(df_clustered.interaction_date).dt.date

    df_clustered = df_clustered[['interaction_date', 'original_message', 'cluster']] # ordenar

    return df_clustered


def upload(df_clustered):

    df_clustered.to_csv(f's3://{bucket}/{prefix}/output.csv', index=False, quoting=csv.QUOTE_ALL)

    print('Output uploaded to S3')


if __name__ == "__main__":

    model, clusters, clean_data = retrieve_data(sequential=False, bucket=bucket, prefix=prefix)

    clusters = get_cluster_size(model['clusterer'], clusters)

    predictions = infer(model, clean_data)

    df_clustered = post_process(predictions, clusters, clean_data)
        
    df_clustered = prepare_for_upload(df_clustered)

    upload(df_clustered)