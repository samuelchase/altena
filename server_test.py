# -*- coding: utf-8 -*-

import requests
import redis
import pprint
import json

import redis
redis_client = redis.Redis()
redis_client.flushdb()

print('checking...  /')
response = requests.get('http://127.0.0.1:5000/')
print(response.content)

print('\n')
print('initializing ML model...')

d = {
     'user_name': 'sam',
     'model_name': 'countryclassifier',
     'instance_name': 'non_china_500_32',
     'hyperparams': [500, 32, 32],
     'model_s3_key': 'countryclassifier.pkl',
     'train_data_s3_key': 'ds-project-test.csv',
     'test_data_s3_key': 'ds-project-train.csv',
     'k': 3
}


response = requests.post('http://127.0.0.1:5000/evaluator/kfold', json=d)
print(json.loads(response.content))

print('\nget models')
response = requests.get('http://127.0.0.1:5000/models/sam/')
print(json.loads(response.content))

print('\nget instances')
response = requests.get('http://127.0.0.1:5000/instances/sam/countryclassifier')
print(response.content)

print('\nget runs')
response = requests.get('http://127.0.0.1:5000/runs/sam/countryclassifier/non_china_500_32')
print(response.content)
