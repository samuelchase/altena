# -*- coding: utf-8 -*-
import redis
import json
# import boto3
import io
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

redis_client = redis.Redis()

def blist_to_ascii(list):
    return [s.decode('ascii') for s in list]

def get_user_models(user_name):
    return blist_to_ascii(redis_client.lrange('{}_models'.format(user_name), 0, -1))

def get_model_instance_names(user_name, model_name):
    models = get_user_models(user_name)
    if '{}_{}'.format(user_name, model_name) not in models:
        return []

    key = '{}_{}_instances'.format(user_name, model_name)
    return blist_to_ascii(redis_client.lrange(key, 0, -1))

def get_model_instances(user_name, model_name):
    instance_names = get_model_instance_names(user_name, model_name)
    if not instance_names:
        return []

    result = []
    for i in instance_names:
        print(i)
        model_info = json.loads(redis_client.get(i))
        result.append(model_info)
    return result

def get_instance_runs(user_name, model_name, instance_name):
    instance_names = get_model_instance_names(user_name, model_name)
    full_instance_name = '{}_{}_{}'.format(user_name, model_name, instance_name)
    print(full_instance_name)
    if full_instance_name not in instance_names:
        return []

    key = '{}_{}_{}_runs'.format(user_name, model_name, instance_name)
    run_names = blist_to_ascii(redis_client.lrange(key, 0, -1))
    print('run names for key {}'.format(key))
    print(run_names)
    runs = [json.loads(redis_client.get(r)) for r in run_names]
    return runs

def register_model(user_name, model_name):
    full_model_name = user_name + '_' + model_name
    models = redis_client.lrange('{}_models'.format(user_name), 0, -1)
    if full_model_name not in models:
        redis_client.lpush('{}_models'.format(user_name), full_model_name)


class MLEvaluator(object):
    def __init__(self, user_name, model_name, instance_name,
                 hyperparams, model_s3_key, train_data_s3_key,
                 test_data_s3_key):

        self.user_name = user_name
        self.model_name = model_name
        self.instance_name = instance_name
        self.hyperparams=hyperparams
        self.model_s3_key = model_s3_key
        self.train_data_s3_key = train_data_s3_key
        self.test_data_s3_key = test_data_s3_key

        self.instance_info = {'user_name': self.user_name,
                              'model_name': self.model_name,
                              'instance_name': self.instance_name,
                              'hyperparameters': self.hyperparams,
                              'model_s3_key':self.model_s3_key,
                              'train_data_s3_key': self.train_data_s3_key,
                              'test_data_s3_key': self.test_data_s3_key,
                             }

        register_model(user_name, model_name)
        self.save_instance_info()

    def data_from_s3(self, s3_url):
        # In reality we would use boto
        # s3 = boto3.client('s3')
        # obj = s3.get_object(Bucket='bucket', Key='key')
        # df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        train_path = os.path.abspath(s3_url)
        training_df = pd.read_csv(train_path, dtype={"PRODUCT.DETAILS":np.object})
        return training_df.to_numpy()

    def save_instance_info(self):
        k = '{}_{}_{}'.format(self.user_name, self.model_name, self.instance_name)
        redis_client.set(k, json.dumps(self.instance_info))
        k = '{}_{}_instances'.format(self.user_name, self.model_name)
        print(k)
        redis_client.lpush(k, '{}_{}_{}'.format(self.user_name, self.model_name, self.instance_name))

    def load_model(self):
        # s3 = boto3.resource('s3')
        # with open(self.model_s3_key, 'wb') as data:
        #     s3.Bucket(self.model_s3_key).download_fileobj("tmp.pkl", data)

        # Stub
        return {'model':100}

    def save_trained_model(self, run_name, trained_model):
        local_url = "{}.p".format(run_name)
        pickle.dump(trained_model, open(local_url, "wb"))
        s3_url = 's3://altena_evaluator_service/trained_models/{}/{}/{}'.format(self.user_name, self.model_name, self.instance_name)
        send_to_s3(s3_url, local_url)
        redis_client.set(run_name + "_trained_model", s3_url)
        return s3_url

    def save_test_results(self, run_name, test_results):
        local_url = "{}.csv".format(run_name)
        s3_url = 's3://altena_evaluator_service/test_results/{}/{}/{}'.format(self.user_name, self.model_name, self.instance_name)
        test_results.to_csv('out.csv', index=False)
        send_to_s3(test_results, local_url)
        redis_client.set(run_name + "_test_results", s3_url)
        return s3_url

    def run_k_crossfold_validation(self, k, shuffle=True, random_gen=1):
        data = self.data_from_s3(self.train_data_s3_key)
        model = self.load_model()
        kfold = KFold(k, shuffle, random_gen)
        i = 0
        for train, test in kfold.split(data):
            current_run = self.instance_name + "_run{}".format(i)

            # prepped_train = model.prep(train)
            # trained_model = model.train(prepped_train, *hyperparams)
            trained_model = trained_model_stub()
            trained_s3_url = self.save_trained_model(current_run.format(i) + 'trained_model', trained_model)

            # prepped_test = model.prep(test)
            # test_results = model.test(prepped_test)
            test_results = test_result_stub()
            tested_s3_url = self.save_test_results(current_run.format(i) + 'test_results', test_results)
            self.save_run_info(current_run, trained_s3_url, tested_s3_url)
            i += 1

    def save_run_info(self, run_name, trained_s3_url, tested_s3_url):
        run_info = {'instance': self.instance_info,
                     'run': json.dumps({'name': run_name,
                             'trained_model': trained_s3_url,
                             'test_results': tested_s3_url
                             })
                    }
        full_run_name = '{}_{}_{}_{}'.format(self.user_name, self.model_name, self.instance_name, run_name)
        print('saving to {}'.format(full_run_name))
        redis_client.set(full_run_name, json.dumps(run_info))
        k = '{}_{}_{}_runs'.format(self.user_name, self.model_name, self.instance_name)
        print('pushing run to {}'.format(k))
        redis_client.lpush(k, full_run_name)


def test_result_stub():
    data = [['accuracy', 10], ['precision', 15], ['fscore', 14]]
    df = pd.DataFrame(data, columns = ['evaluation_type', 'score'])
    return df

def trained_model_stub():
    pretend_trained_model = {'somemodel': 100}
    return pretend_trained_model

def send_to_s3(s3_url, local_url):
    # writes object to local file and sends to s3
    return
