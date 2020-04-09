# -*- coding: utf-8 -*-
import redis
import json
import boto3
import io
import os
import pickle

redis_client = redis.Redis()

def get_user_models(user_name):
    return redis_client.lrange('{}_models'.format(user_name), 0, -1)

def get_model_instance_names(user_name, model_name):
    models = get_user_models(user_name, model_name)
    if model_name not in models:
        return []

    key = '{}_{}_instances'.format(user_name, model_name)
    return redis_client.lrange(key, 0, -1)

def get_model_instances(user_name, model_name):
    instance_names = get_model_instance_names(user_name, model_name)
    if not instance_names:
        return []

    result = []
    for i in instance_names:
        model_info = redis_client.hmget(i)
        result.append(model_info)
    return result

def get_instance_runs(user_name, model_name, instance_name):
    instance_names = get_model_instance_names
    if instance_name not in instance_names:
        return []

    key = '{}_{}_{}_runs'.format(user_name, model_name, instance_name)
    run_names = redis_client.lrange(key, 0, -1)
    runs = [get_instance_run for r in run_names]
    return runs

def register_model(user_name, model_name):
    redis_client.lpush('{}_models'.format(user_name), user_name + '_' + model_name)


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

        self.save_instance_info(instance_info)

    def data_from_s3(self, s3_url):
        # In reality we would use boto
        # s3 = boto3.client('s3')
        # obj = s3.get_object(Bucket='bucket', Key='key')
        # df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        train_path = os.path.abspath(s3_url)
        training_df = pd.read_csv(train_path, dtype={"PRODUCT.DETAILS":np.object})
        return training_df.to_numpy()

    def save_instance_info(self):
        k = '{}_{}_{}'.format(user_name, model_name, instance_name)
        redis_client.hmset(k, instance_info)
        k = '{}_{}_instances'.format(user_name, model_name)
        redis_client.lpush(k, instance_name)

    def load_model(self):
        # s3 = boto3.resource('s3')
        # with open(self.model_s3_key, 'wb') as data:
        #     s3.Bucket(self.model_s3_key).download_fileobj("tmp.pkl", data)

        with open(self.model_s3_key, 'rb') as data:
            return pickle.load(data)

    def save_trained_model(self, run_name, trained_model):
        local_url = "{}.p".format(run_name)
        pickle.dump(trained_model, open(local_url, "wb"))
        s3_url = 's3://altena_evaluator_service/trained_models/{}/{}/{}'.format(self.user_name, self.model_name, self.instance_name)
        send_to_s3(s3_url, local_url)
        redis_client.set(run_name + "_trained_model", s3_url)

    def save_test_results(self, run_name):
        local_url = "{}.csv".format(run_name)
        s3_url = 's3://altena_evaluator_service/test_results/{}/{}/{}'.format(self.user_name, self.model_name, self.instance_name)
        test_results.to_csv('out.csv', index=False)
        send_to_s3(test_results, local_url)
        redis_client.set(run_name + "_test_results", s3_url)

    def run_k_crossfold_validation(self, k, shuffle=True, random_gen=1):
        data = data_from_s3(self.train_data_s3_key)
        model = load_model(self.model_s3_key)
        kfold = KFold(k, shuffle, random_gen)
        i = 0
        for train, test in kfold.split(data):
            current_run = self.instance_name + "_run{}".format(i)

            prepped_train = model.prep(train)
            trained_model = model.train(prepped_train, *hyperparams)
            trained_s3_url = self.save_trained_model(current_run.format(i) + 'trained_model', trained_model)

            prepped_test = model.prep(test)
            test_results = model.test(prepped_test)
            tested_s3_url = self.save_test_results(current_run.format(i) + 'test_results', test_results)
            save_run_info(current_run, trained_s3_url, tested_s3_url)
            i += 1

     def save_run_info(self, run_name, trained_s3_url, tested_s3_url):
         run_info = {'instance': self.instance_info,
                     'run': {'name': run_name,
                             'trained_model': trained_s3_url,
                             'test_results': tested_s3_url
                             }
                    }
        k = '{}_{}_{}_{}'.format(self.user_name, self.model_name, self.instance_name, run_name)
        redis_client.hmset(run_name, run_info)
        k = '{}_{}_{}_runs'.format(self.user_name, self.model_name, self.instance_name)
        redis_client.lpush(k, current_run)

#cache hyperparameters with link to trained model and dataset trained on it
