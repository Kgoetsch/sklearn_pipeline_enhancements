import boto3
import logging
import os
from sklearn.externals import joblib


def save_model_pipeline(pipeline, name=None):
    if name is not None:
        pipeline_save_file_path = './models/%s' % name
    else:
        pipeline_save_file_path = './models/%s' % pipeline.steps[-1][1].__class__.__module__
    if not os.path.isdir(pipeline_save_file_path):
        os.makedirs(pipeline_save_file_path)

    joblib.dump(pipeline, os.path.join(pipeline_save_file_path, pipeline.steps[-1][1].__class__.__module__ + '.pkl'))


def download_sklearn_model(model_path, s3_bucket_input):
    """

    :param model_path: s3 path where model is stored
    :param s3_bucket_input: bucket where model is stored
    :return: str to local temporary model files that have been downloaded
    """
    s3 = boto3.resource('s3')
    logging.info('connected to s3')
    target_bucket = s3.Bucket(s3_bucket_input)
    local_model_location = os.path.join('/tmp', model_path)
    if not os.path.exists(local_model_location):
        os.makedirs(local_model_location)
        logging.info('creating directory %s' % local_model_location)
    else:
        logging.info('Directory already exists at %s' % local_model_location)
    for target_object in target_bucket.objects.filter(
            Delimiter='/',
            Prefix=model_path):
        filename = os.path.join(local_model_location, target_object.key.split('/')[-1])
        logging.info('object name:%s' % filename)
        try:
            s3.meta.client.download_file(Bucket=target_object.bucket_name, Key=target_object.key, Filename=filename)
            logging.info('downloaded %s to %s' % ((target_object.bucket_name + target_object.key), filename))

            # overriding model path to point to downloaded files
            if filename.split('.')[-1] == 'pkl':
                model_path = filename
        except OSError:
            logging.info('could not write: %s' % filename)
        except:
            logging.info('totally failed on: %s' % filename)
    os.chmod(local_model_location, 0777)
    return model_path
