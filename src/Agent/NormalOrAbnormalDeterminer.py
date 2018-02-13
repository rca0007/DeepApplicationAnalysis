'''
Created on Feb 10, 2018

@author: rober
'''
import tensorflow as tensorflow
import pandas as pandas
import numpy as numpy
import argparse
from tensorflow.contrib.learn.python.learn.estimators.linear import LinearClassifier
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, 
                   help='number of training steps')
    
def main(argv):
    args = parser.parse_args(argv[1:])
    dataframe = pandas.read_csv("Data\\ProgramData.csv")
    
    features, labels = input_evaluation_set()
    
    my_feature_columns = []
    for key in features.keys():
        my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))
        
    classifier = tensorflow.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2)
    
    classifier.train(
        input_fn=lambda:train_input_fn(features, labels, args.batch_size),
                                        steps=args.train_steps)
    
    eval_resul= classifier.evaluate(
        input_fn=lambda:eval_input_fn(features, labels, args.batch_size))
    
    expected = [0, 1]
    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(features, labels=None, batch_size=args.batch_size))
    
    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
        
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        
        print(template.format(class_id, 100 * probability, expec))

def input_evaluation_set():
    features = {'SystemCalls' : numpy.array([[1], [2]])}
        
    labels = numpy.array([0, 1])
    return features, labels

def train_input_fn(features, labels, batch_size):
    dataset = tensorflow.data.Dataset.from_tensor_slices((dict(features), labels))
    
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
        
    dataset = tensorflow.data.Dataset.from_tensor_slices(inputs)
    
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    
    return dataset
    

if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main)
    