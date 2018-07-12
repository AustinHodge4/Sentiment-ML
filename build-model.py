import tensorflow as tf
import tensorflow_hub as hub

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from django.conf import settings

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'data')
    train_summary_directory = os.path.join(BASE_DIR, "tensorflowruns/run")
    export_model_directory = os.path.join(BASE_DIR, "exports")

    def get_dataframe(path):
        df = pd.DataFrame()
        for textfile in os.listdir(path):
            with open(os.path.join(path,textfile), encoding='utf-8') as infile:
                for line in infile.readlines():
                    line = line.strip()
                    text, label = line.replace('\t', '')[:-1], int(line[-1:])
                    df = df.append([[text, label]], ignore_index=True)

        df.columns = ['text', 'sentiment']
        return df
    def get_datasets(size=500):
        df = get_dataframe(data_path)
        return df.iloc[size:], df.iloc[:size]

    train_df, test_df = get_datasets()

    # Training input on the whole training set with no limit on training epochs. (default epochs = 1)
    train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["sentiment"], num_epochs=None, shuffle=True)

    # Prediction on the whole training set. 
    #predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["sentiment"], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(test_df, test_df["sentiment"], shuffle=False)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"text": np.array([["This is kind of great. I really enjoy it."]])}, shuffle=False)

    # Custom text feature column (words)
    embedded_text_feature_column = hub.text_embedding_column(key="text", module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    # Create deep NN model
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column], 
        n_classes=2, 
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.02),
        model_dir=train_summary_directory
        )
    # 1000  - 0.03  - 78.7999%
    # 1000  - 0.1   - 77.0%
    # 1000  - 0.003 - 76.0%
    # 1500  - 0.003 - 76.3999%
    # 5000  - 0.03  - 77.3999%
    # 10000 - 0.02  - 75.4000% [10]
    # 10000 - 0.02  - 79.000%
    # 10000 - 0.15  - 76.399%
    # 10000 - 0.015 - 76.800%
    # 10000 - 0.022 - 77.600%
    # 10000 - 0.021 - 77.000%

    # Train the model
    estimator.train(input_fn=train_input_fn, steps=1500)

    # Evaluate model
    #train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
    #print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Test set accuracy: {accuracy:.2%}".format(**test_eval_result))
    print("Test set loss: {loss}".format(**test_eval_result))

    def serving_input_receiver_fn():
        serialized_tf_tensor = tf.placeholder(dtype=tf.string, shape=[None, 1], 
                                            name='input_text_tensor')
        # buid the request for prediction
        receiver_tensors = {'text': serialized_tf_tensor}
        return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)

    # Export ML model
    estimator.export_savedmodel(export_model_directory, serving_input_receiver_fn)

    def get_predictions(estimator, input_fn):
        return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

    labels = {0:'negative', 1:'positive'}
    print("Prediction: {}".format(labels[get_predictions(estimator, predict_input_fn)[0]]))

    # Create a confusion matrix on training data.
    with tf.Graph().as_default():
        cm = tf.confusion_matrix(test_df["sentiment"], 
                            get_predictions(estimator, predict_test_input_fn))
        with tf.Session() as session:
            cm_out = session.run(cm)

    # Normalize the confusion matrix so that each row sums to 1.
    cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]
    labels = ['negative', 'positive']
    sb.heatmap(cm_out, annot=True, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
main()



    