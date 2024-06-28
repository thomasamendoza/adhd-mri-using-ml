import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow import keras


"""
TODO:
    1. fix "deployment size" column output (currently left as 0)
"""

def main():

    # -- DATAFRAME GENERATION --
    data_path = 'vmtable.csv'
    headers = ['vmid', 'subscriptionid', 'deploymentid', 'vmcreated', 'vmdeleted', 'maxcpu', 'avgcpu', 'p95maxcpu',
               'vmcategory', 'vmcorecount', 'vmmemory']
    trace_dataframe = pd.read_csv(data_path, header=None, index_col=False, names=headers, delimiter=',')

    deployment_data_path = 'deployments.txt'
    deployment_headers = ['deploymentid', 'deploymentsize']
    deployment_trace_dataframe = pd.read_csv(deployment_data_path, header=None, index_col=False,
                                             names=deployment_headers, delimiter='\t')

    # Compute VM Lifetime based on VM Created and VM Deleted timestamps and transform to Hour
    trace_dataframe['lifetime'] = np.maximum((trace_dataframe['vmdeleted'] - trace_dataframe['vmcreated']), 300) / 3600
    trace_dataframe['corehour'] = trace_dataframe['lifetime'] * trace_dataframe['vmcorecount']

    # Truncating deploymentid, vmid, subscriptionid to first 5 characters
    trace_dataframe['deploymentid'] = trace_dataframe['deploymentid'].apply(lambda x: x[:10])
    trace_dataframe['vmid'] = trace_dataframe['vmid'].apply(lambda x: x[:5])
    trace_dataframe['subscriptionid'] = trace_dataframe['subscriptionid'].apply(lambda x: x[:10])

    # Merge the two dataframes
    combinedDF = pd.merge(trace_dataframe, deployment_trace_dataframe, on='deploymentid', how='left')
    combinedDF['deploymentsize'].fillna(0, inplace=True)


    # -- PREPROCESSING --
    feat = combinedDF.drop(columns=['vmid', 'subscriptionid', 'deploymentid', 'vmcreated', 'vmdeleted', 'p95maxcpu'])
    target = combinedDF['p95maxcpu']

    numFeats = ['maxcpu', 'avgcpu', 'vmcorecount', 'vmmemory', 'corehour', 'deploymentsize']
    catFeats = ['vmcategory']

    numTransformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    catTransformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numTransformer, numFeats),
            ('cat', catTransformer, catFeats)
        ])

    feat = preprocessor.fit_transform(feat)  # finally, apply to the data

    # -- DATA SPLITTING --
    feat_train, feat_temp, target_train, target_temp = train_test_split(feat, target, test_size=0.3, random_state=24)
    feat_val, feat_test, target_val, target_test = train_test_split(feat_temp, target_temp, test_size=0.5, random_state=24)

    # -- MODEL DEFINITION AND TRAINING --
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(feat_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mse']
    )

    ES = keras.callbacks.EarlyStopping(
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
    )

    history = model.fit(
        feat_train, target_train,
        validation_data=(feat_val, target_val),
        epochs=100,
        batch_size=32,
        callbacks = [ES],
        # verbose = 0
        )

    test_loss, test_mae = model.evaluate(feat_test, target_test)
    print(f'Test MAE: {test_mae}')
    history_df = pd.DataFrame(history.history)
    history_df.loc[5:, ['loss']].plot()

    model.save('vm_p95maxcpu_model.h5')

    return 0

if __name__ == "__main__":
    main()
