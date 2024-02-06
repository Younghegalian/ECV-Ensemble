# noinspection PyInterpreter
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import RobustScaler

num_learners = 15

# Instantiate the scaler
scaler = RobustScaler()

# Load the weak learners and the meta learner
loaded_models = [tf.keras.models.load_model(f'weak_learner_{idx}') for idx in range(num_learners)]
loaded_meta_model = tf.keras.models.load_model('meta_model')


def ensemble_predict(new_data, weak_learners, meta_model):
    # Obtain predictions from the weak learners
    weak_predictions = np.column_stack([model.predict(new_data) for model in weak_learners])

    # Average those predictions
    average_predictions = np.mean(weak_predictions, axis=1, keepdims=True)

    # Append the average to the input data
    new_data_augmented = np.hstack([new_data, weak_predictions])

    # Fit the scaler
    scaler.fit(new_data_augmented)
    # Transform the data
    new_data_augmented = scaler.transform(new_data_augmented)

    # Feed the appended data into the meta model for the final prediction
    final_predictions = meta_model.predict(new_data_augmented)

    return final_predictions


# Load new data from CSV
new_data_df = pd.read_csv('encoded_test.csv')
old_data_df = pd.read_csv('encoded_train.csv')
new_data_df = new_data_df.drop(labels=[0], axis=0).drop(columns=['SAMPLE_ID', 'ID'])
old_data_df = old_data_df.drop(labels=[0], axis=0).drop(columns=['SAMPLE_ID', 'ID', 'CI_HOUR'])
print(new_data_df.shape)

new_data = new_data_df.values
old_data = old_data_df.values

# Fit the scaler to the training data
scaler.fit(old_data)

# Transform the training data
new_data = scaler.transform(new_data)

predictions_from_loaded_models = ensemble_predict(new_data, loaded_models, loaded_meta_model)

# Convert numpy array to DataFrame
df_predictions = pd.DataFrame(predictions_from_loaded_models)

# Clip negative values to 0
df_predictions.clip(lower=0, inplace=True)

# Save to CSV
df_predictions.to_csv('predictions.csv', index=False)
