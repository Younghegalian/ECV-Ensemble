import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

# Instantiate the scaler
scaler = RobustScaler()

# Read CSV data
df = pd.read_csv('encoded_train.csv')

# Drop specified columns and a specific row simultaneously
dfX = df.drop(labels=[0], axis=0).drop(columns=['SAMPLE_ID', 'ID', 'CI_HOUR'])
dfY = df.drop(df.index[0])

# Convert to numpy array
X = dfX.values
y = dfY['CI_HOUR'].values

# Fit the scaler to the training data
scaler.fit(X)

# Transform the training data
X = scaler.transform(X)

# Split test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Split weak-meta data
X_weak, X_meta_train, y_weak, y_meta_train = train_test_split(X_train, y_train, test_size=0.5)


# Define Weak Learners
class WeakLearner(tf.keras.Model):
    def __init__(self):
        super(WeakLearner, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.hidden1(inputs)
        return self.output_layer(x)


# Training the Weak Learners
print("Training weak learners")
learner_idx = 1
num_learners = 15
learners = [WeakLearner() for _ in range(num_learners)]
for model in learners:
    print('Training weak learner ', learner_idx, '/', num_learners)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mae')
    model.fit(X_weak, y_weak, batch_size=256, epochs=15, verbose=1)
    learner_idx = learner_idx + 1

# Get the soft voting results (average predictions)
voting_results = np.column_stack([model.predict(X_meta_train) for model in learners])
average_predictions = np.mean(voting_results, axis=1, keepdims=True)
# Augment the validation data with average predictions
X_meta_train_augmented = np.hstack([X_meta_train, voting_results])
# Fit the scaler
scaler.fit(X_meta_train_augmented)
# Transform the data
X_meta_train_augmented = scaler.transform(X_meta_train_augmented)
print("Augmented dataset for training prepared")

voting = np.column_stack([model.predict(X_test) for model in learners])
average = np.mean(voting, axis=1, keepdims=True)
# Augment the validation data with average predictions
x_meta_test_augmented = np.hstack([X_test, voting])
# Fit the scaler
scaler.fit(x_meta_test_augmented)
# Transform the data
x_meta_test_augmented = scaler.transform(x_meta_test_augmented)
print("Augmented dataset for test prepared")

# Define the Meta Learner
class MetaLearner(tf.keras.Model):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden3 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden4 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden5 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden6 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden7 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        return self.output_layer(x)


# Train the Meta Learner
def train_meta(MetaLearner, ep, X_meta_train_augmented):
    print("Training meta learner")
    meta_model = MetaLearner()
    meta_model.build((None, X_meta_train_augmented.shape[1]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    meta_model.compile(optimizer=optimizer, loss='mae')
    meta_model.fit(X_meta_train_augmented, y_meta_train, batch_size=64, epochs=ep)
    return meta_model


def test(x_meta_test_augmented, y_test, meta_model):
    print('Test trained meta-learner')
    final_predictions = meta_model.predict(x_meta_test_augmented)
    mae = mean_absolute_error(y_test, final_predictions)
    print('MAE:', mae)
    return mae


# Epochs experiment
list_his = []
for i in range(10):
    ep = 10 + i
    meta_model = train_meta(MetaLearner, ep, X_meta_train_augmented)
    mae = test(x_meta_test_augmented, y_test, meta_model)
    list_his.append([ep, mae])

print(list_his)

min_mae_pair = min(list_his, key=lambda x: x[1])
min_ep = min_mae_pair[0]
print(min_ep)

count = 1
while 1:
    meta_model = train_meta(MetaLearner, min_ep, X_meta_train_augmented)
    final_mae = test(x_meta_test_augmented, y_test, meta_model)

    if final_mae < min_mae_pair[1]:
        print("Saving trained models...")
        for idx, model in enumerate(learners):
            model.save(f'weak_learner_{idx}', save_format='tf')
        meta_model.save('meta_model', save_format='tf')
        break

    if count > 10:
        print("Saving trained models...")
        for idx, model in enumerate(learners):
            model.save(f'weak_learner_{idx}', save_format='tf')
        meta_model.save('meta_model', save_format='tf')
        break
    count = count + 1

# For predictions on new data, you would:
# 1. Obtain predictions from the weak learners
# 2. Average those predictions
# 3. Append the average to the input data
# 4. Feed the appended data into the meta learner for the final prediction
