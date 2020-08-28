import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
import matplotlib.pyplot as plt
import os
import re
import random
nltk.download('punkt')

# Load Text
file_paths = ['The Old Man and the Sea.txt', 'The Sun Also Rises.txt', 'A Farewell to Arms.txt']
text = ""
for f in file_paths:
    text += open('text/'+f, 'r', encoding='utf-8-sig').read().strip()

text = re.sub(r'[_*]', '', text)
text = re.sub(r'\s+', ' ', text)  
print('The set of characters: ', sorted(set(text)))

# Tokenization
sentences = nltk.tokenize.sent_tokenize(text)  # Split text into sentences.
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(sentences, target_vocab_size=2**13)
tokenized_sentences = [[tokenizer.vocab_size] + tokenizer.encode(s) + [tokenizer.vocab_size+1] for s in sentences]
print('Number of sentences: ', len(tokenized_sentences))

# Plot the distribution of sentence length.
if True:
    fig, axs = plt.subplots()
    axs.hist(list(map(len, tokenized_sentences)), 20)
    plt.show()


def filter_by_max(_list, _max_length):
    return list(filter(lambda x: len(x) <= _max_length, _list))


max_length = 40
tokenized_sentences = filter_by_max(tokenized_sentences, max_length)
data_size = len(tokenized_sentences)


# Shift sentences by one position to create input and output
data_input = [s[:-1] for s in tokenized_sentences]
data_output = [s[1:] for s in tokenized_sentences]


# Split into training and validation datasets
train_size = (data_size * 90) // 100
train_indices = [*range(data_size)]
random.shuffle(train_indices)
train_input = [data_input[i] for i in train_indices[:train_size]]
train_output = [data_output[i] for i in train_indices[:train_size]]
valid_input = [data_input[i] for i in train_indices[train_size:]]
valid_output = [data_output[i] for i in train_indices[train_size:]]


# Convert to TensorFlow dataset
batch_size = 64
buffer_size = train_size
num_epochs = 50

train_input = tf.keras.preprocessing.sequence.pad_sequences(train_input, padding='post')
train_output = tf.keras.preprocessing.sequence.pad_sequences(train_output, padding='post')
train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.repeat().batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

valid_input = tf.keras.preprocessing.sequence.pad_sequences(valid_input, padding='post')
valid_output = tf.keras.preprocessing.sequence.pad_sequences(valid_output, padding='post')
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_input, valid_output))
valid_dataset = valid_dataset.repeat().batch(batch_size, drop_remainder=True)


# Parameters
embedding_size = 128
vocab_size = tokenizer.vocab_size + 2  # +2 is for start and end.
lstm_units = 128


# Model
def build_model(_lstm_units, _batch_size, _vocab_size, _embedding_size, stateful=False):
    _model = tf.keras.Sequential()
    _model.add(tf.keras.layers.Embedding(_vocab_size, _embedding_size,
                                         batch_input_shape=(_batch_size, None)))
    _model.add(tf.keras.layers.LSTM(_lstm_units,
                                    return_sequences=True, stateful=stateful))
    _model.add(tf.keras.layers.Dense(_vocab_size))
    return _model


model = build_model(lstm_units, batch_size, vocab_size, embedding_size, stateful=False)
print(model.summary())


# Optimizer and Loss Function
optimizer_function = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# Directory where the checkpoints will be saved
checkpoint_path = "training_lstm/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq='epoch',
                                                 verbose=1)

if tf.train.latest_checkpoint(checkpoint_dir):
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    print('Latest checkpoint files are successfully restored.')


# Model Training
model.compile(optimizer=optimizer_function, loss=loss_function)
history = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=num_epochs,
                    steps_per_epoch=train_size//batch_size,
                    validation_steps=(data_size-train_size)//batch_size,
                    verbose=1,
                    callbacks=[cp_callback])


# Text Generation
def text_generator(_model, start, temperature=1.0):
    result = start
    output = [tokenizer.vocab_size] + tokenizer.encode(start)
    output = tf.expand_dims(output, 0)  # Expand by adding batch and time dimensions.

    for i in range(50):
        _model.reset_states()
        prediction = _model(output)
        prediction = tf.squeeze(prediction, 0)
        prediction = prediction / temperature
        prediction = tf.random.categorical(prediction, num_samples=1)[-1, 0]
        prediction = tf.squeeze(prediction).numpy()

        if prediction == tokenizer.vocab_size+1 or prediction == 0:
            return result.strip()

        result += tokenizer.decode([prediction])
        prediction = tf.expand_dims([prediction], 0)
        output = tf.concat([output, prediction], axis=-1)

    return result.strip()


# Prediction Model
prediction_model = build_model(lstm_units, 1, vocab_size, embedding_size, stateful=False)
prediction_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
print(text_generator(prediction_model, 'I '))
