import tensorflow as tf
import numpy as np
import sys
import argparse
import os
from collections import defaultdict

tf.logging.set_verbosity(tf.logging.INFO)
MAX_PREDICTION_SIZE=200

def load_data(data_file, vocab):
    def gen():
        with open(data_file, 'r') as data:
            for line in data:
                if len(line.strip()) > MAX_PREDICTION_SIZE:
                    continue
                yield [vocab[char] for char in line.strip()] + [vocab['</S>']]
    return gen

def train_input_fn(train_data_file, vocab, batch_size):
    def fn():
        # get labels
        ds = tf.data.Dataset.from_generator(load_data(train_data_file, vocab), (tf.int32))
        # add features (inputs)
        ds = ds.map(lambda x: (tf.concat([[vocab['<S>']], x[:-1]], 0), x))
        # batch
        ds = ds.padded_batch(batch_size, padded_shapes=([None], [None]), padding_values=(vocab['<PAD>'], vocab['<PAD>']))
        ds = ds.shuffle(100)
        ds = ds.repeat()
        features = ds.make_one_shot_iterator().get_next()
        return features
    return fn

def eval_input_fn(train_data_file, vocab, batch_size):
    def fn():
        # get labels
        ds = tf.data.Dataset.from_generator(load_data(train_data_file, vocab), (tf.int32))
        # add features (inputs)
        ds = ds.map(lambda x: (tf.concat([[vocab['<S>']], x[:-1]], 0), x))
        # batch
        ds = ds.padded_batch(batch_size, padded_shapes=([None], [None]), padding_values=(vocab['<PAD>'], vocab['<PAD>']))
        features = ds.make_one_shot_iterator().get_next()
        return features
    return fn

def predict_input_fn(vocab):
    def fn():
        return tf.constant([vocab['<S>']])
    return fn

def serving_input_fn():
    return tf.estimator.export.ServingInputReceiver({}, {})

def init_vocab(vocab_file):
    vocab = defaultdict(int)
    vocab['<UNK>'] = 0
    with open(vocab_file, 'r') as f:
        for line in f:
            vocab[line[:-1]] = len(vocab)
    vocab['<S>'] = len(vocab)
    vocab['</S>'] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    return vocab

def idx_to_string(idx, vocab):
    ivocab = {v:k for k,v in vocab.items()}
    if vocab['</S>'] in idx:
        idx = idx[:np.where(idx==vocab['</S>'])[0][0]]
    if vocab['<PAD>'] in idx:
        idx = idx[:np.where(idx==vocab['<PAD>'])[0][0]]
    return ''.join([ivocab[i] for i in idx])

def idxs_to_string(idxs, vocab):
    if len(idxs.shape) == 2:
        strings = []
        for idxs_part in idxs:
            strings.append(idx_to_string(idxs_part))
        return strings
    else:
        #return ''.join([ivocab[idx] for idx in idxs])
        return idx_to_string(idxs, vocab)

def build_model_fn(vocab, embedding_size, hidden_size, temperature, dropout):
    def model_fn(features, labels, mode, params):
        embeddings = tf.get_variable('Embeddings', [len(vocab), embedding_size])
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_size), tf.contrib.rnn.LSTMCell(hidden_size)])
        output_layer = tf.layers.Dense(len(vocab), None)
        if mode == tf.estimator.ModeKeys.TRAIN:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0-dropout, output_keep_prob=1.0-dropout)
            inputs = tf.nn.embedding_lookup(embeddings, features)
            masks = tf.to_float(tf.not_equal(labels, vocab['<PAD>']))
            lengths = tf.reduce_sum(masks, 1)
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=lengths, dtype=tf.float32) 
            logits = output_layer(outputs)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=masks)
            loss = tf.Print(loss, ['loss', loss])
            optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
            opt = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=opt)
        elif mode == tf.estimator.ModeKeys.EVAL:
            inputs = tf.nn.embedding_lookup(embeddings, features)
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32) 
            logits = output_layer(outputs)
            masks = tf.to_float(tf.not_equal(labels, vocab['<PAD>']))
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=masks)
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            #prev = features
            prev = tf.constant([vocab['<S>']])
            state = cell.zero_state(1, dtype=tf.float32)
            predictions = []
            for i in range(MAX_PREDICTION_SIZE):
                inputs = tf.nn.embedding_lookup(embeddings, prev)
                with tf.variable_scope('rnn'):
                    outputs, state = cell(inputs, state)
                logits = output_layer(outputs)
                prev = tf.distributions.Categorical(logits=logits/temperature).sample()
                #prev = tf.Print(prev, ['probs', tf.reduce_max(tf.nn.softmax(logits/temperature))], summarize=200)
                #prev = tf.argmax(logits, axis=1)
                predictions.append(prev)
            predictions = tf.stack(predictions, axis=1)
            tokens = tf.concat([tf.squeeze(predictions), [vocab['</S>']]], 0)
            end = tf.where(tf.equal(tokens, vocab['</S>']))[0]
            tokens = tokens[:end[0]]
            tokens = tf.concat([tokens, [vocab['<PAD>']]], 0)
            pad = tf.where(tf.equal(tokens, vocab['<PAD>']))[0]
            tokens = tokens[:pad[0]]

            #print('vals', list(vocab.keys()))
            mapping = tf.constant(list(vocab.keys()))
            table = tf.contrib.lookup.index_to_string_table_from_tensor(mapping, default_value="<UNK>")
            tokens = table.lookup(tf.to_int64(tokens))

            #TODO PAD with \s, join as one
            #tokens = tf.string_join(tokens)
            predictions = tf.expand_dims(tokens, 0)
            #predictions = tf.Print(predictions, ['P', tf.shape(predictions), predictions], summarize=15)
            #predictions = tf.transpose(tf.expand_dims(predictions, 1))
            return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                    export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:tf.estimator.export.PredictOutput(predictions)})
        else:
            raise ValueError("Other estimator modes are not implemented yet")
    return model_fn

def run_test(args, vocab):
    #Test
    fn = train_input_fn(args.train_file, vocab, args.batch_size)

    iterator = fn()
    sess = tf.Session()
    data = sess.run(iterator)
    for sent in idx_to_string(data[1], vocab):
        print(sent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='model', help='model dir')
    parser.add_argument('--train_file', default=None, help='train file')
    parser.add_argument('--eval_file', default=None, help='dev file')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--embedding_size', default=128, help='batch size')
    parser.add_argument('--hidden_size', default=128, help='batch size')
    parser.add_argument('--train_steps', default=100, type=int, help='train steps')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout')
    parser.add_argument('--vocab_file', help='vocab file')
    parser.add_argument('--samples', default=15, type=int, help='how many samples to generate')
    parser.add_argument('--export', action='store_true', help='export')
    args = parser.parse_args()
    vocab = init_vocab(args.vocab_file)
    model_fn = build_model_fn(vocab, args.embedding_size, args.hidden_size, args.temperature, args.dropout)
    run_config = tf.estimator.RunConfig(model_dir=args.model_dir, save_checkpoints_steps=100)
    estimator = tf.estimator.Estimator(model_fn,
            config=run_config)
    if args.train_file:
        estimator.train(input_fn=train_input_fn(args.train_file, vocab, args.batch_size),
                steps=args.train_steps)
    elif args.eval_file:
        loss = estimator.evaluate(input_fn=eval_input_fn(args.eval_file, vocab, args.batch_size))
        print('loss', loss)
        print('ppl', np.exp(loss['loss']))
    elif args.export:
        #estimator.export_savedmodel(os.path.join(args.model_dir, 'export'), serving_input_fn)
        estimator.export_savedmodel(os.path.join(args.model_dir, 'export'), tf.estimator.export.build_raw_serving_input_receiver_fn({'inp': tf.placeholder(tf.int32, [1])}))
        #estimator.export_savedmodel(os.path.join(args.model_dir, 'export'), tf.estimator.export.build_raw_serving_input_receiver_fn({}))
    else:
        for i, out in enumerate(estimator.predict(input_fn=predict_input_fn(vocab))):
            #print(idx_to_string(out, vocab))
            print(''.join([o.decode('utf8') for o in out]))
            if i > args.samples:
                break

    #run_test(args, vocab)

