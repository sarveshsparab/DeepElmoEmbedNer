import os
from datetime import datetime
import numpy as np
import pickle
import logging
import sys
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from model.Elmo import ElmoModel
from utils.LSTMCNNCRFeeder import LSTMCNNCRFeeder
from utils.conlleval import evaluate
from utils.checkmate import BestCheckpointSaver, best_checkpoint

from model.bilm.data import Batcher
from model.bilm.model import BidirectionalLanguageModel
from model.ner import NER


class DeepElmoEmbedNer(NER):

    # Setting up the logger
    log = logging.getLogger('root')
    logdatetime = datetime.now().strftime("%H_%M_%S")
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
    log.setLevel(logging.DEBUG)
    logFormatter = logging.Formatter(FORMAT)
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler('../logs/app-' + logdatetime + '.log')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Logging for tensorflow
    Path('../results').mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('../logs/tf-' + logdatetime + '.log'),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers

    def __init__(self):
        pass

    def convert_ground_truth(self, data, *args, **kwargs):
        self.log.debug("Invoked convert_ground_truth method")
        self.log.debug("With parameters : ")
        self.log.debug(data)
        self.log.debug(args)
        self.log.debug(kwargs)

        tag_contents = list()
        for i, line in enumerate(data['test']):
            # Check if end of sentence or not
            if len(line) == 0:
                continue
            else:
                tag_contents.append(
                    [None, None, line[kwargs.get("wordPosition", 0)], line[kwargs.get("tagPosition", 3)]])

        self.log.debug("Returning ground truths for the test input file :")
        self.log.debug(tag_contents)

        if kwargs.get("writeGroundTruthToFile", True):
            with Path(kwargs.get("groundTruthPath", '../results/groundTruths.txt')).open(mode='w') as f:
                for x in range(len(tag_contents)):
                    line = ""
                    if tag_contents[x][0] is None:
                        line += "-" + " "
                    else:
                        line += tag_contents[x][0] + " "
                    if tag_contents[x][1] is None:
                        line += "-" + " "
                    else:
                        line += tag_contents[x][1] + " "
                    line += tag_contents[x][2] + " "
                    line += tag_contents[x][3]
                    line += "\n"
                    f.write(line)

        return tag_contents

    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):
        self.log.debug("Invoked read_dataset method")
        self.log.debug("With parameters : ")
        self.log.debug(file_dict)
        self.log.debug(dataset_name)
        self.log.debug(args)
        self.log.debug(kwargs)

        standard_split = ["train", "test", "dev"]
        data = {}
        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    if kwargs.get("fileHasHeaders", True):
                        next(f)
                        next(f)
                    raw_data = f.read().splitlines()
                for i, line in enumerate(raw_data):
                    if len(line.strip()) > 0:
                        raw_data[i] = line.strip().split()
                    else:
                        raw_data[i] = list(line)
                data[split] = raw_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)
        return data

    def train(self, data, *args, **kwargs):

        if not os.path.isfile(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl')):
            self.data_converter(data, *args, **kwargs)

        with open(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl'), 'rb') as fp:
            train_set, val_set, test_set, dicts = pickle.load(fp)

        w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']
        idx2w = {w2idx[k]: k for k in w2idx}
        idx2la = {la2idx[k]: k for k in la2idx}

        train_x, train_chars, train_la = train_set
        val_x, val_chars, val_la = val_set
        test_x, test_chars, test_la = test_set

        self.log.debug('Loading elmo!')
        elmo_batcher = Batcher(kwargs.get("vocabPath", '../dev/vocab.txt'), 50)
        elmo_bilm = BidirectionalLanguageModel(kwargs.get("elmoOptionsFile", '../resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'),
                                               kwargs.get("elmoWeightFile", '../resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'))

        self.log.debug('Loading model!')

        num_classes = len(la2idx.keys())
        max_seq_length = max(
            max(map(len, train_x)),
            max(map(len, test_x)),
        )
        max_word_length = max(
            max([len(ssc) for sc in train_chars for ssc in sc]),
            max([len(ssc) for sc in test_chars for ssc in sc])
        )

        model = ElmoModel(
            True,
            kwargs.get("wordEmbeddingSize", 50),  # Word embedding size
            kwargs.get("charEmbeddingSize", 16),  # Character embedding size
            kwargs.get("LSTMStateSize", 200),  # LSTM state size
            kwargs.get("filterNum", 128),  # Filter num
            kwargs.get("filterSize", 3),  # Filter size
            num_classes,
            max_seq_length,
            max_word_length,
            kwargs.get("learningRate", 0.015),
            kwargs.get("dropoutRate", 0.5),
            elmo_bilm,
            1,  # elmo_mode
            elmo_batcher,
            **kwargs)

        self.log.debug('Start training...')
        self.log.debug('Train size = %d' % len(train_x))
        self.log.debug('Val size = %d' % len(val_x))
        self.log.debug('Test size = %d' % len(test_x))
        self.log.debug('Num classes = %d' % num_classes)

        start_epoch = 1
        max_epoch = kwargs.get("maxEpoch", 100)

        self.log.debug('Epoch = %d' % max_epoch)

        saver = tf.train.Saver()
        best_saver = BestCheckpointSaver(
            save_dir=kwargs.get("bestCheckpointPath", "../results/checkpoints/best"),
            num_to_keep=1,
            maximize=True
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=kwargs.get("checkpointPath", "../results/checkpoints"))
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
        else:
            sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        train_feeder = LSTMCNNCRFeeder(train_x, train_chars, train_la, max_seq_length, max_word_length, kwargs.get("epochWidth", 16))
        val_feeder = LSTMCNNCRFeeder(val_x, val_chars, val_la, max_seq_length, max_word_length, kwargs.get("epochWidth", 16))

        for epoch in range(start_epoch, max_epoch + 1):
            loss = 0
            for step in range(train_feeder.step_per_epoch):
                tokens, chars, labels = train_feeder.feed()

                step_loss = model.train_step(sess, tokens, chars, labels)
                loss += step_loss

                self.log.debug('epoch: %d, size: %d/%d, step_loss: %f, epoch_loss: %f',
                             epoch, train_feeder.offset, train_feeder.size, step_loss, loss)

            preds = []
            for step in range(val_feeder.step_per_epoch):
                tokens, chars, labels = val_feeder.feed()
                pred = model.test(sess, tokens, chars)
                preds.extend(pred)
            true_seqs = [idx2la[la] for sl in val_la for la in sl]
            pred_seqs = [idx2la[la] for sl in preds for la in sl]
            ll = min(len(true_seqs), len(pred_seqs))

            self.log.debug(true_seqs[:ll])
            self.log.debug(pred_seqs[:ll])

            prec, rec, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

            self.log.debug("Epoch: %d, val_p: %f, val_r: %f, val_f1: %f", epoch, prec, rec, f1)

            val_feeder.next_epoch(False)

            saver.save(sess, kwargs.get("checkpointPath", "../results/checkpoints") + '/model.ckpt', global_step=epoch)
            best_saver.handle(f1, sess, epoch)

            logging.info('')
            train_feeder.next_epoch()

        self.log.debug("Training done! ... Saving trained model")
        return model, sess, saver

    def predict(self, data, *args, **kwargs):
        self.log.debug("Invoked predict method")
        self.log.debug("With parameters : ")
        self.log.debug(data)
        self.log.debug(args)
        self.log.debug(kwargs)

        pred_tuple = list()
        ret_pred_tuple = list()

        with open(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl'), 'rb') as fp:
            _, _, _, dicts = pickle.load(fp)
        w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']
        idx2la = {la2idx[k]: k for k in la2idx}

        sess = kwargs.get("sess", "")
        model = kwargs.get("model", "")
        saver = kwargs.get("saver", "")

        best_saved_cp = best_checkpoint(kwargs.get("bestCheckpointPath", "../results/checkpoints/best"), True)
        saver.restore(sess, best_saved_cp)

        # Load the model and TF session
        feeder = self.load_model(None, predictData=data, loadForPredict=True, **kwargs)

        # Fetching the predictions
        for _ in tqdm(range(feeder.step_per_epoch)):
            tokens, chars, labels = feeder.feed()

            out = model.decode(sess, tokens, chars, 1)
            for i, preds in enumerate(out):
                length = len(preds[0])

                st = tokens[i, :length].tolist()
                sl = [idx2la[la] for la in labels[i, :length].tolist()]

                preds = [[idx2la[la] for la in pred] for pred in preds]

                for zipped_res in zip(*[st, sl, *preds]):
                    pred_tuple.append([zipped_res[0], zipped_res[1], zipped_res[2]])
                    ret_pred_tuple.append([None, None, zipped_res[0], zipped_res[2]])

                pred_tuple.append([None, None, None])

        self.log.debug("Returning predictions :")
        self.log.debug(pred_tuple)

        if kwargs.get("writePredsToFile", True):
            with Path(kwargs.get("predsPath", '../results/predictions.txt')).open(mode='w') as f:
                f.write("WORD TRUE_LABEL PRED_LABEL\n\n")
                for x in range(len(pred_tuple)):
                    if pred_tuple[x][0] is None or pred_tuple[x][1] is None or pred_tuple[x][2] is None:
                        f.write("\n")
                    else:
                        f.write(pred_tuple[x][0] + " " + pred_tuple[x][1] + " " + pred_tuple[x][2] + "\n")

        return ret_pred_tuple

    def evaluate(self, predictions, groundTruths, *args, **kwargs):
        self.log.debug("Invoked evaluate method")
        self.log.debug("With parameters : ")
        self.log.debug(predictions)
        self.log.debug(groundTruths)
        self.log.debug(args)
        self.log.debug(kwargs)

        true_vals = list()
        pred_vals = list()

        if predictions is None and groundTruths is None:
            with open(kwargs.get("predsPath", '../results/predictions.txt'), mode='r', encoding='utf-8') as f:
                raw_preds = f.read().splitlines()

            for x in range(len(raw_preds)):
                true_vals.append(raw_preds[x].split(" ")[1])
                pred_vals.append(raw_preds[x].split(" ")[2])

        else:
            true_vals = groundTruths
            pred_vals = predictions

        eval_metrics = evaluate(true_vals, pred_vals, False)

        self.log.debug("Returning evaluation metrics [P, R, F1] :")
        self.log.debug(eval_metrics)

        return eval_metrics

    def save_model(self, file=None, **kwargs):
        pass

    def load_model(self, file=None, *args, **kwargs):
        self.log.debug("Invoked load_model method")
        self.log.debug("With parameters : ")
        self.log.debug(file)
        self.log.debug(args)
        self.log.debug(kwargs)

        with open(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl'), 'rb') as fp:
            train_set, val_set, test_set, dicts = pickle.load(fp)

        if kwargs.get("loadForPredict", False):
            raw_data = {}
            with open(kwargs.get("predictData", None), mode='r', encoding='utf-8') as f:
                if kwargs.get("fileHasHeaders", True):
                    next(f)
                    next(f)
                file_data = f.read().splitlines()
            for i, line in enumerate(file_data):
                if len(line.strip()) > 0:
                    file_data[i] = line.strip().split()
                else:
                    file_data[i] = list(line)

            file_data.append(list(""))

            raw_data['test'] = file_data
            raw_data['train'] = kwargs.get("trainedData", "")

            test_x, test_chars, test_la = self.data_converter(raw_data, None, fetchPredictData=True, **kwargs)
        else:
            test_x, test_chars, test_la = test_set

        train_x, train_chars, train_la = train_set
        val_x, val_chars, val_la = val_set

        max_seq_length = max(
            max(map(len, train_x)),
            max(map(len, test_x)),
        )
        max_word_length = max(
            max([len(ssc) for sc in train_chars for ssc in sc]),
            max([len(ssc) for sc in test_chars for ssc in sc])
        )

        test_feeder = LSTMCNNCRFeeder(test_x, test_chars, test_la, max_seq_length, max_word_length, kwargs.get("epochWidth", 16))

        if kwargs.get("loadForPredict", False):
            return test_feeder
        else:
            return None

    def data_converter(self, data, *args, **kwargs):
        self.log.debug("Invoked data_converter method")
        self.log.debug("With parameters : ")
        self.log.debug(data)
        self.log.debug(args)
        self.log.debug(kwargs)

        word_set = set()
        char_set = set()
        label_set = set()
        vocab = set()
        net_dump = []

        max_word_len = kwargs.get("maxWordLength", 30)

        # Iterate over each file in dictionary
        for file_type in data.keys():
            file_sentence = []
            file_chars = []
            file_labels = []

            line_sentence = []
            line_chars = []
            line_labels = []
            for i, line in enumerate(data[file_type]):
                # Check if end of sentence
                if len(line) == 0 or i == len(data[file_type]):
                    file_sentence.append(line_sentence)
                    file_chars.append(line_chars)
                    file_labels.append(line_labels)

                    line_sentence = []
                    line_chars = []
                    line_labels = []
                else:
                    word = line[kwargs.get("wordPosition", 0)]
                    chars = [ch for ch in word]
                    label = line[kwargs.get("tagPosition", 3)]

                    if len(chars) > max_word_len:
                        chars = chars[:max_word_len]

                    line_sentence.append(word)
                    line_chars.append(chars)
                    line_labels.append(label)

                    # Should only update word in train set
                    if file_type == 'train':
                        word_set.add(word.lower())
                        char_set.update(*chars)
                        label_set.add(label)

                    vocab.add(word)

            net_dump.append([file_sentence, file_chars, file_labels])

        if kwargs.get("fetchPredictData", False):
            labels2idx = {}
            for idx, label in enumerate(sorted(label_set)):
                labels2idx[label] = idx

            net_dump[0][2] = [np.array([labels2idx[la] for la in sl]) for sl in net_dump[0][2]]  # label

            return net_dump[0]

        words2idx = {}
        chars2idx = {}
        labels2idx = {}

        with Path(kwargs.get("trainWordsPath", '../dev/train.word.vocab')).open(mode='w') as f:
            for idx, word in enumerate(sorted(word_set)):
                words2idx[word] = idx
                f.write(word + '\n')

        with Path(kwargs.get("vocabPath", '../dev/vocab.txt')).open(mode='w', encoding='gb18030') as f:
            vocab = sorted(vocab)
            vocab.insert(0, '<S>')
            vocab.insert(1, '</S>')
            vocab.insert(2, '<UNK>')
            for word in vocab:
                f.write(word + '\n')

        with Path(kwargs.get("trainCharPath", '../dev/train.char.vocab')).open(mode='w') as f:
            for idx, char in enumerate(sorted(char_set)):
                chars2idx[char] = idx
                f.write(char + '\n')

        for idx, label in enumerate(sorted(label_set)):
            labels2idx[label] = idx

        for i in range(len(data.keys())):
            net_dump[i][2] = [np.array([labels2idx[la] for la in sl]) for sl in net_dump[i][2]]  # label

        if len(data.keys()) == 3:
            with Path(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl')).open(mode='wb') as f:
                pickle.dump((net_dump[0], net_dump[1], net_dump[2],
                             {
                                 'words2idx': words2idx,
                                 'chars2idx': chars2idx,
                                 'labels2idx': labels2idx
                             }), f)
        elif len(data.keys()) == 2:
            with Path(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl')).open(mode='wb') as f:
                pickle.dump((net_dump[0], net_dump[1],
                             {
                                 'words2idx': words2idx,
                                 'chars2idx': chars2idx,
                                 'labels2idx': labels2idx
                             }), f)
        elif len(data.keys()) == 1:
            with Path(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl')).open(mode='wb') as f:
                pickle.dump((net_dump[0],
                             {
                                 'words2idx': words2idx,
                                 'chars2idx': chars2idx,
                                 'labels2idx': labels2idx
                             }), f)

        self.log.debug("Data conversion done!!")

    def main(self, input_file, **kwargs):

        file_dict = dict()
        file_dict['train'] = input_file
        file_dict['test'] = input_file
        file_dict['dev'] = input_file

        data = self.read_dataset(file_dict, "CoNLL2003", None, **kwargs)
        groundTruth = self.convert_ground_truth(data, None, **kwargs)
        model, sess, saver = self.train(data, None, maxEpoch=1, **kwargs)
        predictions = self.predict(input_file, None, writeInputToFile=False, model=model, sess=sess, saver=saver, trainedData=data['train'], **kwargs)
        self.evaluate([col[3] for col in predictions], [col[3] for col in groundTruth], None, **kwargs)

        return "../results/predictions.txt"