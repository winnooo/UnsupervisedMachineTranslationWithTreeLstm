# Copyright (C) 2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from undreamt import devices
from undreamt.encoder import RNNEncoder
from undreamt.decoder import RNNAttentionDecoder
from undreamt.generator import *
from undreamt.translator import Translator
from stanfordcorenlp import StanfordCoreNLP

import json
import argparse
import numpy as np
import sys
import time
import ast


def main_train():
    # Build argument parser
    parser = argparse.ArgumentParser(description='Train a neural machine translation model')

    # Training corpus
    corpora_group = parser.add_argument_group('training corpora', 'Corpora related arguments; specify either monolingual or parallel training corpora (or both)')
    corpora_group.add_argument('--src', help='the source language monolingual corpus')
    corpora_group.add_argument('--trg', help='the target language monolingual corpus')
    corpora_group.add_argument('--src2trg', metavar=('SRC', 'TRG'), nargs=2, help='the source-to-target parallel corpus')
    corpora_group.add_argument('--trg2src', metavar=('TRG', 'SRC'), nargs=2, help='the target-to-source parallel corpus')
    corpora_group.add_argument('--max_sentence_length', type=int, default=50, help='the maximum sentence length for training (defaults to 50)')
    corpora_group.add_argument('--cache', type=int, default=1000000, help='the cache size (in sentences) for corpus reading (defaults to 1000000)')
    corpora_group.add_argument('--cache_parallel', type=int, default=None, help='the cache size (in sentences) for parallel corpus reading')

    # Tree data
    corpora_group = parser.add_argument_group('trees', 'syntax tree data')
    corpora_group.add_argument('--src_tree', help='the syntax tree data of source language')
    corpora_group.add_argument('--trg_tree', help='the syntax tree data of target language')
    corpora_group.add_argument('--src_lang', help='the source language')
    corpora_group.add_argument('--trg_lang', help='the target language')
    corpora_group.add_argument('--core_nlp', help='path of stanford core nlp')
    corpora_group.add_argument('--backtranslation_one_line_tree', type=ast.literal_eval, default=False, help='whether to use one line tree when backtranslation')

    # Embeddings/vocabulary
    embedding_group = parser.add_argument_group('embeddings', 'Embedding related arguments; either give pre-trained cross-lingual embeddings, or a vocabulary and embedding dimensionality to randomly initialize them')
    embedding_group.add_argument('--src_embeddings', help='the source language word embeddings')
    embedding_group.add_argument('--trg_embeddings', help='the target language word embeddings')
    embedding_group.add_argument('--src_vocabulary', help='the source language vocabulary')
    embedding_group.add_argument('--trg_vocabulary', help='the target language vocabulary')
    embedding_group.add_argument('--embedding_size', type=int, default=0, help='the word embedding size')
    embedding_group.add_argument('--cutoff', type=int, default=0, help='cutoff vocabulary to the given size')
    embedding_group.add_argument('--learn_encoder_embeddings', action='store_true', help='learn the encoder embeddings instead of using the pre-trained ones')
    embedding_group.add_argument('--fixed_decoder_embeddings', action='store_true', help='use fixed embeddings in the decoder instead of learning them from scratch')
    embedding_group.add_argument('--fixed_generator', action='store_true', help='use fixed embeddings in the output softmax instead of learning it from scratch')

    # Architecture
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--layers', type=int, default=2, help='the number of encoder/decoder layers (defaults to 2)')
    architecture_group.add_argument('--hidden', type=int, default=600, help='the number of dimensions for the hidden layer (defaults to 600)')
    architecture_group.add_argument('--disable_bidirectional', action='store_true', help='use a single direction encoder')
    architecture_group.add_argument('--disable_denoising', action='store_true', help='disable random swaps')
    architecture_group.add_argument('--disable_backtranslation', action='store_true', help='disable backtranslation')

    # Optimization
    optimization_group = parser.add_argument_group('optimization', 'Optimization related arguments')
    optimization_group.add_argument('--batch', type=int, default=50, help='the batch size (defaults to 50)')
    optimization_group.add_argument('--learning_rate', type=float, default=0.0002, help='the global learning rate (defaults to 0.0002)')
    optimization_group.add_argument('--dropout', metavar='PROB', type=float, default=0.3, help='dropout probability for the encoder/decoder (defaults to 0.3)')
    optimization_group.add_argument('--param_init', metavar='RANGE', type=float, default=0.1, help='uniform initialization in the specified range (defaults to 0.1,  0 for module specific default initialization)')
    optimization_group.add_argument('--iterations', type=int, default=300000, help='the number of training iterations (defaults to 300000)')

    # Model saving
    saving_group = parser.add_argument_group('model saving', 'Arguments for saving the trained model')
    saving_group.add_argument('--save', metavar='PREFIX', help='save models with the given prefix')
    saving_group.add_argument('--save_interval', type=int, default=0, help='save intermediate models at this interval')

    # Logging/validation
    logging_group = parser.add_argument_group('logging', 'Logging and validation arguments')
    logging_group.add_argument('--log_interval', type=int, default=1000, help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--validation', nargs='+', default=(), help='use parallel corpora for validation')
    logging_group.add_argument('--validation_directions', nargs='+', default=['src2src', 'trg2trg', 'src2trg', 'trg2src'], help='validation directions')
    logging_group.add_argument('--validation_output', metavar='PREFIX', help='output validation translations with the given prefix')
    logging_group.add_argument('--validation_beam_size', type=int, default=0, help='use beam search for validation')

    # Other
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if args.src_embeddings is None and args.src_vocabulary is None or args.trg_embeddings is None and args.trg_vocabulary is None:
        print('Either an embedding or a vocabulary file must be provided')
        sys.exit(-1)
    if (args.src_embeddings is None or args.trg_embeddings is None) and (not args.learn_encoder_embeddings or args.fixed_decoder_embeddings or args.fixed_generator):
        print('Either provide pre-trained word embeddings or set to learn the encoder/decoder embeddings and generator')
        sys.exit(-1)
    if args.src_embeddings is None and args.trg_embeddings is None and args.embedding_size == 0:
        print('Either provide pre-trained word embeddings or the embedding size')
        sys.exit(-1)
    if len(args.validation) % 2 != 0:
        print('--validation should have an even number of arguments (one pair for each validation set)')
        sys.exit(-1)

    # Select device
    device = devices.gpu if args.cuda else devices.cpu
#    if args.cuda:
#        torch.backends.cudnn.benchmark = True

    # Create optimizer lists
    src2src_optimizers = []
    trg2trg_optimizers = []
    src2trg_optimizers = []
    trg2src_optimizers = []

    # Method to create a module optimizer and add it to the given lists
    def add_optimizer(module, directions=()):
        if args.param_init != 0.0:
            for param in module.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        optimizer = torch.optim.Adam(module.parameters(), lr=args.learning_rate)
        for direction in directions:
            direction.append(optimizer)
        return optimizer

    # Start core nlp server
    print("Starting Stanford CoreNLP server")
    core_nlp = StanfordCoreNLP(args.core_nlp, memory='8g')
    if args.backtranslation_one_line_tree == True and args.src_tree is not None and args.trg_tree is not None:
        core_nlp.close()

    # Load word embeddings
    print("Loading word embeddings")
    src_words = trg_words = src_embeddings = trg_embeddings = src_dictionary = trg_dictionary = None
    embedding_size = args.embedding_size
    if args.src_vocabulary is not None:
        f = open(args.src_vocabulary, encoding=args.encoding, errors='surrogateescape')
        src_words = [line.strip() for line in f.readlines()]
        if args.cutoff > 0:
            src_words = src_words[:args.cutoff]
        src_dictionary = data.Dictionary(src_words)
    if args.trg_vocabulary is not None:
        f = open(args.trg_vocabulary, encoding=args.encoding, errors='surrogateescape')
        trg_words = [line.strip() for line in f.readlines()]
        if args.cutoff > 0:
            trg_words = trg_words[:args.cutoff]
        trg_dictionary = data.Dictionary(trg_words)
    if args.src_embeddings is not None:
        f = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
        src_embeddings, src_dictionary = data.read_embeddings(f, args.cutoff, src_words)
        src_embeddings = device(src_embeddings)
        src_embeddings.requires_grad = False
        if embedding_size == 0:
            embedding_size = src_embeddings.weight.data.size()[1]
        if embedding_size != src_embeddings.weight.data.size()[1]:
            print('Embedding sizes do not match')
            sys.exit(-1)
    if args.trg_embeddings is not None:
        trg_file = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
        trg_embeddings, trg_dictionary = data.read_embeddings(trg_file, args.cutoff, trg_words)
        trg_embeddings = device(trg_embeddings)
        trg_embeddings.requires_grad = False
        if embedding_size == 0:
            embedding_size = trg_embeddings.weight.data.size()[1]
        if embedding_size != trg_embeddings.weight.data.size()[1]:
            print('Embedding sizes do not match')
            sys.exit(-1)
    if args.learn_encoder_embeddings:
        src_encoder_embeddings = device(data.random_embeddings(src_dictionary.size(), embedding_size))
        trg_encoder_embeddings = device(data.random_embeddings(trg_dictionary.size(), embedding_size))
        add_optimizer(src_encoder_embeddings, (src2src_optimizers, src2trg_optimizers))
        add_optimizer(trg_encoder_embeddings, (trg2trg_optimizers, trg2src_optimizers))
    else:
        src_encoder_embeddings = src_embeddings
        trg_encoder_embeddings = trg_embeddings
    if args.fixed_decoder_embeddings:
        src_decoder_embeddings = src_embeddings
        trg_decoder_embeddings = trg_embeddings
    else:
        src_decoder_embeddings = device(data.random_embeddings(src_dictionary.size(), embedding_size))
        trg_decoder_embeddings = device(data.random_embeddings(trg_dictionary.size(), embedding_size))
        add_optimizer(src_decoder_embeddings, (src2src_optimizers, trg2src_optimizers))
        add_optimizer(trg_decoder_embeddings, (trg2trg_optimizers, src2trg_optimizers))
    if args.fixed_generator:
        src_embedding_generator = device(EmbeddingGenerator(hidden_size=args.hidden, embedding_size=embedding_size))
        trg_embedding_generator = device(EmbeddingGenerator(hidden_size=args.hidden, embedding_size=embedding_size))
        add_optimizer(src_embedding_generator, (src2src_optimizers, trg2src_optimizers))
        add_optimizer(trg_embedding_generator, (trg2trg_optimizers, src2trg_optimizers))
        src_generator = device(WrappedEmbeddingGenerator(src_embedding_generator, src_embeddings))
        trg_generator = device(WrappedEmbeddingGenerator(trg_embedding_generator, trg_embeddings))
    else:
        src_generator = device(LinearGenerator(args.hidden, src_dictionary.size()))
        trg_generator = device(LinearGenerator(args.hidden, trg_dictionary.size()))
        add_optimizer(src_generator, (src2src_optimizers, trg2src_optimizers))
        add_optimizer(trg_generator, (trg2trg_optimizers, src2trg_optimizers))

    # Build encoder
    from .encoder import ChildSumTreeLSTM
    from .encoder import TreeLSTM
    from .encoder import TopDownTreeLSTM
    tree_lstm = None
    child_sum_lstm = device(ChildSumTreeLSTM(embedding_size, int(args.hidden/2)))
    if not args.disable_bidirectional:
        top_down_tree_lstm = device(TopDownTreeLSTM(embedding_size, int(args.hidden/2)))
        tree_lstm = device(TreeLSTM(sum_tree_lstm = child_sum_lstm, is_bidirectional = True, top_down_tree_lstm = top_down_tree_lstm))
    else:
        tree_lstm = device(TreeLSTM(sum_tree_lstm=child_sum_lstm, is_bidirectional=False))

    encoder = device(RNNEncoder(embedding_size=embedding_size, hidden_size=args.hidden,
                                bidirectional=not args.disable_bidirectional, layers=args.layers, tree_lstm = tree_lstm, dropout=args.dropout))
    add_optimizer(encoder, (src2src_optimizers, trg2trg_optimizers, src2trg_optimizers, trg2src_optimizers))

    # Build decoders
    src_decoder = device(RNNAttentionDecoder(embedding_size=embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
    trg_decoder = device(RNNAttentionDecoder(embedding_size=embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
    add_optimizer(src_decoder, (src2src_optimizers, trg2src_optimizers))
    add_optimizer(trg_decoder, (trg2trg_optimizers, src2trg_optimizers))

    # Build translators
    src2src_translator = Translator(encoder_embeddings=src_encoder_embeddings,
                                    decoder_embeddings=src_decoder_embeddings, generator=src_generator,
                                    src_dictionary=src_dictionary, trg_dictionary=src_dictionary, encoder=encoder,
                                    decoder=src_decoder, denoising=not args.disable_denoising, device=device)
    src2trg_translator = Translator(encoder_embeddings=src_encoder_embeddings,
                                    decoder_embeddings=trg_decoder_embeddings, generator=trg_generator,
                                    src_dictionary=src_dictionary, trg_dictionary=trg_dictionary, encoder=encoder,
                                    decoder=trg_decoder, denoising=not args.disable_denoising, device=device)
    trg2trg_translator = Translator(encoder_embeddings=trg_encoder_embeddings,
                                    decoder_embeddings=trg_decoder_embeddings, generator=trg_generator,
                                    src_dictionary=trg_dictionary, trg_dictionary=trg_dictionary, encoder=encoder,
                                    decoder=trg_decoder, denoising=not args.disable_denoising, device=device)
    trg2src_translator = Translator(encoder_embeddings=trg_encoder_embeddings,
                                    decoder_embeddings=src_decoder_embeddings, generator=src_generator,
                                    src_dictionary=trg_dictionary, trg_dictionary=src_dictionary, encoder=encoder,
                                    decoder=src_decoder, denoising=not args.disable_denoising, device=device)

    # Build trainers
    trainers = []
    src2src_trainer = trg2trg_trainer = src2trg_trainer = trg2src_trainer = None
    srcback2trg_trainer = trgback2src_trainer = None
    if args.src is not None:
        f = open(args.src, encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)

        tree_corpus = None
        if args.src_tree is not None:
            f_tree = open(args.src_tree, encoding=args.encoding, errors='surrogateescape')
            tree_corpus = data.TreeReader(f_tree, max_sentence_length=args.max_sentence_length, cache_size=args.cache)

        src2src_trainer = Trainer(translator=src2src_translator, optimizers=src2src_optimizers, corpus=corpus, tree_corpus = tree_corpus, batch_size=args.batch, src_lang = args.src_lang, trg_lang = args.trg_lang, core_nlp = core_nlp, type = 'src2src')
        trainers.append(src2src_trainer)
        if not args.disable_backtranslation:
            trgback2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=corpus, translator=src2trg_translator, tree_corpus=tree_corpus), tree_corpus = tree_corpus,batch_size=args.batch, src_lang = args.src_lang, trg_lang = args.trg_lang, core_nlp = core_nlp,type = 'trg2src')
            trainers.append(trgback2src_trainer)
    if args.trg is not None:
        f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)

        tree_corpus = None
        if args.trg_tree is not None:
            f_tree = open(args.trg_tree, encoding=args.encoding, errors='surrogateescape')
            tree_corpus = data.TreeReader(f_tree, max_sentence_length=args.max_sentence_length, cache_size=args.cache)

        trg2trg_trainer = Trainer(translator=trg2trg_translator, optimizers=trg2trg_optimizers, corpus=corpus, tree_corpus = tree_corpus, batch_size=args.batch, src_lang = args.src_lang, trg_lang = args.trg_lang, core_nlp = core_nlp,type = 'trg2trg')
        trainers.append(trg2trg_trainer)
        if not args.disable_backtranslation:
            srcback2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=corpus, translator=trg2src_translator, tree_corpus=tree_corpus),  tree_corpus = tree_corpus, batch_size=args.batch, src_lang = args.src_lang, trg_lang = args.trg_lang, core_nlp = core_nlp,type = 'src2trg')
            trainers.append(srcback2trg_trainer)
    if args.src2trg is not None:
        f1 = open(args.src2trg[0], encoding=args.encoding, errors='surrogateescape')
        f2 = open(args.src2trg[1], encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f1, f2, max_sentence_length=args.max_sentence_length, cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
        src2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers, corpus=corpus, batch_size=args.batch)
        trainers.append(src2trg_trainer)
    if args.trg2src is not None:
        f1 = open(args.trg2src[0], encoding=args.encoding, errors='surrogateescape')
        f2 = open(args.trg2src[1], encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f1, f2, max_sentence_length=args.max_sentence_length, cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
        trg2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers, corpus=corpus, batch_size=args.batch)
        trainers.append(trg2src_trainer)

    # Build validators
    src2src_validators = []
    trg2trg_validators = []
    src2trg_validators = []
    trg2src_validators = []
    for i in range(0, len(args.validation), 2):
        src_validation = open(args.validation[i],   encoding=args.encoding, errors='surrogateescape').readlines()
        trg_validation = open(args.validation[i+1], encoding=args.encoding, errors='surrogateescape').readlines()
        if len(src_validation) != len(trg_validation):
            print('Validation sizes do not match')
            sys.exit(-1)
        map(lambda x: x.strip(), src_validation)
        map(lambda x: x.strip(), trg_validation)
        if 'src2src' in args.validation_directions:
            src2src_validators.append(Validator(src2src_translator, src_validation, src_validation, args.batch, args.validation_beam_size))
        if 'trg2trg' in args.validation_directions:
            trg2trg_validators.append(Validator(trg2trg_translator, trg_validation, trg_validation, args.batch, args.validation_beam_size))
        if 'src2trg' in args.validation_directions:
            src2trg_validators.append(Validator(src2trg_translator, src_validation, trg_validation, args.batch, args.validation_beam_size))
        if 'trg2src' in args.validation_directions:
            trg2src_validators.append(Validator(trg2src_translator, trg_validation, src_validation, args.batch, args.validation_beam_size))

    # Build loggers
    loggers = []
    src2src_output = trg2trg_output = src2trg_output = trg2src_output = None
    if args.validation_output is not None:
        src2src_output = '{0}.src2src'.format(args.validation_output)
        trg2trg_output = '{0}.trg2trg'.format(args.validation_output)
        src2trg_output = '{0}.src2trg'.format(args.validation_output)
        trg2src_output = '{0}.trg2src'.format(args.validation_output)
    loggers.append(Logger('Source to target (backtranslation)', srcback2trg_trainer, [], None, args.encoding))
    loggers.append(Logger('Target to source (backtranslation)', trgback2src_trainer, [], None, args.encoding))
    loggers.append(Logger('Source to source', src2src_trainer, src2src_validators, src2src_output, args.encoding))
    loggers.append(Logger('Target to target', trg2trg_trainer, trg2trg_validators, trg2trg_output, args.encoding))
    loggers.append(Logger('Source to target', src2trg_trainer, src2trg_validators, src2trg_output, args.encoding))
    loggers.append(Logger('Target to source', trg2src_trainer, trg2src_validators, trg2src_output, args.encoding))

    # Method to save models
    def save_models(name):
        torch.save(src2src_translator, '{0}.{1}.src2src.pth'.format(args.save, name))
        torch.save(trg2trg_translator, '{0}.{1}.trg2trg.pth'.format(args.save, name))
        torch.save(src2trg_translator, '{0}.{1}.src2trg.pth'.format(args.save, name))
        torch.save(trg2src_translator, '{0}.{1}.trg2src.pth'.format(args.save, name))

    # Training
    print("Training")
    for step in range(1, args.iterations + 1):
        for trainer in trainers:
            trainer.step(args.backtranslation_one_line_tree)

        if args.save is not None and args.save_interval > 0 and step % args.save_interval == 0:
            save_models('it{0}'.format(step))

        if step % args.log_interval == 0:
            print()
            print('STEP {0} x {1}'.format(step, args.batch))
            for logger in loggers:
                logger.log(step)

        step += 1

    save_models('final')
    core_nlp.close()


class Trainer:
    def __init__(self, corpus, tree_corpus, optimizers, translator, core_nlp, src_lang, trg_lang, type = '', batch_size=50):
        self.corpus = corpus
        self.translator = translator
        self.optimizers = optimizers
        self.batch_size = batch_size
        self.reset_stats()
        self.type = type
        self.core_nlp = core_nlp
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.tree_corpus = tree_corpus


    def step(self,one_line_tree):
        # Reset gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # Read input sentences
        t = time.time()

        src = []
        trg = []
        trees = []

        if self.type == 'src2src' or self.type == 'src2trg':
            src, trg = self.corpus.next_batch(self.batch_size, self.core_nlp, self.trg_lang)
            src=['.' if line == '' else line for line in src]
            if self.type == 'src2trg' and one_line_tree:
                #t= time.time()
                trees = self.get_one_line_tree(src)
                #print("oneline="+str(time.time()-t))
            else:
                if self.tree_corpus is None:
                    #t = time.time()
                    trees = self.get_trees_text(src, self.src_lang, self.core_nlp)
                    #print("live=" + str(time.time() - t))
                else:
                    #t = time.time()
                    trees = self.tree_corpus.next_batch(self.batch_size)[0]
                    #print("readcorpus=" + str(time.time() - t))

        if self.type == 'trg2trg' or self.type == 'trg2src':
            src, trg = self.corpus.next_batch(self.batch_size, self.core_nlp, self.src_lang)
            src=['.' if line == '' else line for line in src]
            if self.type == 'trg2src' and one_line_tree:
                #t= time.time()
                trees = self.get_one_line_tree(src)
                #print("oneline="+str(time.time()-t))
            else:
                if self.tree_corpus is None:
                    #t= time.time()
                    trees = self.get_trees_text(src, self.trg_lang, self.core_nlp)
                    #print("live="+str(time.time()-t))
                else:
                    #t = time.time()
                    trees = self.tree_corpus.next_batch(self.batch_size)[0]
                    #print("readcorpus=" + str(time.time() - t))

        self.src_word_count += sum([len(data.tokenize(sentence)) + 1 for sentence in src])  # TODO Depends on special symbols EOS/SOS
        self.trg_word_count += sum([len(data.tokenize(sentence)) + 1 for sentence in trg])  # TODO Depends on special symbols EOS/SOS
        self.io_time += time.time() - t
        #print("trainer read input tree takes " + str(time.time() - t))


        # Compute loss
        #t = time.time()
        loss = self.translator.score(src, trg, trees, train=True)
        self.loss += loss.data[0]
        self.forward_time += time.time() - t
        #print("compute loss takes " + str(time.time() - t))

        # Backpropagate error + optimize
        #t = time.time()
        loss.div(self.batch_size).backward()
        for optimizer in self.optimizers:
            optimizer.step()
        self.backward_time += time.time() - t
        #print("backpropagate " + str(time.time() - t))

#    def get_one_line_tree(self, src):
#        trees = []
#        for sentence in src:
#            token_size = len(sentence.strip().split())
#            one_line_tree = ' '.join(str(i) for i in reversed(range(token_size)))
#            trees.append(one_line_tree)
#        return trees

    def get_one_line_tree(self, src):
        trees = []
        for sentence in src:
            token_size = len(sentence.strip().split())
            one_line_tree = (' '.join(str(i+2) for i in range(token_size -1)) + " 0").strip()
            trees.append(one_line_tree)
        return trees

    def get_trees_text(self, text, lang, corenlp):
        #t = time.time()
        for i in range(len(text)):
            if text[i] == '':
                text[i] = '.'
        if lang == "english":
            props = {'annotators': 'tokenize, ssplit, pos, depparse',
                     'tokenize.whitespace': 'true',
                     'tokenize.language': lang,
                     'depparse.language': lang,
                     'ssplit.eolonly': 'true',
                     'outputFormat': 'json'
                     }
            lines = '\n'.join(text)
            raw_data = corenlp.annotate(lines, properties=props)
            json_data = json.loads(raw_data)['sentences']
            trees = [self.convert_raw_tree(tree['basicDependencies']) for tree in json_data]
            #print('Geting tree takes' + str(time.time() - t))
            return trees
        if lang == "french":
            props = {'annotators': 'tokenize, ssplit, pos, depparse',
                     'tokenize.whitespace': 'true',
                     'tokenize.language': lang,
                     'depparse.language': lang,
                     'ssplit.eolonly': 'true',
                     'outputFormat': 'json',
                     'pos.model' : 'edu/stanford/nlp/models/pos-tagger/french/french-ud.tagger',
                     'depparse.model' : 'edu/stanford/nlp/models/parser/nndep/UD_French.gz'
                     }
            lines = '\n'.join(text)
            raw_data = corenlp.annotate(lines, properties=props)
            json_data = json.loads(raw_data)['sentences']
            trees = [self.convert_raw_tree(tree['basicDependencies']) for tree in json_data]
            #print('Geting tree takes' + str(time.time() - t))
            return trees


    def convert_raw_tree(self, raw_tree):
        raw_tree.sort(key=lambda x: x['dependent'])
        tree_list = [row['governor'] for row in raw_tree]
        return ' '.join(str(e) for e in tree_list)

    def reset_stats(self):
        self.src_word_count = 0
        self.trg_word_count = 0
        self.io_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.loss = 0

    def perplexity_per_word(self):
        return np.exp(self.loss/self.trg_word_count)

    def total_time(self):
        return self.io_time + self.forward_time + self.backward_time

    def words_per_second(self):
        return self.src_word_count / self.total_time(),  self.trg_word_count / self.total_time()


class Validator:
    def __init__(self, translator, source, reference, batch_size=50, beam_size=0):
        self.translator = translator
        self.source = source
        self.reference = reference
        self.sentence_count = len(source)
        self.reference_word_count = sum([len(data.tokenize(sentence)) + 1 for sentence in self.reference])  # TODO Depends on special symbols EOS/SOS
        self.batch_size = batch_size
        self.beam_size = beam_size

        # Sorting
        lengths = [len(data.tokenize(sentence)) for sentence in self.source]
        self.true2sorted = sorted(range(self.sentence_count), key=lambda x: -lengths[x])
        self.sorted2true = sorted(range(self.sentence_count), key=lambda x: self.true2sorted[x])
        self.sorted_source = [self.source[i] for i in self.true2sorted]
        self.sorted_reference = [self.reference[i] for i in self.true2sorted]

    def perplexity(self):
        loss = 0
        for i in range(0, self.sentence_count, self.batch_size):
            j = min(i + self.batch_size, self.sentence_count)
            loss += self.translator.score(self.sorted_source[i:j], self.sorted_reference[i:j], train=False).data[0]
        return np.exp(loss/self.reference_word_count)

    def translate(self):
        translations = []
        for i in range(0, self.sentence_count, self.batch_size):
            j = min(i + self.batch_size, self.sentence_count)
            batch = self.sorted_source[i:j]
            if self.beam_size <= 0:
                translations += self.translator.greedy(batch, train=False)
            else:
                translations += self.translator.beam_search(batch, train=False, beam_size=self.beam_size)
        return [translations[i] for i in self.sorted2true]


class Logger:
    def __init__(self, name, trainer, validators=(), output_prefix=None, encoding='utf-8'):
        self.name = name
        self.trainer = trainer
        self.validators = validators
        self.output_prefix = output_prefix
        self.encoding = encoding

    def log(self, step=0):
        if self.trainer is not None or len(self.validators) > 0:
            print('{0}'.format(self.name))
        if self.trainer is not None:
            print('  - Training:   {0:10.2f}   ({1:.2f}s: {2:.2f}tok/s src, {3:.2f}tok/s trg; epoch {4})'
                .format(self.trainer.perplexity_per_word(), self.trainer.total_time(),
                self.trainer.words_per_second()[0], self.trainer.words_per_second()[1], self.trainer.corpus.epoch))
            self.trainer.reset_stats()
        for id, validator in enumerate(self.validators):
            t = time.time()
            perplexity = validator.perplexity()
            print('  - Validation: {0:10.2f}   ({1:.2f}s)'.format(perplexity, time.time() - t))
            if self.output_prefix is not None:
                f = open('{0}.{1}.{2}.txt'.format(self.output_prefix, id, step), mode='w',
                         encoding=self.encoding, errors='surrogateescape')
                for line in validator.translate():
                    print(line, file=f)
                f.close()
        sys.stdout.flush()