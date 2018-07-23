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

import argparse
import sys
import torch
import json
from stanfordcorenlp import StanfordCoreNLP

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate using a pre-trained model')
    parser.add_argument('model', help='a model previously trained with train.py')
    parser.add_argument('--batch_size', type=int, default=50, help='the batch size (defaults to 50)')
    parser.add_argument('--beam_size', type=int, default=12, help='the beam size (defaults to 12, 0 for greedy search)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output file (defaults to stdout)')
    parser.add_argument('--core_nlp', help='path of stanford core nlp')
    parser.add_argument('--lang', help='source language')
    args = parser.parse_args()

    # Load model
    translator = torch.load(args.model)

    # Start core nlp server
    print("Starting Stanford CoreNLP server")
    core_nlp = StanfordCoreNLP(args.core_nlp, memory='8g')

    # Translate sentences
    end = False
    fin = open(args.input, encoding=args.encoding, errors='surrogateescape')
    fout = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    while not end:
        batch = []
        while len(batch) < args.batch_size and not end:
            line = fin.readline()
            if not line:
                end = True
            else:
                batch.append(line)

        trees = get_trees_text(batch, args.lang, core_nlp)

        if args.beam_size <= 0 and len(batch) > 0:
            for translation in translator.greedy(batch, trees, train=False):
                print(translation, file=fout)
        elif len(batch) > 0:
            for translation in translator.beam_search(batch, trees, train=False, beam_size=args.beam_size):
                print(translation, file=fout)
        fout.flush()
    fin.close()
    fout.close()

def get_trees_text(text, lang, corenlp):
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
        trees = [convert_raw_tree(tree['basicDependencies']) for tree in json_data]
        #print('Geting tree takes' + str(time.time() - t))
        return trees


def convert_raw_tree(raw_tree):
        raw_tree.sort(key=lambda x: x['dependent'])
        tree_list = [row['governor'] for row in raw_tree]
        return ' '.join(str(e) for e in tree_list)

if __name__ == '__main__':
    main()
