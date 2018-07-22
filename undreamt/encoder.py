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

from undreamt import data
from .tree import Tree

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNNEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, bidirectional=False, layers=1, dropout=0):
        super(RNNEncoder, self).__init__()
        if bidirectional and hidden_size % 2 != 0:
            raise ValueError('The hidden dimension must be even for bidirectional encoders')
        self.directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.layers = layers
        self.hidden_size = hidden_size // self.directions
        self.special_embeddings = nn.Embedding(data.SPECIAL_SYMBOLS+1, embedding_size, padding_idx=0)
        #self.rnn = nn.GRU(embedding_size, self.hidden_size, bidirectional=bidirectional, num_layers=layers,dropout=dropout)

        self.rnn = TreeLSTM()

    def forward(self, ids, lengths, word_embeddings, hidden, trees):
        sorted_lengths = sorted(lengths, reverse=True)
        is_sorted = sorted_lengths == lengths
        is_varlen = sorted_lengths[0] != sorted_lengths[-1]
        if not is_sorted:
            true2sorted = sorted(range(len(lengths)), key=lambda x: -lengths[x])
            sorted2true = sorted(range(len(lengths)), key=lambda x: true2sorted[x])
            ids = torch.stack([ids[:, i] for i in true2sorted], dim=1)
            lengths = [lengths[i] for i in true2sorted]
        embeddings = word_embeddings(data.word_ids(ids)) + self.special_embeddings(data.special_ids(ids))
        if is_varlen:
            embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths)
        #output, hidden = self.rnn(embeddings, hidden)
        output, hidden = self.rnn(trees, embeddings)

        #if self.bidirectional:
        #    hidden = torch.stack([torch.cat((hidden[2*i], hidden[2*i+1]), dim=1) for i in range(self.layers)])
        if is_varlen:
            output = nn.utils.rnn.pad_packed_sequence(output)[0]
        if not is_sorted:
            hidden = torch.stack([hidden[:, i, :] for i in sorted2true], dim=1)
            output = torch.stack([output[:, i, :] for i in sorted2true], dim=1)
        return hidden, output

    def initial_hidden(self, batch_size):
        return Variable(torch.zeros(self.layers*self.directions, batch_size, self.hidden_size), requires_grad=False)


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0: # 叶子节点
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            #child_c, child_h = zip(*map(lambda x: x.state, tree.children)) #for x in tree.children, zip x.state
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))    # added vicky
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)    #then tree would have one more attribute:state, including both memory cell and hidden state
        return tree



# putting the whole model together
class TreeLSTM(nn.Module):

    def __init__(self):
        super(TreeLSTM, self).__init__()
        self.childsumtreelstm = ChildSumTreeLSTM(300, 300)

    def forward(self, trees, inputs):
        contexts = []
        hs = []
        for i in range(len(inputs[0,:,:])):  #batch size
            treewithstate = self.childsumtreelstm(self.read_tree(trees[i]),inputs[:,i,:])
            contexts.append(torch.stack(treewithstate.get_context()).squeeze())
            hs.append(treewithstate.state[1])

        contexts_tensor = torch.stack(contexts).squeeze().permute(1,0,2)
        hs_tensor = torch.stack(hs).permute(1,0,2)
        return contexts_tensor, hs_tensor

    def read_trees(self, trees_text):
        return [self.read_tree(line) for line in trees_text]

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent

        root.add_eos(len(parents))

        return root

