class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.context = list()
        self.context_top = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def get_context(self):
        self.get_child_context(self)
        sorted_context = sorted(self.context)
        return [sorted_context[i][1] for i in range(len(sorted_context))]

    def get_child_context(self, tree):
        if tree.num_children != 0:
            for child in tree.children:
                self.get_child_context(child)

        self.context.append([tree.idx,tree.state[1]])

    def get_context_top(self):
        self.get_child_context_top(self)
        sorted_context = sorted(self.context_top)
        return [sorted_context[i][1] for i in range(len(sorted_context))]

    def get_child_context_top(self, tree):
        if tree.num_children != 0:
            for child in tree.children:
                self.get_child_context_top(child)

        self.context_top.append([tree.idx, tree.state_top[1]])

    def add_eos(self, index):
        eos = Tree()
        eos.idx=index
        self.add_child(eos)
