class DSU:
    '''
    An efficient Disjoint Set Union (DSU) data structure.
    It is mainly used to track each Connected Component (CC)
    and their size, but main job is to resolve merge equivalences
    '''
    def __init__(self):
        '''
        Initialize the DSU
        `parent` maps a label to its parent label
        `size` stores the size of the CCs
        '''
        self.parent={}
        self.size={}
    
    def make_set(self, v, pixel_count= 1):
        '''
        **Inputs**
        ----
        - `v`: is an `int` which is the label for a CC
        - `pixel_count`: the no.of pixels in `v`

        **Outputs**
        ----
        None

        **Description**
        ----
        Registers a new CC in the DSU, if not already present
        '''
        if self.parent.get(v) is None:
            self.parent[v]= v
            self.size[v]= pixel_count
        
    def find_set(self, v):
        '''
        **Inputs**
        ----
        - `v`: is an `int` which is the label for a CC

        **Outputs**
        ----
        - `p`: is an `int` denoting the root label of `v`

        **Description**
        ----
        finds the parent label for `v`
        '''
        if self.parent.get(v) is None:
            self.make_set(v)
            return v
        
        ## path compression optimization (2 pass)
        root= v
        while self.parent[root]!=root:
            root= self.parent[root]

        ## now that root found,
        current_node= v
        while current_node != root:
            p= self.parent[current_node]
            self.parent[current_node]= root
            current_node= p
        
        return current_node
        

    def union_set(self, a, b):
        '''
        **Inputs**
        ----
        - `a`: is an `int` which is the label for a CC
        - `b`: another CC

        **Outputs**
        ----
        None

        **Description**
        ----
        Merges two CCs `a` and `b`
        '''
        root_a= self.find_set(a)
        root_b= self.find_set(b)

        ## if not already the same set,
        if root_a != root_b:
            ## attach the smaller one to the root of larger one
            if self.size[root_a] < self.size[root_b]:
                root_a, root_b= root_b, root_a
            
            ## merge smaller into larger
            self.parent[root_b]= root_a
            self.size[root_a]+= self.size[root_b]

            ## keep map clean, delete the size of root_b
            del self.size[root_b]
        
    def get_largest_component(self):
        '''
        **Inputs**
        ----
        None

        **Outputs**
        ----
        - `a`: largest component in the DSU
        - `sz`: size of largest component

        **Description**
        ----
        Gets the largest component and its size
        '''
        if not self.size:
            return 0, 0

        ## find key with largest size
        largest_label= max(self.size, key=self.size.get)
        max_size= self.size[largest_label]
        return largest_label, max_size

    

