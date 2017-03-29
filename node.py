# Node class represent every key in the dictionary. It contains a String 
# representing the token, a number representing the frequency, and a pointer
# to the posting in disk

class Node:
    def __init__(self, term, doc_frequency, pointer, size):
        self.term = term
        self.doc_frequency = doc_frequency
        self.pointer = pointer
        self.length = size

    def get_term(self):
        return self.term
    
    def get_doc_frequency(self):
        return self.doc_frequency

    def get_pointer(self):
        return self.pointer

    def set_pointer(self, pointer):
        self.pointer = pointer

    def set_length(self, length):
        self.length = length
    
    def print_node(self):
        print (self.term, self.doc_frequency, self.pointer, self.length)
