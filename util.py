import random
import math

"""
Get the id and content from a line with the form [prefix]-[ID]\t[CONTENT] or form [prefix]-[ID]\t[prob]\t[CONTENT]

line: the line to parse
prefix: the one character prefix before the dash
curr_id: the current id. if the empty string, then no checks will be performed. otherwise, raise an error if the
id from the line does not match the curr_id
has_prob: boolean flag for whether or not the line has a negative log base 2 probability. False by default


Example: get_id_and_content("S-123 This is a line", "S", -1) would return
("123", "This is a line")
"""
def get_id_and_content(line, prefix, curr_id, has_prob=False):
    line_split = line.split("\t")
    line_prefix = line[0]
    if line_prefix != prefix:
        raise AssertionError("Expected prefix: " + prefix + " found prefix: " + line_prefix)

    line_id = line_split[0][2:]
    if curr_id != "" and curr_id != line_id:
        raise AssertionError("Expected id: " + curr_id + " found id: " + line_id)
    content_index = 1
    if has_prob:
        content_index = 2
    content = line_split[content_index]
    if has_prob:
        return (line_id, line_split[1], content)
    return (line_id, content)

"""
Class for holding the results from fairseq generate
"""
class Results:
    """
    Initialize a new Results object by passing the filepath to an output from fairseq generate

    Contains a list of ids, a dictionary mapping each id to its corresponding (S, T, H, D, P) tuple
    and a dictionary mapping each id to its negative log base 2 probability
    S: the source sentence
    T: the target translation
    H: the hypothesis
    D: the detokenized hypothesis
    P: the "probability" for each token (NLL for the baseline model)
    """
    def __init__(self, filepath):
        self.ids = [] # list of ids
        self.results = {} # dictionary mapping an id to a tuple (S, T, H, D, P)
        self.probs = {} # dictionary mapping an id to its corresponding negative log base 2 probability

        parsing_translation = False # determine if we are currently parsing a translation or the header/footer
        curr_id = -1 # current id we're looking at
        f = open(filepath, "r")

        # use readline() because readlines() will be very slow with large files
        line = f.readline()
        while line:
            if line.startswith("S-"):
                (S_id, S) = get_id_and_content(line, "S", "")
                line = f.readline()
                (_, T) = get_id_and_content(line, "T", S_id)
                line = f.readline()
                (_, prob, H) = get_id_and_content(line, "H", S_id, True)
                line = f.readline()
                (_, _, D) = get_id_and_content(line, "D", S_id, True)
                line = f.readline()
                (_, P) = get_id_and_content(line, "P", S_id)
                self.ids.append(S_id)
                self.results[S_id] = (S, T, H, D, P)
                self.probs[S_id] = 2 ** float(prob)
            else:
                line = f.readline()
        f.close()

    """
    Get the number of results
    """
    def size(self):
        return len(self.ids)

    """
    Write n random T, H pairs to the specified output file
    
    n should be less than the number of results
    """
    def write_n_random(self, n, out):
        random.shuffle(self.ids)
        f = open(out, "w")
        for i in range(n):
            n_id = self.ids[i]
            (S, T, H, D, P) = self.results[n_id]
            f.write(str(i) + " " + T)
            f.write(str(i) + " " + H)
        f.close()

    # write all the targets to a target output file
    def write_T(self, out):
        f = open(out, "w")
        for i in range(self.size()):
            (_, T, _, _, _) = self.results[str(i)]
            f.write(T)
        f.close()

    # write all the hypotheses to a target output file
    def write_H(self, out):
        f = open(out, "w")
        for i in range(self.size()):
            (_, _, H, _, _) = self.results[str(i)]
            f.write(H)
        f.close()

def avg_analysis(path):
    f = open(path, "r")
    lines = f.readlines()
    count = 0
    max_prob = 0
    top_10_prob = 0
    entropy = 0
    for line in lines:
        split = line.strip().split(" ")
        max_prob += float(split[0])
        top_10_prob += float(split[1])
        entropy += float(split[2])
        count += 1
    print(max_prob / count, top_10_prob / count, entropy / count)