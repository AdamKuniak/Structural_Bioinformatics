import numpy as np

class Nussimov():
    """
    A class that performs Nussimov algorithm for determining RNA structure
    """
    def __init__(self, sequence, L):
        """
        :param sequence: RNA sequence
        :param L: The minimal length of a hairpin
        """
        self.sequence = sequence
        self.L = L

    def is_pairable(self, nuc_i, nuc_j):
        """
        Function to determine whether two given nucleotides can form a valid pair of nucleotides based on Watson-Crick pairing and Wobble pairing
        :param nuc_i: first nucleotide
        :param nuc_j: second nucleotide
        :return: True if nucleotides form a valid pair of nucleotides
        """
        if nuc_i == "A" and nuc_j == "U" or nuc_i == "U" and nuc_j == "A":
            return True
        if nuc_i == "G" and nuc_j == "C" or nuc_i == "C" and nuc_j == "G":
            return True
        if nuc_i == "G" and nuc_j == "U" or nuc_i == "U" and nuc_j == "G":
            return True
        return False

    def nussimov_forward(self):
        """
        Function to perform the forward steps of Nussimov algorithm
        :return: Filled matrix of numbers of nucleotide pairs until given position i, j
        """
        n = len(self.sequence)
        self.matrix = np.zeros((n, n), dtype=int)

        for offset in range(self.L + 1, n):
            for i in range(n - offset):
                # print((i, i + offset), end="\n")
                j = i + offset
                left = self.matrix[i, j - 1]
                bottom = self.matrix[i + 1, j]
                best = np.maximum(left, bottom)
                if self.is_pairable(self.sequence[i], self.sequence[j]):
                    diag = self.matrix[i + 1, j - 1] + 1
                    best = np.maximum(best, diag)
                else:
                    continue

                best_bif = 0
                for k in range(i, j):
                    bif = self.matrix[i, k] + self.matrix[k + 1, j]
                    if bif > best_bif:
                        best_bif = bif

                best = np.maximum(best, best_bif)

                self.matrix[i, i + offset] = best

        return self.matrix

    def nussimov_backtrack(self):
        """
        Backtracks through the matrix to find the nucleotide pairs
        :return: The nucleotide pair
        """
        n = len(self.sequence)
        stack = [(0, n-1)]
        pairs = []

        while stack:
            i, j = stack.pop()
            if i >= j:
                continue
            elif self.matrix[i, j] == self.matrix[i, j - 1]:
                stack.append((i, j - 1))
            elif self.matrix[i, j] == self.matrix[i + 1, j]:
                stack.append((i + 1, j))
            elif self.matrix[i, j] == self.matrix[i + 1, j - 1] + 1 and j - i > self.L and self.is_pairable(self.sequence[i], self.sequence[j]):
                stack.append((i + 1, j - 1))
                pairs.append((i, j))
            else:
                for k in range(i, j):
                    if self.matrix[i, k] + self.matrix[k + 1, j] == self.matrix[i, j]:
                        stack.append((i, k))
                        stack.append((k + 1, j))
                        break

        return pairs

    def pairs_to_brackets(self, pairs):
        """
        Converts the pair of nucleotides into a brackets/dot string for visual representation of the structure
        :param pairs: Nucleotide pairs
        :return: The brackets/dot string
        """
        brackets = ["."] * len(self.sequence)
        for i, j in pairs:
            brackets[i] = "("
            brackets[j] = ")"
        bracket_string = "\n" + "".join(brackets)

        return bracket_string

def main():
    A = Nussimov("GGGAAACCCGGGAAACCC", 3)
    M = A.nussimov_forward()

    for i in range(M.shape[0]):
        print()
        for j in range(M.shape[1]):
            print(M[i, j], end=" ")

    pairs = A.nussimov_backtrack()
    bracket_string = A.pairs_to_brackets(pairs)
    print(bracket_string)

if __name__ == '__main__':
    main()