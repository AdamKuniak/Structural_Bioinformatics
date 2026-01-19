import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import matplotlib.pyplot as plt
from Bio.PDB import PDBList, PDBParser, PPBuilder

class BreakChecker():
    """
    A class to check if a given protein has breaks in it chains
    """
    def __init__(self, protein_id: str):
        self.id = protein_id
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(self.id, pdir=".", file_format='pdb')
        parser = PDBParser()
        self.structure = parser.get_structure(self.id, f"pdb{self.id}.ent")

        if self.has_chain_brakes():
            print(f"Protein with PDB ID {self.id} has chain brakes")
        else:
            print(f"Protein with PDB ID {self.id} doesn't have chain brakes")
    def has_chain_brakes(self):
        """
        Function that takes a protein and returns whether it has chain brakes or not
        :param protein: PDB id of the protein to check for chain brakes
        :return: Boolean of whether the protein has chain brakes or not
        """
        ppb = PPBuilder()
        has_breaks = False

        for model in self.structure:
            for chain in model:
                peptides = ppb.build_peptides(chain)  # list of all peptides in a chain
                # if there is more than one peptide in a chain
                if len(peptides) > 1:
                    has_breaks = True
                    break

        return has_breaks

class ProteinMutator():
    def __init__(self, protein_id):
        self.id = protein_id
        self.model_name = "facebook/esm2_t33_650M_UR50D"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        pdbl = PDBList()
        pdbl.retrieve_pdb_file(self.id, pdir=".", file_format='pdb')
        parser = PDBParser()
        self.structure = parser.get_structure(self.id, f"pdb{self.id}.ent")
        ppb = PPBuilder()

        pp = ppb.build_peptides(self.structure)
        # I know there are no breaks and only one chain
        self.sequence = str(pp[0].get_sequence())

    def get_mutation_probs(self):
        """
        Function that for every position in protein sequence except the first predicts the most probable amino acid for the position that is different from the original amino acid
        :return: list tuples of floats representing the most probable amino acid and the amino acid
        """
        self.model.eval()
        mask_token = self.tokenizer.mask_token  # token to mask amino acids
        mutation_probabilities = []  # list of highest mutational probabilities
        mutation_options = []  # list of most probable mutations
        topk = 2

        # For every position except the first, I am predicting the probabilities of all amino acids based on the rest of the sequence
        for i in range(0, len(self.sequence)):
            seq_list = list(self.sequence)
            seq_list[i] = mask_token
            masked_sequence = "".join(seq_list)

            # Tokenize
            tokenized_seq = self.tokenizer(masked_sequence, return_tensors="pt")

            with torch.no_grad():
                logits = self.model(**tokenized_seq).logits[0, i + 1, :]  # logits for only the masked position, +1 because of the <cls> token

            probabilities = torch.softmax(logits, dim=0)
            assert probabilities.shape == (33,)
            top_2_values, top_2_indices = torch.topk(probabilities, topk)

            # Decode
            amino_acids = [self.tokenizer.decode(idx) for idx in top_2_indices]
            original_aa = self.sequence[i]

            # if the most probable AA is the one in the sequence, I am interested in the second best, otherwise I take the best
            if original_aa == amino_acids[0]:
                mutation_options.append(amino_acids[1])
                mutation_probabilities.append(top_2_values[1].item())
            else:
                mutation_options.append(amino_acids[0])
                mutation_probabilities.append(top_2_values[0].item())

        return mutation_probabilities, mutation_options

    def mutate(self, mutation_probabilities, mutation_options, mode="deterministic"):
        """
        Mutates the protein sequence based on the obtained probabilities
        :param mode: deterministic or random; deterministic picks the highest probability, random picks the randomly
        :return: list of mutated sequences of different mutational probability
        """
        # Set specific seed
        np.random.seed(42)
        mutated_sequences = []
        mutation_probs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]

        for prob in mutation_probs:
            n_mutations = int(prob * len(mutation_options))  # number of mutations I will perform
            if mode == "deterministic":
                mut_positions = np.argpartition(mutation_probabilities, -n_mutations)[-n_mutations:]  # get the highest probability positions
            elif mode == "random":
                mut_positions = np.random.choice(len(mutation_options), n_mutations, replace=False)
            else:
                raise ValueError("Mode must be either deterministic or random")
            mutated_seq_list = list(self.sequence)

            # Mutate
            for position in mut_positions:
                mutated_seq_list[position] = mutation_options[position]

            mutated_sequences.append("".join(mutated_seq_list))
            with open(f"mutated_sequences/mutated_sequence_{prob}.txt", "w") as f:
                f.write("".join(mutated_seq_list))

        return mutated_sequences



def main():
    protein_id = "2igd"
    #BreakChecker(protein_id)
    Mutator = ProteinMutator(protein_id)
    probs, options = Mutator.get_mutation_probs()
    mutated_sequences = Mutator.mutate(probs, options, "deterministic")
    print(mutated_sequences)


if __name__ == '__main__':
    main()