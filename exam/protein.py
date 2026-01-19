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
    def __init__(self, protein_id, sequence):
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
        self.sequence = str(pp[0].get_sequence())

        if sequence is not None:
            self.sequence = sequence

    def get_mutation_probs(self):
        """
        Function that for every position in protein sequence except the first predicts the most probable amino acid for the position that is different from the original amino acid
        :return: list tuples of floats representing the most probable amino acid and the amino acid
        """
        self.model.eval()
        mask_token = self.tokenizer.mask_token  # token to mask amino acids
        mutation_list = []

        # For every position except the first, I am predicting the probabilities of all amino acids based on the rest of the sequence
        for i in range(0, len(self.sequence) ):
            seq_list = list(self.sequence)
            seq_list[i] = mask_token
            masked_sequence = "".join(seq_list)

            # Tokenize
            tokenized_seq = self.tokenizer(masked_sequence, return_tensors="pt")

            with torch.no_grad():
                logits = self.model(**tokenized_seq).logits[0, i + 1, :]  # logits for only the masked position, +1 because of the <cls> token

            probabilities = torch.softmax(logits, dim=0)
            assert probabilities.shape == (33,)
            top_2_values, top_2_indices = torch.topk(probabilities, 2)

            # Decode
            amino_acids = [self.tokenizer.decode(idx) for idx in top_2_indices]
            original_aa = self.sequence[i]

            # if the most probable AA is the one in the sequence, I am interested in the second best, otherwise I take the best
            if original_aa == amino_acids[0]:
                mutation_list.append((amino_acids[1], top_2_values[1].item()))
            else:
                mutation_list.append((amino_acids[0], top_2_values[0].item()))

        return mutation_list

    #def mutate(self)



def main():
    protein_id = "2igd"
    #BreakChecker(protein_id)
    Mutator = ProteinMutator(protein_id, "MKVLELTDNDGTLTE")
    list = Mutator.get_mutation_probs()
    print(list)


if __name__ == '__main__':
    main()