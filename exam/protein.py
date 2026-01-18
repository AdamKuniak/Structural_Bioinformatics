import numpy as np
#from transformers import AutoTokenizer, AutoModelForMaskedLM
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

def main():
    protein_id = "2igd"
    BreakChecker(protein_id)


if __name__ == '__main__':
    main()