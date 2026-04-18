"""
Viralmap
Version: v1.0
ADAPT (2025)
"""

# // imports
from huggingface_hub import snapshot_download
import transformers
from halo import Halo
import os


# // class
class VMAPUtils():
    def __init__(self):
        return

    @staticmethod
    def validate_fasta(file_path:str):
        """
        validate FASTA file:
            - Each header must be unique
            - No disallowed characters (#, !, $, etc.)
            - Each header must have at least one non-empty sequence line
            - Sequence lines must contain only valid amino acid characters
            - At least one valid header must be present
            - Must NOT be a DNA or RNA sequence (should be a protein sequence)

        returns: True if file is valid, otherwise raises ValueError
        raises: ValueError: if any of the validation checks fail
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")

        # // bools, initialization
        disallowed_in_header = set("#")
        valid_amino_acids = set("ARNDCQEGHILKMFPSTWYVBXZJU")  
        nucleotide_chars = set("ATGC")      # // DNA nucleotides
        nucleotide_chars_rna = set("AUGC")  # // RNA nucleotides
        headers_seen = set()
        current_header = None
        found_at_least_one_header = False
        has_sequence_for_current_header = False
        has_non_nucleotide_character = False  # // track if at least one sequence is non-nucleotide

        # // iterate
        total_lines = sum(1 for _ in open(file_path, 'r'))
        spinner = Halo(text='Validating FASTA:', spinner='star', color='white')
        spinner.start()

        with open(file_path, 'r') as f:
            for i, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\n") 
                spinner.text = f'Validating FASTA: {i+1}/{total_lines} proteins ({(i+1)/total_lines*100:.2f}%)'
                
                if line.startswith(">"):
                    if current_header and not has_sequence_for_current_header:
                        spinner.fail()
                        raise ValueError(f"Header '{current_header}' has no sequence lines following it.")

                    header_text = line[1:].strip()
                    if not header_text:
                        spinner.fail()
                        raise ValueError(f"Empty header found at line {i}")
                    if any(ch in disallowed_in_header for ch in header_text):
                        spinner.fail()
                        raise ValueError(f"Header '{header_text}' has disallowed characters at line {i}")
                    if header_text in headers_seen:
                        spinner.fail()
                        raise ValueError(f"Duplicate header '{header_text}' found at line {i}")
                    headers_seen.add(header_text)
                    current_header = header_text
                    found_at_least_one_header = True
                    has_sequence_for_current_header = False

                elif line.strip() == "":
                    continue  # // ignore blank lines
                else:
                    if not current_header:
                        spinner.fail()
                        raise ValueError(f"Sequence found before any header at line {i}")

                    has_sequence_for_current_header = True
                    sequence = line.replace(" ", "").upper()

                    # // validate each character
                    for char in sequence:
                        if char not in valid_amino_acids:
                            spinner.fail()
                            raise ValueError(f"Invalid character '{char}' in sequence for header '{current_header}' at line {i}")

                    # // check if the sequence contains at least one non-nucleotide character
                    if any(char not in nucleotide_chars and char not in nucleotide_chars_rna for char in sequence):
                        has_non_nucleotide_character = True
            if current_header and not has_sequence_for_current_header:
                spinner.fail()
                raise ValueError(f"Last header '{current_header}' has no sequence lines.")
            if not found_at_least_one_header:
                spinner.fail()
                raise ValueError("No valid FASTA headers found in file.")

            # // ensure the sequence is not purely nucleotide-based
            if not has_non_nucleotide_character:
                spinner.fail()
                raise ValueError("FASTA file appears to contain only nucleotide sequences (DNA/RNA), but protein sequences are expected.")

        # // 
        spinner.succeed(f"FASTA validated ({len(headers_seen)} proteins)")

    # // 
    @staticmethod
    def get_weights(weights_path: str | None = None):
        """
        get weights
        """
        if weights_path and os.path.isdir(weights_path):
            return weights_path

        try:
            path = snapshot_download(
                repo_id="shrishdwivedi/vmap",
                allow_patterns=["*.pt", "*.npz"],
                local_files_only=True,
            )
        except Exception:
            print("  Downloading model weights from HuggingFace...")
            path = snapshot_download(
                repo_id="shrishdwivedi/vmap",
                allow_patterns=["*.pt", "*.npz"],
            )
            print("  ✔ Model weights downloaded")

        # // pre-cache ESM-2 (tokenizer + model weights)
        transformers.logging.set_verbosity_error()
        transformers.AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        transformers.AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

        return path