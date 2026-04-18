"""
Viralmap
Version: v1.0
ADAPT (2025)
"""

# // imports
import os
import argparse
import sys
import torch

# // classes
from .utils import VMAPUtils
from .predict import VMAPInfer

# // main
def main():
    """
    Viralmap cli
    """
    parser = argparse.ArgumentParser(description="VIRALMAP (VMAP) ADAPT (2026) v1.0", 
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-id","--run_id",
                        type=str,
                        default=None,
                        required=True,
                        help="Unique identifier for this run (used to name output directory)")
    parser.add_argument("-i", "--input",
                        type=str,
                        default=None,
                        required=True,
                        help="Path to input FASTA file")
    parser.add_argument("-m", "--mode",
                        action="store_true",
                        help="Run predictions in sensitive mode (lower thresholds for PTM classes, weaker none-state cooling in HMMs). Default: baseline mode with sensitive thresholds for PTM classes only (recommended)")
    parser.add_argument("-w","--weights",
                        type=str,
                        default=None,
                        required=False,
                        help="Path to model weights directory. If not provided, downloads from HuggingFace.")
    parser.add_argument("-o","--output_dir",
                        type=str,
                        default=str(os.getcwd()),
                        required=False,
                        help="Path to where output directory will be created (default: pwd). The directory will be named id_vmap_out")
    args = parser.parse_args()

    # // start
    print("\nADAPT (2026) ViralMap v1.0")

    # // validate FASTA file
    utils = VMAPUtils()
    try:
        utils.validate_fasta(file_path=args.input)
    except ValueError as e:
        print(f"FASTA validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # // model weights
    model_dir = utils.get_weights(weights_path=args.weights)
    # // inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer = VMAPInfer(vmap_base=model_dir, vmap_mode=args.mode, device=device)

    try: 
        output_dir = os.path.join(args.output_dir, f"{args.run_id}_vmap_out")
        os.makedirs(output_dir, exist_ok=True)
        infer.predict(file_path=args.input, output_dir=output_dir)
        print("inference complete.\n")

    except ValueError as e:
        print(f"prediction failed: {e}", file=sys.stderr)
        sys.exit(1)

 
if __name__=="__main__":
    main() 



        



