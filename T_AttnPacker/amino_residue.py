import os
import sys
code_root = os.path.dirname(os.path.dirname(os.getcwd()))
if code_root not in sys.path:
    print(f"Added {code_root} to python path")
    sys.path.append(code_root)
# Faster Inference on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import torch
from protein_learning.models.inference_utils import (
    Inference,
    make_predicted_protein,
    default
)
import protein_learning.common.protein_constants as pc
import time

RESOURCE_ROOT = "./AttnPackerPlusRotPTM/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Wrapper class for inference
runner = Inference(RESOURCE_ROOT, use_design_variant = False)
runner = runner.to(DEVICE)

pdb_path = "./Test/6gtg_test.pdb"
#pdb_path = "./pdbs/T0967.pdb"
start = time.time()
prediction = runner.infer(
    pdb_path=pdb_path,
    #Boolean Tensor indicating which residues to design
    seq_mask=None,
    #Whether to format output (process into logits, seq labels, etc. or return raw node and pair output)
    format=True,
    #Chunk inference by successively packing smaller crops of size chunk_size
    #This allows packing of arbitrarily long proteins
    chunk_size = 500,
)
print(f"Ran Inference on {runner.device} in time {round(time.time()-start,2)} seconds")

predicted_protein = make_predicted_protein(model_out = prediction['model_out'], seq = prediction['seq'])
# Save predicted PDB to this location
pdb_out_path = f"./Test/{predicted_protein.name}_packed.pdb"
#pdb_out_path = f"./attnpacker_inference/examples/{predicted_protein.name}_packed.pdb"
print(f"saving pdb to {pdb_out_path}")
# write predicted protein with plddt score in b-factor column
predicted_protein.to_pdb(pdb_out_path, beta=prediction["pred_plddt"].squeeze())