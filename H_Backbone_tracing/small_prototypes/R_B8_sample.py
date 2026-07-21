from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Sample:
    """
    This class is specifically for my batch running script.\n
    This class ONLY STORES paths of inputs/intermediates worth looking at in chimeraX. It should handle NO file management logic
    """
    name: str

    # Unet
    unet_possibility_mrc_dir: Path
    key_atom_ca_mrc: Path
    secondary_struct_helix_mrc: Path
    # Clustering                             #! an input at this stage
    clustered_ca_pdb: Path                   # TODO: extend this to allow cif files

    # 2ND HALF (stuff that will be generated in this part of the pipeline)
    # Masking with alpha helix
    masked_ca_pdb: Path | None = None
    # Backbone tracing
    backbone_graph_dir: Path | None = None
    backbone_components_dir: Path | None = None
    # Residue assignment
    residue_sequences_dir: Path | None = None
    # Queries (Fasta file format)
    queries_fasta_file: Path | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inputs": {
                "unet_possibility_mrc_dir": str(self.unet_possibility_mrc_dir) if self.unet_possibility_mrc_dir else None,
                "key_atom_ca_mrc": str(self.key_atom_ca_mrc) if self.key_atom_ca_mrc else None,
                "secondary_struct_helix_mrc": str(self.secondary_struct_helix_mrc) if self.secondary_struct_helix_mrc else None,
                "clustered_ca_pdb": str(self.clustered_ca_pdb) if self.clustered_ca_pdb else None,
            },
            "outputs": {
                "masked_ca_pdb": str(self.masked_ca_pdb) if self.masked_ca_pdb else None,
                "backbone_graph_dir": str(self.backbone_graph_dir) if self.backbone_graph_dir else None,
                "backbone_components_dir": str(self.backbone_components_dir) if self.backbone_components_dir else None,
                "residue_sequences_dir": str(self.residue_sequences_dir) if self.residue_sequences_dir else None,
                "queries_fasta_file": str(self.queries_fasta_file) if self.queries_fasta_file else None,
            }
        }


if __name__ == "__main__":
    s = Sample(1234, Path("/data"), "", "", "")
    # print(s.to_dict())
    print(s)