from H_Backbone_tracing.R_B8_clustering_eval_driver import get_column_values
from pathlib import Path
import requests


def fetch_fasta_sequences(pdb_ids, output_path, individual_files=False, mode="a"):
    '''
        pdb_ids: pdb ids as list of str
        output_path: file or dir. Treated as a file name if individual_files is False. Treated as a dir if individual_files is True.
    '''

    output_path = Path(output_path)

    # If writing individual files, treat output_path as a directory
    if individual_files:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        with output_path.open(mode, encoding="utf-8") as f:
            f.write("")

    for pdb_id in pdb_ids:
        pdb_id = pdb_id.lower()

        url = (
            f"https://www.ebi.ac.uk/pdbe/api/v2/pdb/entry/"
            f"{pdb_id}/fasta?mode=view"
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            fasta_text = response.text.strip()

            if not fasta_text:
                print(f"✗ {pdb_id}: empty response")
                continue

            if individual_files:
                file_path = output_path / f"{pdb_id}.fasta"
                with file_path.open("w", encoding="utf-8") as f:
                    f.write(fasta_text + "\n")
            else:
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(fasta_text + "\n\n")

            print(f"✓ {pdb_id}")

        except requests.RequestException as e:
            print(f"✗ {pdb_id}: {e}")

    print(output_path)


if __name__ == "__main__":
    csv_file = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\New\ready_to_test_new.csv"

    pdb_ids = get_column_values(csv_file, "fitted_pdbs")
    pdb_ids = sorted(pdb_ids)

    # test
    # pdb_ids = [
    #     "3D12",
    #     "1CRN",
    #     "4HHB",
    # ]

    print(pdb_ids)

    current_dir = Path(__file__).resolve().parent
    # print(current_dir); assert False

    # Option 1: single combined FASTA
    fetch_fasta_sequences(
        pdb_ids,
        output_path=current_dir / "R_C8_my_search_pool.fasta",
        individual_files=False,
        mode="w",
    )

    # # Option 2: individual FASTA files
    # fetch_fasta_sequences(
    #     pdb_ids,
    #     output_path=current_dir / ".." / "misc_outputs" / "gt_fasta_files",
    #     individual_files=True,
    # )

    # could add a 3rd option (new fxn?) that just concats all the individual files into a single fasta