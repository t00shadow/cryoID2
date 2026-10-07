from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from cryoid_pipeline.models import PredictionBundle
from cryoid_pipeline.structure_selection import StructureSelection, select_ranked_rows
from D_BLAST_searching.blast_tools import blast_executable


PDB_SEQRES_URL = "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
BLAST_FIELDS = [
    "qseqid",
    "sseqid",
    "bitscore",
    "evalue",
    "pident",
    "length",
    "qlen",
    "slen",
    "qcovs",
]
ProgressCallback = Callable[[str], None]
PercentCallback = Callable[[int], None]
CancellationCheckpoint = Callable[[], None]


@dataclass
class PublicRnaDatabase:
    fasta_path: Path
    blast_prefix: Path
    metadata: dict[str, object]
    warnings: list[str]


@dataclass
class RnaStructureResult:
    outputs: list[Path]
    ranked_locus_count: int
    selected_locus_count: int
    structure_bearing_count: int
    selected_structure_count: int
    fit_success_count: int
    metadata: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _write_fasta(records: list[tuple[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")
    temporary.replace(path)
    return path


def parse_pdb_rna(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    name: str | None = None
    keep = False
    chunks: list[str] = []
    with opener(path, "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                if keep and name and chunks:
                    sequence = re.sub(r"[^ACGTN]", "N", "".join(chunks).upper().replace("U", "T"))
                    records.append((name, sequence))
                fields = line[1:].split(maxsplit=1)
                name = fields[0]
                description = fields[1] if len(fields) == 2 else ""
                keep = bool(
                    re.search(r"\bmol:na\b", description, re.IGNORECASE)
                    and re.search(r"\bRNA\b", description, re.IGNORECASE)
                    and not re.search(r"\bDNA\b", description, re.IGNORECASE)
                )
                chunks = []
            elif keep and line:
                chunks.append(line)
    if keep and name and chunks:
        sequence = re.sub(r"[^ACGTN]", "N", "".join(chunks).upper().replace("U", "T"))
        records.append((name, sequence))
    return records


def _executable(name: str, blast_bin_dir: Path | None = None) -> str:
    return blast_executable(name, blast_bin_dir)


def _blast_database_exists(prefix: Path) -> bool:
    return any(prefix.parent.glob(prefix.name + ".n*"))


def _blast_database_hashes(prefix: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(prefix.parent.glob(prefix.name + ".n*"))
        if path.is_file()
    }


def ensure_public_rna_database(
    project_dir: Path,
    *,
    progress: ProgressCallback,
    check_cancel: CancellationCheckpoint,
    blast_bin_dir: Path | None = None,
) -> PublicRnaDatabase:
    cache = Path(project_dir) / "cache" / "rna_structures"
    cache.mkdir(parents=True, exist_ok=True)
    metadata_path = cache / "database_metadata.json"
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        existing = {}
    active_name = str(existing.get("active_snapshot") or "")
    active_dir = cache / "snapshots" / active_name if active_name else cache / "missing"

    def snapshot_paths(root: Path) -> tuple[Path, Path, Path]:
        return root / "pdb_seqres.txt.gz", root / "pdb_rna_chains.fasta", root / "pdb_rna_chains"

    archive, fasta, blast_prefix = snapshot_paths(active_dir)
    local_valid = (
        archive.is_file()
        and archive.stat().st_size > 0
        and fasta.is_file()
        and fasta.stat().st_size > 0
        and _blast_database_exists(blast_prefix)
    )
    if local_valid:
        try:
            expected_archive = str(existing.get("archive_sha256") or "")
            expected_fasta = str(existing.get("rna_fasta_sha256") or "")
            expected_blast = existing.get("blast_database_sha256") or {}
            local_valid = bool(
                (not expected_archive or _sha256(archive) == expected_archive)
                and (not expected_fasta or _sha256(fasta) == expected_fasta)
                and (
                    not expected_blast
                    or _blast_database_hashes(blast_prefix) == expected_blast
                )
            )
        except OSError:
            local_valid = False
    warnings: list[str] = []
    remote: dict[str, str] = {}
    remote_checked = False
    try:
        check_cancel()
        response = requests.head(PDB_SEQRES_URL, allow_redirects=True, timeout=30)
        response.raise_for_status()
        remote = {
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
            "content_length": response.headers.get("Content-Length", ""),
        }
        remote_checked = True
    except (requests.RequestException, OSError) as exc:
        warnings.append(f"Could not check the public PDB snapshot: {exc}")

    changed = not local_valid
    if remote_checked and local_valid:
        previous_remote = existing.get("remote") if isinstance(existing, dict) else {}
        changed = bool(
            (remote.get("etag") and remote.get("etag") != (previous_remote or {}).get("etag"))
            or (
                remote.get("last_modified")
                and remote.get("last_modified") != (previous_remote or {}).get("last_modified")
            )
        )

    if changed:
        progress("RNA structures: downloading the current public PDB SEQRES archive.")
        staging = cache / f".staging_{os.getpid()}_{int(datetime.now().timestamp())}"
        staging.mkdir(parents=True, exist_ok=False)
        staged_archive, staged_fasta, staged_blast = snapshot_paths(staging)
        try:
            check_cancel()
            with requests.get(PDB_SEQRES_URL, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with staged_archive.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        check_cancel()
                        if chunk:
                            handle.write(chunk)
            if staged_archive.stat().st_size <= 0:
                raise RuntimeError("Downloaded PDB sequence archive is empty.")
            parsed = parse_pdb_rna(staged_archive)
            if not parsed:
                raise RuntimeError("Downloaded PDB sequence archive contains no pure RNA chains.")
            _write_fasta(parsed, staged_fasta)
            makeblastdb = _executable("makeblastdb", blast_bin_dir)
            subprocess.run(
                [
                    makeblastdb,
                    "-in",
                    str(staged_fasta),
                    "-dbtype",
                    "nucl",
                    "-out",
                    str(staged_blast),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if not _blast_database_exists(staged_blast):
                raise RuntimeError("makeblastdb did not create an RNA nucleotide database.")
            snapshot_name = _sha256(staged_archive)
            snapshots = cache / "snapshots"
            snapshots.mkdir(parents=True, exist_ok=True)
            destination = snapshots / snapshot_name
            if not destination.exists():
                staging.replace(destination)
            active_name = snapshot_name
            active_dir = destination
            archive, fasta, blast_prefix = snapshot_paths(active_dir)
            local_valid = True
        except Exception:
            if local_valid:
                warnings.append("Public PDB refresh failed; reused the last validated local snapshot.")
                remote = dict(existing.get("remote") or {})
            else:
                raise
        finally:
            # If an identical content-addressed snapshot already existed, or a
            # refresh failed, do not leave a partial staging cache behind.
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    if not (
        archive.is_file()
        and fasta.is_file()
        and _blast_database_exists(blast_prefix)
    ):
        raise FileNotFoundError("No validated public PDB RNA-chain cache is available.")
    check_cancel()
    metadata: dict[str, object] = {
        "source_url": PDB_SEQRES_URL,
        "active_snapshot": active_name,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "remote": remote if remote_checked else existing.get("remote", {}),
        "archive_sha256": _sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "rna_fasta_sha256": _sha256(fasta),
        "rna_fasta_bytes": fasta.stat().st_size,
        "blast_database_sha256": _blast_database_hashes(blast_prefix),
        "blastn_version": subprocess.run(
            [_executable("blastn", blast_bin_dir), "-version"], capture_output=True, text=True, check=False
        ).stdout.splitlines()[0],
        "warnings": warnings,
    }
    _atomic_json(metadata_path, metadata)
    return PublicRnaDatabase(fasta.resolve(), blast_prefix.resolve(), metadata, warnings)


def _read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name.upper()] = "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            elif name is not None:
                chunks.append(line)
    if name is not None:
        records[name.upper()] = "".join(chunks).upper()
    return records


def _canonical_chain(subject: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"pdb\|([^|]+)\|(.+)", subject, re.IGNORECASE)
    if match:
        pdb_id, chain_id = match.group(1).upper(), match.group(2)
        return f"{pdb_id}_{chain_id}", pdb_id, chain_id
    pdb_chain = subject.upper()
    pdb_id, _, chain_id = pdb_chain.partition("_")
    return pdb_chain, pdb_id, chain_id


def _write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def select_structure_hits(
    ranking: list[dict[str, str]],
    hits_by_query: dict[str, list[dict[str, object]]],
    selection: StructureSelection,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, object]]]:
    selected_loci = select_ranked_rows(ranking, selection)
    structured = [
        row for row in selected_loci if row["candidate_id"].upper() in hits_by_query
    ]
    selected: list[dict[str, object]] = []
    seen_sequences: set[str] = set()
    for locus in structured:
        kept = 0
        for hit in hits_by_query[locus["candidate_id"].upper()]:
            sequence = str(hit["structure_sequence"])
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            kept += 1
            selected.append(
                {
                    "original_rank": int(locus["rank"]),
                    "selection_order_for_query": kept,
                    **hit,
                }
            )
            if kept >= 3:
                break
    return selected_loci, structured, selected


def run_rna_structure_pipeline(
    *,
    project_dir: Path,
    repo_root: Path,
    run_dir: Path,
    density_map: Path,
    prediction_bundle: PredictionBundle,
    ranking_path: Path,
    structure_selection: StructureSelection,
    progress: ProgressCallback,
    progress_percent: PercentCallback,
    check_cancel: CancellationCheckpoint,
    blast_bin_dir: Path | None = None,
    keep_tmp: bool = False,
) -> RnaStructureResult:
    del repo_root  # Stage I now invokes its implementation directly from the installed package.
    output = Path(run_dir) / "tmp" / "structure_fetch"
    output.mkdir(parents=True, exist_ok=True)
    database = ensure_public_rna_database(
        output,
        progress=progress,
        check_cancel=check_cancel,
        blast_bin_dir=blast_bin_dir,
    )
    database_run_metadata = _atomic_json(output / "public_database.json", database.metadata)
    master_sequences = _read_fasta(database.fasta_path)
    _write_tsv(
        Path(run_dir) / "tmp" / "sequences_with_structures.tsv",
        [
            {"pdb_chain": pdb_chain, "sequence": sequence}
            for pdb_chain, sequence in master_sequences.items()
        ],
        ["pdb_chain", "sequence"],
    )
    with ranking_path.open(encoding="utf-8", newline="") as handle:
        ranking = list(csv.DictReader(handle, delimiter="\t"))
    selected_loci = select_ranked_rows(ranking, structure_selection)
    queries = [
        (row["candidate_id"], row["sequence"].replace("U", "T"))
        for row in selected_loci
    ]
    query_path = _write_fasta(queries, output / "ranked_loci.fasta")
    blast_path = output / "vs_public_pdb_rna.tsv"
    progress(
        f"RNA structures: searching the selected top {len(selected_loci)} of {len(ranking)} "
        "ranked loci against public PDB RNA chains."
    )
    progress_percent(64)
    command = [
        _executable("blastn", blast_bin_dir),
        "-task",
        "blastn",
        "-query",
        str(query_path),
        "-db",
        str(database.blast_prefix),
        "-evalue",
        "1e-3",
        "-max_target_seqs",
        "100",
        "-outfmt",
        "6 " + " ".join(BLAST_FIELDS),
        "-out",
        str(blast_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    check_cancel()
    chain_sequences = _read_fasta(database.fasta_path)
    hits_by_query: dict[str, list[dict[str, object]]] = {}
    with blast_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, fieldnames=BLAST_FIELDS, delimiter="\t"):
            if float(raw["evalue"]) > 1e-3 or float(raw["qcovs"]) < 30.0:
                continue
            chain_name, pdb_id, chain_id = _canonical_chain(raw["sseqid"])
            sequence = chain_sequences.get(raw["sseqid"].upper()) or chain_sequences.get(chain_name.upper())
            if sequence is None:
                continue
            hits_by_query.setdefault(raw["qseqid"].upper(), []).append(
                {
                    "query": raw["qseqid"].upper(),
                    "pdb_chain": chain_name,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "bitscore": float(raw["bitscore"]),
                    "evalue": float(raw["evalue"]),
                    "pident": float(raw["pident"]),
                    "alignment_length": int(raw["length"]),
                    "query_length": int(raw["qlen"]),
                    "structure_length": int(raw["slen"]),
                    "query_coverage": float(raw["qcovs"]),
                    "structure_sequence": sequence.replace("T", "U"),
                }
            )
    for hits in hits_by_query.values():
        hits.sort(key=lambda row: (-float(row["bitscore"]), float(row["evalue"]), str(row["pdb_chain"])))

    selected_loci, structured, selected = select_structure_hits(
        ranking, hits_by_query, structure_selection
    )
    candidate_rows = [
        {
            "original_rank": row["rank"],
            "candidate_id": row["candidate_id"],
            "qualifying_hit_count": len(hits_by_query.get(row["candidate_id"].upper(), [])),
            "has_qualifying_structure": row["candidate_id"].upper() in hits_by_query,
        }
        for row in selected_loci
    ]
    candidates_path = _write_tsv(
        output / "structure_candidates.tsv",
        candidate_rows,
        ["original_rank", "candidate_id", "qualifying_hit_count", "has_qualifying_structure"],
    )
    hit_fields = [
        "query",
        "original_rank",
        "selection_order_for_query",
        "pdb_chain",
        "pdb_id",
        "chain_id",
        "bitscore",
        "evalue",
        "pident",
        "alignment_length",
        "query_length",
        "structure_length",
        "query_coverage",
        "structure_sequence",
    ]
    selected_path = _write_tsv(output / "selected_structure_hits.tsv", selected, hit_fields)
    summary: dict[str, object] = {
        "structure_selection": structure_selection.details(len(ranking)),
        "ranked_loci": len(ranking),
        "selected_loci": len(selected_loci),
        "structure_bearing_loci": len(structured),
        "selected_unique_structure_chains": len(selected),
        "sequence_ranking_preserved": True,
        "database": database.metadata,
    }
    summary_path = _atomic_json(output / "selection_summary.json", summary)
    outputs = [database_run_metadata, blast_path, candidates_path, selected_path, summary_path]
    if not selected:
        progress("RNA structures: no selected ranked loci have qualifying public structures; sequence ranking is unchanged.")
        return RnaStructureResult(
            outputs, len(ranking), len(selected_loci), len(structured), 0, 0, summary
        )

    progress_percent(75)
    structures = output / "downloaded_models"
    structures.mkdir(parents=True, exist_ok=True)
    map_digits = "".join(re.findall(r"\d+", density_map.stem)) or "unknown"
    from .rawfield import fit_rna_structures

    fit_rows = fit_rna_structures(
        hits_path=selected_path,
        structures_dir=structures,
        density_map=density_map,
        phosphate_map=prediction_bundle.key_atom_path(2),
        scaffold_map=prediction_bundle.secondary_structure_path(4),
        output_dir=output,
        emdb_number=map_digits,
        progress=progress,
        check_cancel=check_cancel,
        keep_tmp=keep_tmp,
    )
    progress_percent(97)
    fit_success_count = sum(row.get("status") == "ok" for row in fit_rows)
    outputs.extend(path for path in output.rglob("*") if path.is_file() and path not in outputs)
    summary["fit_success_count"] = fit_success_count
    _atomic_json(summary_path, summary)
    return RnaStructureResult(
        outputs,
        len(ranking),
        len(selected_loci),
        len(structured),
        len(selected),
        fit_success_count,
        summary,
    )
