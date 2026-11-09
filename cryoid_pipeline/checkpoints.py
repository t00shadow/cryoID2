"""Atomic, content-validated stage checkpoints; temporary work is never a cache.

Custom scientific runners must declare a stable ``checkpoint_id`` to opt in.
Opaque callables cannot safely describe the settings that change their results.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


SCHEMA = 1
LOCATIONS = {"tracing": "tracing/manifest.json", "search": "search/manifest.json",
             "structure": "structure.json", "complete": "complete.json"}


def json_value(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def key(value) -> str:
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".checkpoint-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(json_value(value), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def project_lock(root: Path):
    """OS-released lock: a killed process cannot leave a stale exclusive lock."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".cryoid-run.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OSError("Another CryoID run is using this project. Wait for it to finish or abort it first.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Checkpoints:
    def __init__(self, project: Path):
        self.project = project.resolve()
        self.digests = {}
        self.sources = {}
        self.keys = {}
        self.request = None

    @staticmethod
    def supported(pipeline) -> bool:
        return all(runner is None or isinstance(getattr(runner, "checkpoint_id", None), str)
                   and bool(runner.checkpoint_id) for runner in (
                       pipeline.unet_runner, pipeline.protein_query_runner, pipeline.protein_blast_runner,
                       pipeline.rna_query_runner, pipeline.rna_ranking_runner, pipeline.struct_fetch_runner,
                       pipeline.rna_structure_runner))

    def digest(self, path: Path, *, source=False) -> dict:
        path = path.resolve()
        stat = path.stat()
        state = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self.digests.get(path)
        if cached is None or cached[0] != state:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            after = path.stat()
            if state != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError(f"File changed while validating a checkpoint: {path}")
            cached = (state, {"size": stat.st_size, "sha256": digest.hexdigest()})
            self.digests[path] = cached
        if source:
            self.sources[path] = cached
        return dict(cached[1])

    def assert_sources_unchanged(self) -> None:
        for path, (_state, expected) in list(self.sources.items()):
            if self.digest(path) != expected:
                raise ValueError(f"Input or scientific code changed during this run: {path}. Run again with stable inputs.")

    def configure(self, pipeline, density: Path, pool: Path, mode: str, stop: bool) -> None:
        from C_Unet_model_predicting.prediction_provenance import _checkpoint_paths
        root = Path(__file__).resolve().parents[1]

        def code(directory):
            files = []
            for base, directories, names in os.walk(root / directory):
                directories[:] = sorted(d for d in directories if not d.startswith(".")
                                         and d not in {"__pycache__", "validation", "experiments", "build"})
                files.extend(Path(base) / name for name in names if name.endswith(".py"))
            return key([(p.relative_to(root).as_posix(), self.digest(p, source=True)) for p in sorted(files)])

        def versions(*names):
            result = {}
            for name in names:
                try:
                    result[name] = metadata.version(name)
                except metadata.PackageNotFoundError:
                    result[name] = "missing"
            return result

        def runner_id(runner):
            return getattr(runner, "checkpoint_id", "production")

        weights = None
        if pipeline.unet_runner is None:
            if pipeline.unet_weights_dir is None:
                raise FileNotFoundError("Weights are not resolved yet")
            weights_root = Path(pipeline.unet_weights_dir).resolve()
            weights = [(p.relative_to(weights_root).as_posix(), self.digest(p, source=True))
                       for p in _checkpoint_paths(weights_root)]
        inference = key({"schema": SCHEMA, "density": self.digest(density, source=True),
                         "map_name": density.name, "mode": mode, "rcl": pipeline.unet_rcl,
                         "weights": weights, "runner": runner_id(pipeline.unet_runner),
                         "code": code("C_Unet_model_predicting"),
                         "versions": versions("torch", "numpy", "scipy", "mrcfile")})
        tracing = key({"prediction": inference, "mode": mode,
                       "min_trace_length": pipeline.protein_min_trace_length if mode == "protein" else None,
                       "runner": runner_id(pipeline.protein_query_runner if mode == "protein" else pipeline.rna_query_runner),
                       "code": code("H_Backbone_tracing"),
                       "versions": versions("numpy", "scipy", "scikit-learn", "mrcfile")})
        programs = ("blastp", "blastn", "makeblastdb")
        blast = [(name, self.digest(pipeline.blast_bin_dir / (name + (".exe" if os.name == "nt" else "")), source=True))
                 for name in programs]
        search = key({"tracing": tracing, "pool": self.digest(pool, source=True),
                      "runner": runner_id(pipeline.protein_blast_runner if mode == "protein" else pipeline.rna_ranking_runner),
                      "code": code("D_BLAST_searching"),
                      "blast": blast,
                      "versions": versions("parasail", "biopython", "numpy")})
        selection = pipeline._effective_structure_selection(mode)
        structure = key({"search": search, "selection": [selection.kind, str(selection.numeric.normalize())],
                         "runner": runner_id(pipeline.struct_fetch_runner if mode == "protein" else pipeline.rna_structure_runner),
                         "code": code("I_Structure_fitting"),
                         "versions": versions("numpy", "scipy", "gemmi", "biopython")})
        self.keys = {"prediction": inference, "tracing": tracing, "search": search, "structure": structure}
        self.request = key({"scope": "identification" if stop else "full",
                            "science": search if stop else structure})

    def stage_key(self, stage: str, files) -> str:
        return key({"settings": self.keys[stage], "upstream": [(Path(p).name, self.digest(Path(p)))
                                                               for p in sorted(set(map(Path, files)))]})

    def safe_path(self, relative: str) -> Path:
        path = (self.project / relative).resolve()
        path.relative_to(self.project)
        return path

    def find(self, stage: str, expected: str):
        for manifest in sorted((self.project / "runs").glob(f"run_*/checkpoints/{LOCATIONS[stage]}"), reverse=True):
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
                if (record.get("schema") != SCHEMA or record.get("stage") != stage
                        or record.get("key") != expected or not isinstance(record.get("payload"), dict)
                        or not record.get("files")):
                    continue
                payload = record["payload"]
                if record.get("payload_sha256") != key(payload):
                    continue
                if stage == "complete":
                    result = payload.get("result")
                    if not isinstance(result, dict) or not result.get("stages") or not all(
                        isinstance(s, dict) and s.get("status") == "ok" for s in result["stages"]
                    ):
                        continue
                    self.safe_path(result["run_dir"])
                else:
                    saved_stage = payload.get("stage")
                    if not isinstance(saved_stage, dict) or saved_stage.get("status") != "ok" or not all(
                        isinstance(saved_stage.get(k), str) for k in ("name", "message")
                    ) or not set(saved_stage) <= {"name", "status", "message", "outputs", "details"} or not isinstance(payload.get("outputs"), list):
                        continue
                    registered = {str(self.safe_path(item["path"])) for item in record["files"]}
                    if not payload["outputs"] or not set(payload["outputs"]) <= registered:
                        continue
                    if stage == "tracing" and (payload.get("query") not in registered or not isinstance(payload.get("source"), str)):
                        continue
                    if stage == "search":
                        hit = payload.get("hit")
                        if (payload.get("ranking") not in registered or not isinstance(hit, dict)
                                or not {"query_id", "record_id", "score", "sequence"} <= set(hit)
                                or not set(hit) <= {"query_id", "record_id", "score", "sequence", "description", "chains"}
                                or not isinstance(hit.get("chains", []), list)):
                            continue
                for item in record["files"]:
                    if self.digest(self.safe_path(item["path"])) != {"size": item["size"], "sha256": item["sha256"]}:
                        raise ValueError("Checkpoint output is damaged")
                return manifest, record
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        return None

    def save(self, stage: str, run: Path, expected: str, payload: dict, files) -> Path:
        self.assert_sources_unchanged()
        entries = []
        for path in sorted(set(Path(p).resolve() for p in files)):
            entries.append({"path": path.relative_to(self.project).as_posix(), **self.digest(path)})
        if not entries:
            raise ValueError(f"Cannot commit an empty {stage} checkpoint")
        destination = run / "checkpoints" / LOCATIONS[stage]
        atomic_json(destination, {"schema": SCHEMA, "stage": stage, "key": expected,
                                  "payload": payload, "payload_sha256": key(payload), "files": entries})
        return destination

    @staticmethod
    def remap(value, old: Path, new: Path):
        if isinstance(value, dict):
            return {k: Checkpoints.remap(v, old, new) for k, v in value.items()}
        if isinstance(value, list):
            return [Checkpoints.remap(v, old, new) for v in value]
        if isinstance(value, str):
            try:
                return str(new / Path(value).relative_to(old))
            except ValueError:
                pass
        return value

    def materialize(self, found, run: Path) -> dict:
        manifest, record = found
        old_run = manifest
        while old_run.parent.name != "runs":
            old_run = old_run.parent
        old_run.resolve().relative_to((self.project / "runs").resolve())
        run.resolve().relative_to((self.project / "runs").resolve())
        for item in record["files"]:
            source = self.safe_path(item["path"])
            destination = run / source.relative_to(old_run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if self.digest(destination) != {"size": item["size"], "sha256": item["sha256"]}:
                raise ValueError(f"Checkpoint changed while being restored: {source}")
        return self.remap(record["payload"], old_run, run)

    def promote(self, run: Path, stage: str, files) -> tuple[list[Path], dict[str, str]]:
        promoted, mapping = [], {}
        prefix = run / "tmp" / ("tracing" if stage == "tracing" else "sequence_search")
        for source in dict.fromkeys(Path(p).resolve() for p in files):
            source.relative_to(run.resolve())
            if not source.is_file():
                raise FileNotFoundError(f"Completed {stage} output is missing: {source}")
            if source == run / "sequence_ranking.tsv":
                destination = source
            else:
                try:
                    relative = source.relative_to(prefix)
                except ValueError:
                    relative = Path("files") / source.relative_to(run)
                destination = run / "checkpoints" / stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if self.digest(source) != self.digest(destination):
                    raise ValueError(f"Completed stage output changed during checkpoint publication: {source}")
            promoted.append(destination)
            mapping[str(source)] = str(destination)
        return promoted, mapping
