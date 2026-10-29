# Bundled BLAST+

CryoID ships NCBI BLAST+ **2.17.0** for **Windows x86-64**, **Linux x86-64**,
**macOS Intel (x86-64)** and **macOS Apple Silicon (arm64)**.
It remains a Python package: freezing CryoID into an executable is unnecessary.
All four payloads are included in source checkouts and Python wheels. Normal setup
and execution never download BLAST or look for a system installation.

`bundled/<platform>/bin/` contains `blastp`, `blastn`, and `makeblastdb`, plus
the Windows DLLs and executable manifests supplied by NCBI. Each platform's
`LICENSE` and `README` are retained. `bundled/manifest.json` records the upstream
archive URL, MD5, SHA-256, and every shipped file's size and SHA-256.
The stage's `.gitattributes` prevents Git newline conversion of these payloads.

The official archives come from the [NCBI 2.17.0 release directory](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/).
NCBI describes its software as public domain in its
[developer information](https://blast.ncbi.nlm.nih.gov/doc/blast-help/developerinfo.html);
consult the included notices for terms and third-party acknowledgments.

The Windows payload also includes `nghttp2.dll`, an HTTP/2 library used by
NCBI's networking components. Its MIT copyright and permission notice is
retained in [`licenses/nghttp2-LICENSE.txt`](licenses/nghttp2-LICENSE.txt)
and included in Python distributions. The notice comes from
[nghttp2's upstream COPYING file](https://github.com/nghttp2/nghttp2/blob/master/COPYING).
These dependency notices do not set the license of CryoID's own code.

## Runtime behavior

`blast_tools.py` selects the matching private directory for the CLI, GUI,
protein search, and RNA structure lookup. BLAST requires no user configuration.

BLAST still performs the same CPU search with unchanged search arguments,
scoring, and ranking code. CUDA accelerates U-Net inference only. Missing or
broken bundled tools produce an installation repair message in Check setup.
An unrelated system BLAST installation is never used as a fallback.

Linux uses the upstream dynamically linked executables and standard OS runtime
libraries: glibc, zlib, bzip2, libgomp, and libgcc. These resolved on Ubuntu in
WSL during validation. A minimal Linux image may need its distribution's
runtime packages; this bundle does not supply a complete Linux userspace.
Wheels preserve Linux and macOS executable permissions even when built on Windows.
The Mac payloads are the official `x64-macosx` and `aarch64-macosx` archives;
their Mach-O deployment target is macOS 14.0. They link only to Apple system
libraries and frameworks. Selection follows the architecture of the running
Python interpreter. Native arm64 Python does not need Rosetta.

## Reproduce the bundle

From the repository root, with Python and network access:

```bash
python D_BLAST_searching/bundle_blast.py
python -m build --wheel
```

To refresh only the Mac payloads, pass
`--platform darwin-x86_64 --platform darwin-arm64` to `bundle_blast.py`.
Local setup notes and the scientific comparison narrative are retained under
`experiments/`; the archived scientific comparisons below predate macOS.

The maintainer script verifies the NCBI MD5 and the recorded archive SHA-256,
then writes only the required tools and support files. Archives stay under
`D_BLAST_searching/.downloads/` (ignored). To change BLAST versions, update the
pin deliberately and repeat the scientific comparisons before releasing.

## Scientific verification

See the machine-readable reports in [`validation/`](validation/).
The comparison checks full `sequence_ranking.tsv` files, including scores,
and every sequence in ordered `ranked_candidates.fasta` files, byte for byte.
It also builds fresh nucleotide databases and checks RNA BLASTn hits.
Existing U-Net predictions are reused; inference is not rerun.

To regenerate the local cases and compare against a known reference installation:

```bash
python D_BLAST_searching/verify_bundle.py --prepare --reference-bin /path/to/reference/bin
```

The Linux comparison can extract an independent reference directly from the
checksum-verified upstream archive and compare outputs across platforms:

```bash
python D_BLAST_searching/verify_bundle.py --reference-archive D_BLAST_searching/.downloads/ncbi-blast-2.17.0+-x64-linux.tar.gz --cross-report D_BLAST_searching/validation/windows-x86_64.json
```

Input provenance and hashes are in `validation/inputs.json`. All generated
queries, databases, outputs, and build artifacts stay in `validation/work/`
(ignored). Source prediction maps and candidate pools are left intact.
The archived datasets are local experiment data and are not distributed
inside the wheel.
