"""Build hooks that preserve executable permissions for the bundled tools."""
from pathlib import Path
import stat
from zipfile import ZipFile

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


class BundledBlastWheel(bdist_wheel):
    def run(self):
        super().run()
        for _command, _version, filename in self.distribution.dist_files:
            wheel = Path(filename)
            if wheel.suffix != ".whl":
                continue
            temporary = wheel.with_suffix(".whl.tmp")
            with ZipFile(wheel) as source, ZipFile(temporary, "w") as destination:
                for entry in source.infolist():
                    content = source.read(entry)
                    if (entry.filename.startswith("D_BLAST_searching/bundled/")
                            and "/bin/" in entry.filename
                            and "/windows-" not in entry.filename):
                        entry.create_system = 3
                        entry.external_attr = (stat.S_IFREG | 0o755) << 16
                    destination.writestr(entry, content)
            temporary.replace(wheel)
