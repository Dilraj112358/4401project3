"""Extract the Colab results ZIP and list its files.

Run: python extract_distilbert_results.py
Optional: python extract_distilbert_results.py another_results.zip --output results
"""

import argparse
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import zipfile


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = PROJECT_DIR / "wine_distilbert_results.zip"


def _same_contents(archive, member, target):
    if target.stat().st_size != member.file_size:
        return False
    with archive.open(member) as source, target.open("rb") as existing:
        while chunk := source.read(1024 * 1024):
            if existing.read(len(chunk)) != chunk:
                return False
        return existing.read(1) == b""


def extract_results(archive_path, output_dir):
    """Return extracted file paths; preserve existing files that differ.

    Re-running with the same archive is supported. Validate every entry and
    existing destination before writing anything from the archive.
    """
    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"ZIP file not found: {archive_path}\n"
            "Put the downloaded results ZIP in the project folder and run again."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"The output location is already a file: {output_dir}")

    with zipfile.ZipFile(archive_path) as archive:
        planned = []
        seen = set()
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or PureWindowsPath(name).drive
                or not relative.parts
                or ".." in relative.parts
                or any(
                    ":" in part
                    or part.endswith((".", " "))
                    or PureWindowsPath(part).is_reserved()
                    for part in relative.parts
                )
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsupported path in ZIP: {member.filename!r}")

            target = output_dir.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(output_dir) or target == output_dir:
                raise ValueError(f"Path leaves the output folder: {member.filename!r}")
            if target in seen:
                raise ValueError(f"Duplicate destination in ZIP: {member.filename!r}")
            seen.add(target)

            for parent in target.parents:
                if parent.exists() and not parent.is_dir():
                    raise FileExistsError(f"A file blocks the output folder: {parent}")
                if parent == output_dir:
                    break

            if member.is_dir():
                if target.exists() and not target.is_dir():
                    raise FileExistsError(f"A file blocks this directory: {target}")
            elif target.exists():
                if not target.is_file() or not _same_contents(archive, member, target):
                    raise FileExistsError(
                        f"An existing file would be overwritten: {target}\n"
                        "Use a different output folder to keep both versions."
                    )
            planned.append((member, target))

        # Check file/directory conflicts inside the archive before extraction.
        file_targets = {target for member, target in planned if not member.is_dir()}
        for _, target in planned:
            if any(parent in file_targets for parent in target.parents):
                raise ValueError(f"The ZIP uses a file as a directory: {target}")

        for member, target in planned:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)

    return sorted(file_targets, key=lambda path: path.relative_to(output_dir).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, help="Destination folder")
    options = parser.parse_args()
    output_dir = (options.output or options.archive.with_suffix("")).resolve()
    try:
        paths = extract_results(options.archive, output_dir)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Results folder: {output_dir}")
    print(f"Files available: {len(paths)}\n")
    for path in paths:
        print(f"{path.relative_to(output_dir).as_posix()} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
