import os
import tarfile
import zipfile

import requests


def download_file(url: str, filename: str) -> None:
    response = requests.get(url, stream=True, timeout=5)
    total = int(response.headers.get("content-length", 0))
    totalMb = round(total / 1024 / 1024)

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress = f.tell()
                progressMb = round(progress / 1024 / 1024)
                progressPercent = 100 * progress // total
                print(
                    f"\rDownloaded {progressMb} of {totalMb} MB ({progressPercent}%)",
                    end="",
                )
    print()


def unzip_file(filename: str, output_folder: str) -> None:
    with zipfile.ZipFile(filename, "r") as zip_ref:
        for member in zip_ref.namelist():
            # Ensure file path is safe
            if os.path.isabs(member) or ".." in member:
                print(f"Skipping potentially unsafe file {member}")
                continue

            # Handle directories
            if member.endswith("/"):
                os.makedirs(os.path.join(output_folder, member), exist_ok=True)
                continue

            # Extract file
            source = zip_ref.open(member)
            target = open(os.path.join(output_folder, member), "wb")

            with source, target:
                while True:
                    chunk = source.read(1024)
                    if not chunk:
                        break
                    target.write(chunk)


def untar_file(filename: str, output_folder: str) -> None:
    with tarfile.open(filename, "r:gz") as tar:
        for member in tar.getmembers():
            # Ensure file path is safe
            if os.path.isabs(member.name) or ".." in member.name:
                print(f"Skipping potentially unsafe file {member.name}")
                continue

            # Handle directories
            if member.type == tarfile.DIRTYPE:
                os.makedirs(os.path.join(output_folder, member.name), exist_ok=True)
                continue

            # Extract file
            source = tar.extractfile(member)
            if source is None:
                continue
            target = open(os.path.join(output_folder, member.name), "wb")

            with source, target:
                while True:
                    chunk = source.read(1024)
                    if not chunk:
                        break
                    target.write(chunk)
