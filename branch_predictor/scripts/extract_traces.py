#!/usr/bin/env python3
"""
Extract trace files from compressed archives.

Usage:
    python extract_traces.py --input traces/int.tar.xz --output data/branch/
"""

import argparse
import tarfile
import lzma
import gzip
import os
from pathlib import Path


def extract_archive(input_path: Path, output_dir: Path) -> None:
    """Extract compressed archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = input_path.suffix.lower()
    
    if suffix in ['.xz', '.lzma']:
        # tar.xz or tar.lzma
        if '.tar' in input_path.name.lower():
            with lzma.open(input_path) as xz_file:
                with tarfile.open(fileobj=xz_file) as tar:
                    print(f"Extracting {input_path.name}...")
                    tar.extractall(output_dir)
                    print(f"Extracted to {output_dir}")
        else:
            # Single .xz file
            output_file = output_dir / input_path.stem
            with lzma.open(input_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Extracted to {output_file}")
    
    elif suffix == '.gz':
        if '.tar' in input_path.name.lower():
            with tarfile.open(input_path, 'r:gz') as tar:
                print(f"Extracting {input_path.name}...")
                tar.extractall(output_dir)
                print(f"Extracted to {output_dir}")
        else:
            output_file = output_dir / input_path.stem
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Extracted to {output_file}")
    
    elif suffix == '.tar':
        with tarfile.open(input_path, 'r:') as tar:
            tar.extractall(output_dir)
        print(f"Extracted to {output_dir}")
    
    else:
        print(f"Unknown format: {suffix}")


def main():
    parser = argparse.ArgumentParser(description="Extract trace files")
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input archive file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Extract all archives in input directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if args.all and input_path.is_dir():
        # Extract all archives in directory
        for archive in input_path.glob('*.tar.*'):
            extract_archive(archive, output_dir)
        for archive in input_path.glob('*.xz'):
            if '.tar' not in archive.name:
                extract_archive(archive, output_dir)
    else:
        extract_archive(input_path, output_dir)


if __name__ == "__main__":
    main()
