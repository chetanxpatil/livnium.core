"""
Download DIMACS graph coloring instances.

Downloads from standard DIMACS repository.
"""

import urllib.request
import ssl
import tarfile
import zipfile
from pathlib import Path
from typing import Optional


def download_dimacs_graphs(download_dir: Path) -> Path:
    """
    Download DIMACS graph coloring instances.
    
    Returns:
        Path to directory containing graph files
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # DIMACS graph coloring instances
    # Note: DIMACS instances may need to be downloaded manually from:
    # https://mat.tepper.cmu.edu/COLOR/instances/
    # or https://github.com/mivia-lab/graph-coloring-instances
    graphs = {
        # Try alternative sources
        'col-40-5': 'https://github.com/mivia-lab/graph-coloring-instances/raw/main/col/col-40-5.col',
        'col-50-5': 'https://github.com/mivia-lab/graph-coloring-instances/raw/main/col/col-50-5.col',
        'DSJC125.1': 'https://github.com/mivia-lab/graph-coloring-instances/raw/main/DSJC/DSJC125.1.col',
        'flat300_20_0': 'https://github.com/mivia-lab/graph-coloring-instances/raw/main/flat/flat300_20_0.col',
        'flat300_26_0': 'https://github.com/mivia-lab/graph-coloring-instances/raw/main/flat/flat300_26_0.col',
        'flat300_28_0': 'https://github.com/mivia-lab/graph-coloring-instances/raw/main/flat/flat300_28_0.col',
    }
    
    print("Downloading DIMACS graph coloring instances...")
    print("(This may take a moment)")
    print()
    
    # Create SSL context to bypass certificate issues
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    downloaded = []
    failed = []
    
    for name, url in graphs.items():
        output_path = download_dir / f"{name}.col"
        
        if output_path.exists():
            print(f"  {name}: Already exists, skipping")
            downloaded.append(name)
            continue
        
        try:
            print(f"  Downloading {name}...", end=" ", flush=True)
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                with open(output_path, 'wb') as f:
                    f.write(response.read())
            print("✓")
            downloaded.append(name)
        except Exception as e:
            print(f"✗ Error: {e}")
            failed.append(name)
    
    print()
    if downloaded:
        print(f"✓ Downloaded {len(downloaded)} graphs")
    if failed:
        print(f"⚠ Failed to download {len(failed)} graphs: {failed}")
        print()
        print("You can manually download from:")
        print("https://mat.tepper.cmu.edu/COLOR/instances/")
        print()
        print("Or use generate_test_graphs.py to create synthetic instances")
    
    return download_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download DIMACS graph coloring instances')
    parser.add_argument('--output-dir', type=str, 
                       default='benchmark/graph_coloring/dimacs',
                       help='Output directory for graphs')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    download_dimacs_graphs(output_dir)
    
    print(f"\nGraphs saved to: {output_dir}")
    print("\nYou can now run:")
    print(f"  python benchmark/graph_coloring/run_graph_coloring_benchmark.py --graph-dir {output_dir}")


if __name__ == '__main__':
    main()

