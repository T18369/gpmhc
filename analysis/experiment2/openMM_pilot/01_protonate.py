#!/usr/bin/env python3

import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Protonate an AF2-multimer pMHC structure using PDB2PQR/PROPKA."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ph", type=float, default=5.0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    pqr = os.path.join(args.output_dir,f"{base}.pqr")
    pdb = os.path.join(args.output_dir,f"{base}_protonated.pdb")
    log = os.path.join(args.output_dir,f"{base}.log")

    cmd = [
        "pdb2pqr",
        "--ff=AMBER",
        "--titration-state-method=propka",
        f"--with-ph={args.ph}",
        "--keep-chain",
        f"--pdb-output={pdb}",
        args.input,
        pqr,
    ]

    print("Running PDB2PQR / PROPKA")
    print(f"Input pH: {args.ph}")
    print(f"Input: {args.input}")

    with open(log, "w") as handle:
        subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print(f"Protonated PDB: {pdb}")
    print(f"PQR: {pqr}")
    print(f"Log: {log}")

if __name__ == "__main__":
    main()