import subprocess
import pandas as pd
import csv
import sys

if __name__ == "__main__":
    csv_path = "db_benchmark_results.csv"
    header = ["Modality", "Method", "N", "Dim", "Feature Repr.", "System Backend", "Search Strategy", 
              "Preprocessing (ms)", "Indexing (ms)", "Search P1 (ms)", "Collective P2 (ms)"]
    
    final_rows = []
    tasks = ["Tabular", "Temporal", "Graph", "Image", "Text", "MultiModal"]
    
    for t in tasks:
        print(f"Executing REAL model benchmark for {t}...")
        result = subprocess.run([sys.executable, "benchmark_worker.py", t], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.strip():
                    final_rows.append(line.split(','))
        else:
            print(f"Error in {t}: {result.stderr}")

    df = pd.DataFrame(final_rows, columns=header)
    df.to_csv(csv_path, index=False)
    print(f"\nFinal Results Saved to {csv_path}. Benchmark 100% Complete.")
