import os
import re
import pandas as pd
from collections import defaultdict

def parse_filename(filename):
    """Extract hyperparameters and loc setting from filename."""
    match = re.match(r"tsh(\d+)-hidden(\d+)-lr([0-9\.e-]+)-wd([0-9\.e-]+)-kernal(\d+)-alpha([0-9\.e-]+)(loc)?\.log", filename)
    if match:
        tsh, hidden, lr, wd, kernal, alpha, loc = match.groups()
        return {
            "tsh": int(tsh),
            "hidden": int(hidden),
            "lr": float(lr),
            "wd": float(wd),
            "kernal": int(kernal),
            "alpha": float(alpha),
            "loc": bool(loc)  # True if 'loc' is present
        }
    return None

def extract_metrics(filepath):
    """Extract the last four metric lines from the log file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract last 4 relevant lines
    metrics = {}
    for line in lines[-4:]:
        match = re.search(r"(MRR|Hit@1|Hit@3|Hit@10): ([0-9\.]+) ", line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    
    return metrics if len(metrics) == 4 else None

def process_logs(log_folder):
    """Process all logs in the folder and extract best MRR results."""
    results = defaultdict(list)
    
    for filename in os.listdir(log_folder):
        if filename.endswith(".log"):
            params = parse_filename(filename)
            if params:
                metrics = extract_metrics(os.path.join(log_folder, filename))
                if metrics:
                    key = (params['tsh'], params['loc'])
                    results[key].append({**params, **metrics})
    
    # Find best MRR for each tsh and loc setting
    best_results = []
    for (tsh, loc), runs in results.items():
        best_run = max(runs, key=lambda x: x['MRR'])
        best_results.append(best_run)
    
    return pd.DataFrame(best_results)

def main():
    log_folder = "./o_log/EpiSim/DTHGNN/detect"  # Change this to your actual log directory
    df = process_logs(log_folder)
    print(df)

if __name__ == "__main__":
    main()