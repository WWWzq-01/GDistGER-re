# Scripts Directory

This directory contains various shell scripts for running, monitoring, and evaluating the GDistGER distributed graph embedding system.

- **Note**: you can copy `*.sh` to `build/` to execute directly
## Main Execution Scripts

### Primary Run Scripts
- **`run.sh`** - Main execution script that runs GDistGER on multiple datasets (fk, ytb, soc, LJ, com, twt) using 8 nodes
- **`mpich_master.sh`** - Core execution script for running the distributed graph embedding with MPI
- **`mpich_run.sh`** - Alternative MPI execution script

### Dataset-Specific Scripts
- **`run_LJ.sh`** - Runs GDistGER specifically on the LiveJournal (LJ) dataset
- **`run_soc.sh`** - Runs GDistGER on social network datasets
- **`run_ytb.sh`** - Runs GDistGER on YouTube (ytb) dataset
- **`run_50.sh`** - Runs experiments with 50 processes
- **`run_100.sh`** - Runs experiments with 100 processes

### Node Configuration Scripts
- **`mpich_master-50.sh`** - MPI master script configured for 50 nodes
- **`mpich_master-100.sh`** - MPI master script configured for 100 nodes

## Monitoring and System Scripts

### Resource Monitoring
- **`dist_stat.sh`** - Monitors distributed system statistics including CPU, memory, GPU utilization across all nodes
- **`record.sh`** - Continuously records CPU and GPU usage to log files for performance analysis
- **`a.sh`** - Simple script to read and display IP addresses from the hosts file

### Version Control
- **`git-v.sh`** - Git version control helper script

## Evaluation Scripts

### Performance Evaluation
- **`auc.sh`** - Calculates AUC (Area Under Curve) scores for link prediction tasks using generated embeddings

### Backup Scripts
- **`LJ-0.98-com-0.96-args-bak.sh`** - Backup script with specific arguments for LiveJournal and community detection experiments

### Testing
- **`test.sh`** - Testing script for GDistGER functionality

## Usage Examples

### Running on a specific dataset:
```bash
./mpich_master.sh <dataset_name> <num_nodes>
# Example: ./mpich_master.sh wiki 8
```

### Running all experiments:
```bash
./run.sh
```

### Monitoring system resources:
```bash
./dist_stat.sh
```

### Evaluating results:
```bash
./auc.sh <dataset_name>
# Example: ./auc.sh wiki
```

### Recording performance metrics:
```bash
./record.sh &  # Run in background
```

## Prerequisites

- **MPI Environment**: OpenMPI or MPICH installed and configured
- **GPU Support**: NVIDIA GPUs with nvidia-smi for GPU monitoring
- **SSH Access**: Passwordless SSH access to all nodes listed in `hosts` file
- **Dataset Files**: Properly formatted graph datasets in the specified directories

## Configuration Files

- **`hosts`** - Contains the list of IP addresses for distributed execution
- Logs are stored in the `log/` directory with format `<dataset>_<nodes>.log`

## Notes

- All scripts should be executed from the build directory of the GDistGER project
- Ensure the `hosts` file is properly configured with accessible node IP addresses
- GPU monitoring scripts require NVIDIA drivers and nvidia-smi tool
- Results and embeddings are typically output to the `out/` directory
- Performance logs include timing information and resource usage statistics

## Directory Structure Expected

```
GDistGER-re/
├── build/                  # Build directory (execution location)
│   ├── bin/               # Compiled binaries
│   ├── hosts              # Node configuration file
│   └── log/               # Execution logs
├── dataset/               # Input datasets
└── scripts/               # This directory
```