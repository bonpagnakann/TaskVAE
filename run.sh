#!/usr/bin/env bash

# This script submits a job to the OAR job scheduler with specific GPU and walltime requirements.

# Function to display usage
usage() {
  echo "Usage: $0 [-s script] [-h host] [-g gpu] [-t hours]"
  echo "  -s script   Path to the script to execute (default: './script_gpu3.sh')"
  echo "  -h host     Host to submit the job to (default: 'lig-gpu3.imag.fr')"
  echo "  -g gpu      Number of GPUs required"
  echo "  -t hours    Walltime in hours"
  exit 1
}

# Parse command line arguments
while getopts "s:h:g:t:" opt; do
  case $opt in
    s) script="$OPTARG"
    ;;
    h) host="$OPTARG"
    ;;
    g) gpu="$OPTARG"
    ;;
    t) hour="$OPTARG"
    ;;
    \?) usage
    ;;
  esac
done

# Check if GPU and hour parameters are provided
if [ -z "$gpu" ] || [ -z "$hour" ]; then
  usage
fi

# Print the configuration
echo "Submitting job with the following configuration:"
echo "Script: $script"
echo "Host: $host"
echo "GPU: $gpu"
echo "Walltime: $hour hours"

# Submit the job using oarsub
oarsub -l /host=1/gpu=$gpu,walltime=$hour:00:00 -p "host='$host'" $script