#!/bin/bash
#SBATCH --job-name=diagnose_singularity
#SBATCH --nodelist=hpc-pr-a-pod10
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/diagnose_%j.out
#SBATCH --error=logs/diagnose_%j.err

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CARLA_SIF="${PROJECT_ROOT}/carla_official.sif"

echo "=========================================="
echo "SINGULARITY DIAGNOSTIC"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "UID: $(id -u)"
echo "Groups: $(id -G)"
echo ""

# Check Singularity version
echo "Singularity version:"
singularity --version 2>&1 || echo "singularity command failed"
apptainer --version 2>&1 || echo "apptainer command not found"
echo ""

# Check authentication method
echo "Authentication check:"
echo "getent passwd output:"
getent passwd $(whoami) || echo "getent failed"
echo ""
echo "NSS config:"
cat /etc/nsswitch.conf | grep passwd || echo "nsswitch.conf not accessible"
echo ""

# Check for SSSD/LDAP
echo "SSSD check:"
ls -la /var/lib/sss/pipes/ 2>&1 || echo "No SSSD pipes"
ls -la /etc/sssd/ 2>&1 || echo "No SSSD config"
echo ""

# Test 1: Basic singularity exec
echo "TEST 1: Basic singularity exec"
singularity exec "$CARLA_SIF" bash -c "whoami && id" 2>&1 || echo "FAILED"
echo ""

# Test 2: With --nv flag
echo "TEST 2: With --nv"
singularity exec --nv "$CARLA_SIF" bash -c "whoami && id" 2>&1 || echo "FAILED"
echo ""

# Test 3: With containall
echo "TEST 3: With --containall"
singularity exec --containall --nv "$CARLA_SIF" bash -c "whoami && id" 2>&1 || echo "FAILED"
echo ""

# Test 4: With --no-home
echo "TEST 4: With --no-home"
singularity exec --no-home --nv "$CARLA_SIF" bash -c "whoami && id" 2>&1 || echo "FAILED"
echo ""

# Test 5: Check if running as nobody works
echo "TEST 5: Check container's passwd file"
singularity exec "$CARLA_SIF" bash -c "cat /etc/passwd | tail -5" 2>&1 || echo "FAILED"
echo ""

# Test 6: Try with environment variable
echo "TEST 6: With SINGULARITYENV_USER"
SINGULARITYENV_USER=carla singularity exec "$CARLA_SIF" bash -c "whoami && id" 2>&1 || echo "FAILED"
echo ""

# Test 7: Check Singularity configuration
echo "TEST 7: Singularity configuration"
echo "SINGULARITY_NOSUID=${SINGULARITY_NOSUID}"
echo "APPTAINER_NOSUID=${APPTAINER_NOSUID}"
singularity exec "$CARLA_SIF" bash -c "echo UID inside: \$(id -u)" 2>&1 || echo "FAILED"
echo ""

echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="