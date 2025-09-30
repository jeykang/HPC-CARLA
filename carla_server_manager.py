#!/usr/bin/env python3
"""
CARLA Server Manager - Fixed to work with Singularity containers
Maintains persistent CARLA servers for each GPU
"""

import os
import sys
import time
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime
import threading
import logging

class CarlaServer:
    """Manages a single CARLA server instance through Singularity"""
    
    def __init__(self, gpu_id, rpc_port, streaming_port=0, 
                 carla_sif=None, project_root=None):
        self.gpu_id = gpu_id
        self.rpc_port = rpc_port
        self.streaming_port = streaming_port
        
        # Get paths from environment or defaults
        self.project_root = project_root or os.environ.get('PROJECT_ROOT', os.getcwd())
        self.carla_sif = carla_sif or os.path.join(self.project_root, 'carla_official.sif')
        
        self.process = None
        self.start_time = None
        self.restart_count = 0
        self.last_health_check = None
        self.is_healthy = False
        
    def start(self):
        """Start the CARLA server through Singularity"""
        if self.is_running():
            logging.info(f"GPU {self.gpu_id}: CARLA already running on port {self.rpc_port}")
            return True
        
        logging.info(f"GPU {self.gpu_id}: Starting CARLA on port {self.rpc_port}")
        
        # Check if Singularity image exists
        if not os.path.exists(self.carla_sif):
            logging.error(f"GPU {self.gpu_id}: CARLA Singularity image not found at {self.carla_sif}")
            return False
        
        # Build Singularity command with passwd/group binding
        cmd = [
            'singularity', 'exec', '--nv',
            '--bind', f'{self.project_root}:/workspace',
            '--bind', '/etc/passwd:/etc/passwd:ro',  # Bind passwd file (read-only)
            '--bind', '/etc/group:/etc/group:ro',    # Bind group file (read-only)
            self.carla_sif,
            'bash', '-c',
            f"""
            export CUDA_VISIBLE_DEVICES={self.gpu_id}
            export XDG_RUNTIME_DIR=/tmp/runtime-carla-gpu{self.gpu_id}-$$
            mkdir -p $XDG_RUNTIME_DIR
            export SDL_AUDIODRIVER=dsp
            export SDL_VIDEODRIVER=offscreen
            cd /home/carla
            exec ./CarlaUE4.sh -carla-rpc-port={self.rpc_port} -carla-streaming-port={self.streaming_port} -nosound -quality-level=Epic
            """
        ]
        
        # Set up environment for the Singularity process
        env = os.environ.copy()
        # Don't override CUDA_VISIBLE_DEVICES if SLURM already set it
        if 'SLURM_JOB_ID' not in env:
            env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        # Start process through Singularity
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Capture initial output for debugging
            import select
            import time as time_module
            
            time_module.sleep(2)  # Give it a moment to start
            
            # Check if process died immediately
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logging.error(f"GPU {self.gpu_id}: CARLA process died immediately")
                logging.error(f"GPU {self.gpu_id}: STDOUT: {stdout.decode('utf-8', errors='replace')}")
                logging.error(f"GPU {self.gpu_id}: STDERR: {stderr.decode('utf-8', errors='replace')}")
                return False
                
            self.start_time = datetime.now()
            
            # Wait for server to be ready
            if self._wait_for_ready(timeout=120):
                self.is_healthy = True
                logging.info(f"GPU {self.gpu_id}: CARLA server ready on port {self.rpc_port}")
                return True
            else:
                logging.error(f"GPU {self.gpu_id}: CARLA failed to start properly")
                # Capture output for debugging
                stdout, stderr = self.process.communicate(timeout=5)
                logging.error(f"GPU {self.gpu_id}: STDOUT: {stdout.decode('utf-8', errors='replace')}")
                logging.error(f"GPU {self.gpu_id}: STDERR: {stderr.decode('utf-8', errors='replace')}")
                self.stop()
                return False
                
        except Exception as e:
            logging.error(f"GPU {self.gpu_id}: Failed to start CARLA: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def stop(self):
        """Stop the CARLA server"""
        if self.process:
            logging.info(f"GPU {self.gpu_id}: Stopping CARLA server")
            try:
                self.process.terminate()
                time.sleep(2)
                if self.process.poll() is None:
                    self.process.kill()
                self.process = None
            except:
                pass
        
        # Also kill any process on this port
        try:
            subprocess.run(f"fuser -k {self.rpc_port}/tcp", shell=True, capture_output=True)
        except:
            pass
        
        self.is_healthy = False
    
    def restart(self):
        """Restart the CARLA server"""
        logging.info(f"GPU {self.gpu_id}: Restarting CARLA server (restart #{self.restart_count + 1})")
        self.stop()
        time.sleep(5)
        self.restart_count += 1
        return self.start()
    
    def is_running(self):
        """Check if the server process is running"""
        if self.process:
            return self.process.poll() is None
        return False
    
    def _wait_for_ready(self, timeout=120):
        """Wait for CARLA server to be ready"""
        import socket
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.is_running():
                return False
            
            # Try to connect to the RPC port
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', self.rpc_port))
                sock.close()
                
                if result == 0:
                    # Port is open, server is likely ready
                    time.sleep(10)  # Give it a bit more time to fully initialize
                    return True
            except:
                pass
            
            time.sleep(2)
        
        return False
    
    def health_check(self):
        """Check if the server is healthy"""
        if not self.is_running():
            self.is_healthy = False
            return False
        
        # Try to connect to the port
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', self.rpc_port))
            sock.close()
            
            self.is_healthy = (result == 0)
            self.last_health_check = datetime.now()
            return self.is_healthy
        except:
            self.is_healthy = False
            return False
    
    def get_status(self):
        """Get server status information"""
        return {
            'gpu_id': self.gpu_id,
            'rpc_port': self.rpc_port,
            'is_running': self.is_running(),
            'is_healthy': self.is_healthy,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'restart_count': self.restart_count,
            'uptime': str(datetime.now() - self.start_time) if self.start_time else None
        }


class CarlaServerManager:
    """Manages multiple CARLA servers across GPUs"""
    
    def __init__(self, project_root=None, num_gpus=8, base_rpc_port=2000, port_spacing=100):
        self.project_root = Path(project_root or os.environ.get('PROJECT_ROOT', os.getcwd()))
        self.num_gpus = num_gpus
        self.base_rpc_port = base_rpc_port
        self.port_spacing = port_spacing
        self.servers = {}
        self.running = False
        self.monitor_thread = None
        
        # Path to CARLA Singularity image
        self.carla_sif = self.project_root / 'carla_official.sif'
        
        # Set up logging
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'carla_server_manager.log'),
                logging.StreamHandler()
            ]
        )
        
        # Status file for external monitoring
        self.status_file = self.project_root / 'collection_state' / 'carla_servers.json'
        self.status_file.parent.mkdir(exist_ok=True, parents=True)
    
    def start_all(self, gpu_list=None):
        """Start CARLA servers on specified GPUs"""
        if gpu_list is None:
            gpu_list = list(range(self.num_gpus))
        
        logging.info(f"Starting CARLA servers on GPUs: {gpu_list}")
        logging.info(f"Using Singularity image: {self.carla_sif}")
        
        # Check if Singularity image exists
        if not self.carla_sif.exists():
            logging.error(f"CARLA Singularity image not found at {self.carla_sif}")
            return 0
        
        for gpu_id in gpu_list:
            rpc_port = self.base_rpc_port + (gpu_id * self.port_spacing)
            
            # Check if port is already in use
            if self._is_port_in_use(rpc_port):
                logging.warning(f"Port {rpc_port} already in use, attempting cleanup")
                self._free_port(rpc_port)
                time.sleep(2)
            
            # Create and start server with Singularity support
            server = CarlaServer(
                gpu_id, 
                rpc_port,
                carla_sif=str(self.carla_sif),
                project_root=str(self.project_root)
            )
            if server.start():
                self.servers[gpu_id] = server
                logging.info(f"Successfully started CARLA on GPU {gpu_id}")
            else:
                logging.error(f"Failed to start CARLA on GPU {gpu_id}")
        
        if self.servers:
            # Start monitoring thread
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_servers)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
        
        self._update_status_file()
        
        return len(self.servers)
    
    def stop_all(self):
        """Stop all CARLA servers"""
        logging.info("Stopping all CARLA servers")
        self.running = False
        
        for gpu_id, server in self.servers.items():
            server.stop()
        
        self.servers.clear()
        self._update_status_file()
    
    def restart_server(self, gpu_id):
        """Restart a specific server"""
        if gpu_id in self.servers:
            if self.servers[gpu_id].restart():
                logging.info(f"Successfully restarted CARLA on GPU {gpu_id}")
                self._update_status_file()
                return True
            else:
                logging.error(f"Failed to restart CARLA on GPU {gpu_id}")
                # Remove from active servers
                del self.servers[gpu_id]
                self._update_status_file()
                return False
        else:
            # Try to start a new server for this GPU
            rpc_port = self.base_rpc_port + (gpu_id * self.port_spacing)
            server = CarlaServer(
                gpu_id,
                rpc_port,
                carla_sif=str(self.carla_sif),
                project_root=str(self.project_root)
            )
            if server.start():
                self.servers[gpu_id] = server
                logging.info(f"Started new CARLA server on GPU {gpu_id}")
                self._update_status_file()
                return True
        return False
    
    def get_server_info(self, gpu_id):
        """Get information about a specific server"""
        if gpu_id in self.servers:
            return self.servers[gpu_id].get_status()
        return None
    
    def _monitor_servers(self):
        """Monitor server health and restart if necessary"""
        while self.running:
            for gpu_id, server in list(self.servers.items()):
                if not server.health_check():
                    logging.warning(f"GPU {gpu_id}: CARLA server unhealthy, attempting restart")
                    
                    # Try to restart up to 3 times
                    if server.restart_count < 3:
                        if not self.restart_server(gpu_id):
                            logging.error(f"GPU {gpu_id}: Failed to restart, removing from pool")
                    else:
                        logging.error(f"GPU {gpu_id}: Max restarts reached, removing from pool")
                        server.stop()
                        del self.servers[gpu_id]
            
            self._update_status_file()
            time.sleep(30)  # Check every 30 seconds
    
    def _update_status_file(self):
        """Update the status file for external monitoring"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'servers': {}
        }
        
        for gpu_id, server in self.servers.items():
            status['servers'][str(gpu_id)] = server.get_status()
        
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to update status file: {e}")
    
    def _is_port_in_use(self, port):
        """Check if a port is in use"""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
    
    def _free_port(self, port):
        """Try to free a port by killing the process using it"""
        try:
            # Use fuser to kill process on port
            subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        except:
            pass
    
    def wait_for_ready(self, timeout=120):
        """Wait for all servers to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_ready = True
            for gpu_id, server in self.servers.items():
                if not server.is_healthy:
                    all_ready = False
                    break
            
            if all_ready and self.servers:
                return True
            
            time.sleep(2)
        
        return False


def main():
    """Main entry point for standalone server management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage persistent CARLA servers')
    parser.add_argument('command', choices=['start', 'stop', 'status', 'restart'],
                        help='Command to execute')
    parser.add_argument('--gpus', nargs='+', type=int,
                        help='Specific GPUs to operate on')
    parser.add_argument('--project-root', default=os.environ.get('PROJECT_ROOT', os.getcwd()),
                        help='Project root directory')
    
    args = parser.parse_args()
    
    manager = CarlaServerManager(project_root=args.project_root)
    
    if args.command == 'start':
        gpu_list = args.gpus if args.gpus else None
        count = manager.start_all(gpu_list)
        print(f"Started {count} CARLA servers")
        
        if count > 0:
            print("Waiting for servers to be ready...")
            if manager.wait_for_ready():
                print("All servers ready!")
            else:
                print("Some servers failed to start properly")
        
        # Keep running until interrupted
        try:
            print("Server manager running. Press Ctrl+C to stop all servers.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            manager.stop_all()
    
    elif args.command == 'stop':
        manager.stop_all()
        print("All servers stopped")
    
    elif args.command == 'status':
        # Read status from file
        status_file = Path(args.project_root) / 'collection_state' / 'carla_servers.json'
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            print(f"CARLA Server Status (as of {status['timestamp']})")
            print("-" * 60)
            
            for gpu_id, info in status['servers'].items():
                print(f"GPU {gpu_id}:")
                print(f"  Port: {info['rpc_port']}")
                print(f"  Running: {info['is_running']}")
                print(f"  Healthy: {info['is_healthy']}")
                print(f"  Uptime: {info['uptime']}")
                print(f"  Restarts: {info['restart_count']}")
        else:
            print("No status file found. Servers may not be running.")
    
    elif args.command == 'restart':
        if args.gpus:
            for gpu_id in args.gpus:
                if manager.restart_server(gpu_id):
                    print(f"Restarted server on GPU {gpu_id}")
                else:
                    print(f"Failed to restart server on GPU {gpu_id}")
        else:
            print("Please specify GPUs to restart with --gpus")


if __name__ == '__main__':
    main()