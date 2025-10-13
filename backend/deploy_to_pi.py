import paramiko
import os

def deploy_latest_agent(PI_IP, PI_USER, PI_PASSWORD, LOCAL_PATH, REMOTE_PATH="/home/prath/gemini_agents"):
    FILES_TO_COPY = ["app.py", "deploy_to_pi.py", "other_module.py"]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_IP, username=PI_USER, password=PI_PASSWORD)

    ssh.exec_command(f"mkdir -p {REMOTE_PATH}")

    sftp = ssh.open_sftp()
    for f in FILES_TO_COPY:
        local_file = os.path.join(LOCAL_PATH, f)
        remote_file = os.path.join(REMOTE_PATH, f)
        if os.path.exists(local_file):
            print(f"Copying {local_file} -> {remote_file}")
            sftp.put(local_file, remote_file)
        else:
            print(f"WARNING: Local file not found: {local_file}")
    sftp.close()

    cmd = f"nohup python3 {REMOTE_PATH}/app.py > {REMOTE_PATH}/agent.log 2>&1 &"
    ssh.exec_command(cmd)
    print("Gemini agent started on Pi.")
    ssh.close()

# Optional: run directly when script is executed
if __name__ == "__main__":
    PI_IP = ""
    PI_USER = ""
    PI_PASSWORD = ""
    LOCAL_PATH = r"C:\Users\prath\OneDrive\lappy_data\projects\MAJOR PROJECT SHIT\Agentic-Ai\backend"
    deploy_latest_agent(PI_IP, PI_USER, PI_PASSWORD, LOCAL_PATH)