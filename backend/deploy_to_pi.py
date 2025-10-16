import paramiko
import os
import time
import select

def deploy_latest_agent(PI_IP, PI_USER, PI_PASSWORD, LOCAL_PATH, REMOTE_PATH="/home/prath/gemini_agents/generated_agents"):
    FILES_TO_COPY = ["agent.py"]  # only the agent that reads the sensor

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_IP, username=PI_USER, password=PI_PASSWORD)

    # Ensure remote folder exists
    ssh.exec_command(f"mkdir -p {REMOTE_PATH}")

    # Upload files
    sftp = ssh.open_sftp()
    for f in FILES_TO_COPY:
        local_file = os.path.join(LOCAL_PATH, f)
        remote_file = f"{REMOTE_PATH}/{f}"  # use forward slash
        if os.path.exists(local_file):
            print(f"Copying {local_file} -> {remote_file}")
            sftp.put(local_file, remote_file)
        else:
            print(f"WARNING: Local file not found: {local_file}")
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command(f"python3 {REMOTE_PATH}/agent.py")

    print("Gemini agent started on Pi. Reading output...\n")

    try:
        print("inside try")
        while True:
            print("inside while")
            # Use select to wait for data on stdout
            rl, wl, xl = select.select([stdout.channel], [], [], 0.1)
            if rl:
                print("inside first if")
                output = stdout.channel.recv(1024).decode()
                if output:
                    print(output, end="")

            # Check stderr if any
            if stderr.channel.recv_ready():
                err = stderr.channel.recv(1024).decode()
                if err:
                    print(err, end="")

            # Exit if process is done
            if stdout.channel.exit_status_ready():
                break

        # Print any remaining output
        print(stdout.read().decode(), end="")
        print(stderr.read().decode(), end="")

    except KeyboardInterrupt:
        print("\nStopping live output streaming.")
    finally:
        ssh.close()
        print("\nSSH connection closed.")

if __name__ == "__main__":
    PI_IP = ""  
    PI_USER = ""
    PI_PASSWORD = ""
    LOCAL_PATH = r""
    REMOTE_PATH = ""
    deploy_latest_agent(PI_IP, PI_USER, PI_PASSWORD, LOCAL_PATH, REMOTE_PATH)
