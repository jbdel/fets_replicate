import os
import paramiko


def open_communication(central_path):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    while True:
        try:
            ssh_client.connect(hostname="epad-public.stanford.edu", username="distributed",
                               password="Ch@ngeMe")
            break
        except:
            print('Wrong password')

    os.makedirs(central_path, exist_ok=True)

    try:
        sftp = ssh_client.open_sftp()
        sftp.mkdir(central_path)
        print('Generate central folder for saving models and training progress sync_federation')
    except:
        pass

    return ssh_client


def put_file(ssh_client, central_path, file):
    ftp_client = ssh_client.open_sftp()
    ftp_client.put(os.path.join(central_path, file), os.path.join(central_path, file))
    ftp_client.close()

def get_file(ssh_client, central_path, file):
    ftp_client = ssh_client.open_sftp()
    ftp_client.get(os.path.join(central_path, file), os.path.join('.', central_path, file))
    ftp_client.close()
