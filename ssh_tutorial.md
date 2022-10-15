# How to Connect VS Code on Windows to the CoE 197Z SSH Server
2022-OCT-15 Vash Patrick Ancheta

This tutorial assumes that you already have VS Code installed. If you haven't yet, you can visit the official download page: https://code.visualstudio.com/download.

## Step 1: Install OpenSSH Client
 1. [Run PowerShell as an Administrator](https://learn.microsoft.com/en-us/powershell/scripting/windows-powershell/starting-windows-powershell?view=powershell-7.2#with-administrative-privileges-run-as-administrator).

 2. Input the following command to check if OpenSSH Client is already installed: `Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH*'`. The output should look like this if OpenSSH Client is installed, otherwise, move to step 3:
 ```
 Name  : OpenSSH.Client~~~~0.0.1.0
 State : Installed

 Name  : OpenSSH.Server~~~~0.0.1.0
 State : NotPresent
 ```

 3. To install OpenSSH Client, input the following command: `Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0`. You might need to restart your computer for changes to take place.

## Step 2: Connect to SSH using PowerShell to Change Password
For this step, you no longer need PowerShell to be ran as Administrator.
 1. Figure out your username. Sir Rowel set it to your surname.

 2. Figure out your password. Sir Rowel set it to f"{surname}{X}{Y}{Z}" (e.g. "mysurname123") where X, Y, and Z are the last three digits of your student number.

 3. Input the following command to your PowerShell instance: `ssh your_username@104.171.201.119`. Make sure to replace "your_username" with your actual username. You'll get the following output:
 ```
 The authenticity of host '104.171.201.119 (104.171.201.119)' can't be established.
 ECDSA key fingerprint is SHA256:6sMIQjgiIPnMUQ+NcQPhXEHe0aOLsSaNuAtWfNx+3io.
 Are you sure you want to continue connecting (yes/no/[fingerprint])?
 ```

 4. Type "yes" and press Enter. You'll get the following output:
 ```
 Warning: Permanently added '104.171.201.119' (ECDSA) to the list of known hosts.
 your_username@104.171.201.119's password:
 ```

 5. Type your password and press Enter. You'll get some system information ending with the following output. You're in!
 ```
 your_username@lambda-hyperplane:~$
 ```

 6. Enter the following command: `passwd`. You'll get the following output:
 ```
 Changing password for your_username.
 Current password:
 ```

 7. Type your current password and press Enter. You'll get the following prompt:
 ```
 New password:
 ```

 8. Type your new password and press Enter. You'll get the following prompt:
 ```
 Retype new password:
 ```

 9. Retype your new password and press Enter. You'll get the following output:
 ```
 passwd: password updated successfully
 ```

 10. Type the following command: `logout`. You'll get the following output:
 ```
 Connection to 104.171.201.119 closed.
 ```

## Step 3: Install the Remote - SSH Extension on VS Code
 1. Use VS Code to install the [Remote - SSH Extension](https://code.visualstudio.com/docs/remote/ssh-tutorial#_install-the-extension) by Microsoft.

 2. On VS Code, you'll see a new icon on the bottom left of the window. Click it.
 ![SSH Extension button](https://code.visualstudio.com/assets/docs/remote/ssh-tutorial/remote-status-bar.png)

 3. Select "Connect to Host..."
 ![SSH Extension prompt](https://code.visualstudio.com/assets/docs/remote/ssh-tutorial/remote-ssh-commands.png)

 4. Select "Add New SSH Host..."

 5. Input `ssh your_username@104.171.201.119` and press Enter.

 6. Select "C:\\Users\\your_windows_username\\.ssh\\config".

 7. You'll see a prompt to confirm adding the host. You can click connect or use what we did in 3 to connect to the host we added.
 ![select host](https://i.postimg.cc/d0zLnxvH/host.png)

 8. It will ask for the platform of the host. Select "Linux".

 9. It will ask for your SSH password. Enter the password you set earlier. After that, you will see the following window:
 ![connected window](https://i.postimg.cc/W4PM1zqb/connected-vscode.png)

## Step 4: Print Device Name via Python
 1. Click on the Open Folder button.
 ![connected window](https://i.postimg.cc/W4PM1zqb/connected-vscode.png)

 2. Enter `/home/your_username`. Make sure to use your username.
 ![home folder](https://i.postimg.cc/yNdvHWMg/home-folder.png)

 3. It will ask for your password. Enter it again.

 4. Create a new file. Name it `hello_world.py`.
 ![create file](https://i.postimg.cc/rw8L5CrB/new-file.png)
 ![file name](https://i.postimg.cc/5y78pmts/file-name.png)

 5. Write the following code inside `hello_world.py`:
 ```
 import platform
 print(f"Hello {platform.node()}!")
 ```
 ![file contents](https://i.postimg.cc/6QnSS8nr/file-content.png)

 6. Open up a terminal on the server by pressing Ctrl+Shift+\`. Type the following command and press Enter: `python hello_world.py`. You should get the following output:
 ```
 Hello lambda-hyperplane!
 ```

## Step 5: Clone Git Repository
This is to follow, after our boilerplate code is complete.