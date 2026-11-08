# Use the 4060 Ti PC from the laptop (SSH setup)

**Goal:** sit at the PC once (~15 minutes), then train on its GPU from the laptop for good.

**You need:** both machines on the same home network, and the PC's Windows password (the account password, not the PIN).

---

## Step 1: at the PC, one paste (~5 min)

1. Right-click Start → **Terminal (Admin)**, or **Windows PowerShell (Admin)**.
2. Paste all of this and press Enter:

```powershell
# Install and start the SSH server, start it automatically with Windows
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

# Allow SSH through the firewall (home network only)
if (-not (Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH Server (sshd)" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -Profile Private
}

# Use PowerShell (not cmd) for remote sessions
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force

# Never sleep or hibernate while plugged in (a sleeping PC can't take commands)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

# Show what the laptop needs to know
"PC name:  $env:COMPUTERNAME"
"User:     $env:USERNAME"
"IP:       " + ((Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" }).IPAddress -join ", ")
```

3. **Write down the three lines it prints** (PC name, user, IP). The install line can take a few minutes.
4. Make sure the PC's network is set to **Private**: Settings → Network & internet → your Wi-Fi or Ethernet → Network profile type → Private.

That's all at the PC. Go back to the laptop.

---

## Step 2: on the laptop, Claude does this part

Tell Claude the PC name, user and IP. Claude will:
- create an SSH key for the PC: `~\.ssh\id_ed25519_gpupc`
- add a shortcut, so `ssh gpupc` just works.

You'll need the step 3 command that Claude gives you.

---

## Step 3: on the laptop, one command that you run (~1 min)

This copies the laptop's key to the PC, so logins never need a password again. **You** type the PC's Windows password once; Claude never sees it.

Claude will give you the exact command (it contains your key and user name). It works like this:
- It connects to the PC using your password, one time only.
- It saves the key where Windows expects it. For admin accounts that's `C:\ProgramData\ssh\administrators_authorized_keys`, not the usual Linux spot. This is a Windows quirk, and the command handles it.
- It sets the file permissions Windows requires, or the key gets ignored.

Run it in a terminal on the laptop, type the password when asked, done.

---

## Step 4: Claude tests and sets up the PC

Claude checks:
- `ssh gpupc` logs in with no password
- `nvidia-smi` shows the 4060 Ti

Then Claude installs Python, PyTorch with CUDA and the project on the PC, all over SSH. You don't need to be there.

---

## Every time after that

- **The PC just has to be on.** Nobody needs to be logged in at its screen.
- Training runs in the background on the PC. Claude starts runs so they keep going even if the laptop disconnects, posts ETAs, and copies results back.
- **Optional extras later:**
  - **Wake-on-LAN:** turn the PC on from the laptop. It needs a BIOS setting and a network card that supports it.
  - **Tailscale (free):** reach the PC from outside your home. Don't open SSH on your router to the internet.

## If something doesn't work

| Problem | Fix |
|---|---|
| `ssh gpupc` times out | The PC is off or asleep, or the network is set to Public (step 1.4). Try the IP instead of the PC name |
| Asks for a password every time | The key file permissions on the PC are wrong. Ask Claude to re-run step 3 |
| "Connection refused" | The SSH service isn't running. At the PC, in an admin PowerShell: `Start-Service sshd` |
| `nvidia-smi` not found | The NVIDIA driver needs an update on the PC (GeForce Experience or nvidia.com) |
