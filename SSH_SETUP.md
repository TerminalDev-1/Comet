# Use the 4060 Ti PC from the laptop (SSH setup)

**Goal:** sit at the PC once (~5 minutes), then train on its GPU from the laptop for good.

**You need:** both machines on the same home network. No password needed: the laptop's key is built into the setup script.

---

## Step 1: at the PC, one line (~5 min)

The laptop's key is already made, and its public half is built into [pc_setup.ps1](pc_setup.ps1). So there's **no password step**.

1. Right-click Start → **Terminal (Admin)**, or **Windows PowerShell (Admin)**.
2. Type this one line and press Enter:

```powershell
irm https://raw.githubusercontent.com/TerminalDev-1/Comet/main/pc_setup.ps1 | iex
```

It downloads the setup script from your own repo and runs it. That works without being signed in to GitHub, because the repo is public. The script:
- installs and starts the SSH server, and makes it start automatically with Windows
- allows SSH through the firewall, on private/home networks only
- makes PowerShell the remote shell
- adds the laptop's key (to the admin key file Windows uses, with the permissions it requires)
- turns off sleep and hibernate while plugged in

3. When it says **DONE**, send Claude the lines it prints (PC name, user, IP, network, GPU).
4. If "Network" doesn't say **Private**: Settings → Network & internet → your connection → Network profile type → Private.

**Prefer an agent?** Codex (or any coding agent) on the PC can do it too: ask it to run that one line in an **admin** terminal. Running the line yourself is just as quick.

That's all at the PC.

---

## Step 2: back on the laptop, Claude does the rest

Send Claude the printed lines. Claude adds an `ssh gpupc` shortcut and checks:

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
| Asks for a password | The key wasn't added. Run the step 1 line again on the PC (it's safe to re-run) |
| "Connection refused" | The SSH service isn't running. At the PC, in an admin PowerShell: `Start-Service sshd` |
| `nvidia-smi` not found | The NVIDIA driver needs an update on the PC (GeForce Experience or nvidia.com) |
