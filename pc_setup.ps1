# One-time setup on the 4060 Ti PC, so the laptop can use it over SSH.
# Run in an ADMIN PowerShell on the PC. Safe to run twice.
# See SSH_SETUP.md for what each part does.

$ErrorActionPreference = "Stop"

# The laptop's PUBLIC key (safe to share; the private half never leaves the laptop)
$pub = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOCCNDFut9ZL7SvUgfqadHSJjBhYjwbi0zPINa65ABoD laptop-to-gpupc"

Write-Host "1/5 Installing OpenSSH Server (can take a few minutes)..."
if ((Get-WindowsCapability -Online -Name "OpenSSH.Server*").State -ne "Installed") {
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
}
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

Write-Host "2/5 Firewall rule (private/home networks only)..."
if (-not (Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH Server (sshd)" -Enabled True `
        -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -Profile Private | Out-Null
}

Write-Host "3/5 PowerShell as the remote shell..."
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
    -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force | Out-Null

Write-Host "4/5 Adding the laptop's key (no password needed after this)..."
# Admin accounts read this file (a Windows quirk); it must be owned by Administrators + SYSTEM only.
$adminKeys = "$env:ProgramData\ssh\administrators_authorized_keys"
if (-not (Test-Path $adminKeys) -or -not (Select-String -Path $adminKeys -SimpleMatch $pub -Quiet)) {
    Add-Content -Path $adminKeys -Value $pub
}
icacls.exe $adminKeys /inheritance:r /grant "*S-1-5-32-544:F" /grant "*S-1-5-18:F" | Out-Null
# Non-admin accounts read this one; writing both covers either case.
$userKeys = "$HOME\.ssh\authorized_keys"
New-Item -ItemType Directory -Force -Path "$HOME\.ssh" | Out-Null
if (-not (Test-Path $userKeys) -or -not (Select-String -Path $userKeys -SimpleMatch $pub -Quiet)) {
    Add-Content -Path $userKeys -Value $pub
}
Restart-Service sshd

Write-Host "5/5 No sleep or hibernate while plugged in..."
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

$ips = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
    $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" }).IPAddress -join ", "
$profiles = (Get-NetConnectionProfile | ForEach-Object { "$($_.Name)=$($_.NetworkCategory)" }) -join ", "
Write-Host ""
Write-Host "DONE. Send these lines to Claude on the laptop:" -ForegroundColor Green
Write-Host "  PC name:  $env:COMPUTERNAME"
Write-Host "  User:     $env:USERNAME"
Write-Host "  IP:       $ips"
Write-Host "  Network:  $profiles   (must say Private)"
$gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }).Name
Write-Host "  GPU:      $gpu"
