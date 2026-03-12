param(
    [string]$remote="storage_local"
)

Write-Host "Pulling latest Git changes..."
git pull

Write-Host "Pulling latest DVC data from remote..."
dvc pull -r $remote

Write-Host "Done. Your data should now be up-to-date!"