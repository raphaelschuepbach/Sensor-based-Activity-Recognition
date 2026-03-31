param(
    [string]$msg="update data",
    [string]$remote="storage_local"
)

Write-Host "Adding DVC changes..."
dvc add Daten

Write-Host "Adding Git files..."
git add .
git add Daten.dvc
git add .gitignore

Write-Host "Committing..."
git commit -m $msg

Write-Host "Pushing data to DVC remote..."
dvc push -r $remote

Write-Host "Pushing code to Git..."
git push

Write-Host "Done."