param(
    [string]$msg="update data"
)

Write-Host "Adding DVC changes..."
dvc add Daten

Write-Host "Adding Git files..."
git add Daten.dvc
git add .dvc/config
git add .gitignore

Write-Host "Committing..."
git commit -m $msg

Write-Host "Pushing data to DVC remote..."
dvc push

Write-Host "Pushing code to Git..."
git push

Write-Host "Done."