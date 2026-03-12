param(
    [string]$msg="update data",
    [string]$remote="storage_local"
)

# --- DVC Änderungen hinzufügen ---
Write-Host "Adding DVC changes..."
dvc add Daten

# --- Git Änderungen hinzufügen ---
Write-Host "Adding Git files..."
git add Daten.dvc
git add .gitignore
# git add .dvc/config  <-- NICHT mehr hinzufügen

# --- Git Commit ---
Write-Host "Committing..."
git commit -m $msg

# --- DVC Push ---
Write-Host "Pushing data to DVC remote..."
dvc push -r $remote

# --- Git Push ---
Write-Host "Pushing code to Git..."
git push

Write-Host "Done."