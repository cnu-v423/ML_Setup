
import os
import shutil
from datetime import datetime

def backup_project():
    """Backup entire project to Google Drive"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"/content/drive/MyDrive/building_detection_backup/backup_{timestamp}"
    
    print(f"Creating backup at: {backup_dir}")
    shutil.copytree("/content/building_detection", backup_dir)
    print("Backup completed!")
    return backup_dir

def restore_from_backup(backup_dir=None):
    """Restore project from backup"""
    if backup_dir is None:
        backup_base = "/content/drive/MyDrive/building_detection_backup"
        backups = sorted([d for d in os.listdir(backup_base) if d.startswith('backup_')])
        if not backups:
            print("No backups found!")
            return
        backup_dir = os.path.join(backup_base, backups[-1])
    
    print(f"Restoring from: {backup_dir}")
    if os.path.exists("/content/building_detection"):
        shutil.rmtree("/content/building_detection")
    shutil.copytree(backup_dir, "/content/building_detection")
    print("Restore completed!")

def list_backups():
    """List all available backups"""
    backup_base = "/content/drive/MyDrive/building_detection_backup"
    backups = sorted([d for d in os.listdir(backup_base) if d.startswith('backup_')])
    
    print("\nAvailable backups:")
    for i, backup in enumerate(backups, 1):
        backup_path = os.path.join(backup_base, backup)
        size = sum(d.stat().st_size for d in os.scandir(backup_path) if d.is_file())
        print(f"{i}. {backup} (Size: {size/1024/1024:.2f} MB)")
    return backups

def cleanup_old_backups(keep_last_n=5):
    """Keep only the n most recent backups"""
    backup_base = "/content/drive/MyDrive/building_detection_backup"
    backups = sorted([d for d in os.listdir(backup_base) if d.startswith('backup_')])
    
    if len(backups) > keep_last_n:
        for backup in backups[:-keep_last_n]:
            backup_path = os.path.join(backup_base, backup)
            print(f"Removing old backup: {backup}")
            shutil.rmtree(backup_path)
