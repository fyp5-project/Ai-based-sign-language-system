# app/gdrive_utils.py
import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import shutil # For shutil.which to check for ffmpeg

# SCOPES for Google Drive API (read-only is often sufficient for downloading)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_gdrive_service(service_account_key_path: str):
    """Authenticates and returns a Google Drive service object."""
    if not os.path.exists(service_account_key_path):
        print(f"[GDRIVE ERROR] Service account key file not found at: {service_account_key_path}")
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(
            service_account_key_path, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds, cache_discovery=False) # Disable discovery cache for serverless
        print("[GDRIVE INFO] Successfully authenticated with Google Drive API.")
        return service
    except Exception as e:
        print(f"[GDRIVE ERROR] Failed to authenticate/build Google Drive service: {e}")
        return None

def download_file_from_drive(service, file_id: str, local_download_path: str):
    """Downloads a file from Google Drive given its file ID."""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(local_download_path), exist_ok=True)

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        print(f"[GDRIVE INFO] Starting download for file ID: {file_id} to {local_download_path}")
        while done is False:
            status, done = downloader.next_chunk()
            if status:
                print(f"[GDRIVE DEBUG] Download {int(status.progress() * 100)}%.")
        
        fh.seek(0)
        with open(local_download_path, 'wb') as f:
            f.write(fh.read())
        print(f"[GDRIVE INFO] File downloaded successfully: {local_download_path}")
        return True
    except Exception as e:
        print(f"[GDRIVE ERROR] Failed to download file ID {file_id}: {e}")
        return False

def find_file_id_in_folder(service, folder_id: str, file_name: str) -> Optional[str]:
    """Finds a file's ID within a specific Google Drive folder."""
    try:
        query = f"'{folder_id}' in parents and name = '{file_name}' and trashed = false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])
        if files:
            return files[0].get('id')
        else:
            print(f"[GDRIVE WARNING] File '{file_name}' not found in folder ID '{folder_id}'.")
            return None
    except Exception as e:
        print(f"[GDRIVE ERROR] Error finding file '{file_name}' in folder '{folder_id}': {e}")
        return None

def download_folder_contents_from_drive(service, folder_id: str, local_target_dir: str, specific_files: Optional[list] = None):
    """
    Downloads files from a Google Drive folder to a local directory.
    If specific_files is provided, only those files are downloaded.
    Otherwise, attempts to download all files (can be slow for many files).
    """
    os.makedirs(local_target_dir, exist_ok=True)
    downloaded_any = False
    try:
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, pageSize=1000, # Adjust pageSize as needed
                                       fields="nextPageToken, files(id, name, mimeType)").execute()
        items = results.get('files', [])

        if not items:
            print(f"[GDRIVE WARNING] No files found in folder ID '{folder_id}'.")
            return False
        
        for item in items:
            file_id = item['id']
            file_name = item['name']
            mime_type = item['mimeType']

            # Skip Google Drive's own folder type
            if mime_type == 'application/vnd.google-apps.folder':
                print(f"[GDRIVE INFO] Skipping subfolder '{file_name}' during flat download.")
                continue

            if specific_files and file_name not in specific_files:
                continue # Skip if not in the specific list

            local_file_path = os.path.join(local_target_dir, file_name)
            
            # Avoid re-downloading if file already exists and has size (simple check)
            if os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
                print(f"[GDRIVE INFO] File '{file_name}' already exists locally. Skipping download.")
                downloaded_any = True # Count as success if already there
                continue

            print(f"[GDRIVE INFO] Attempting to download '{file_name}' (ID: {file_id}) to '{local_file_path}'")
            if download_file_from_drive(service, file_id, local_file_path):
                downloaded_any = True
            else:
                print(f"[GDRIVE WARNING] Failed to download '{file_name}'.")
        return downloaded_any # True if at least one file was successfully handled (downloaded or skipped)
    except Exception as e:
        print(f"[GDRIVE ERROR] Error listing/downloading folder '{folder_id}': {e}")
        return False

