"""
Reprocess all media records that have empty ai_tags.
Run this while the AI service (app.py) is running on port 5000.
"""
import requests
from config import supabase

def reprocess_untagged():
    result = supabase.table("media").select("id, file_path, ai_tags, perception_hash").eq("status", "APPROVED").execute()
    
    untagged = [r for r in result.data if not r.get("ai_tags") or len(r["ai_tags"]) == 0]
    
    print(f"Found {len(untagged)} untagged images to reprocess.\n")
    
    for i, media in enumerate(untagged):
        media_id = media["id"]
        file_path = media["file_path"]
        
        print(f"[{i+1}/{len(untagged)}] Processing {media_id}...")
        
        try:
            resp = requests.post(
                "http://localhost:5000/api/process",
                json={"media_id": media_id, "file_path": file_path},
                timeout=60
            )
            data = resp.json()
            
            if data.get("status") == "success":
                print(f"  OK - Tags: {data.get('ai_tags')}")
                print(f"  OK - Category: {data.get('classification', {}).get('predicted_category')}")
            elif data.get("status") == "duplicate":
                print(f"  DUPLICATE of {data.get('matching_id')}")
            else:
                print(f"  ERROR: {data.get('error')}")
        except Exception as e:
            print(f"  FAILED: {e}")
        
        print()
    
    print("Done! All images reprocessed.")

if __name__ == "__main__":
    reprocess_untagged()
