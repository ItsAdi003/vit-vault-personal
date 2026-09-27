"""
Process Route — The Main AI Processing Endpoint
-------------------------------------------------
POST /api/process
  Receives: { "media_id": "uuid", "file_path": "storage/path" }
  
  Pipeline:
  1. Downloads the image from Supabase Storage
  2. Runs Duplicate Detection (pHash)
  3. Runs Auto-Tagging (CLIP zero-shot)
  4. Generates Semantic Embedding (CLIP 512-dim vector)
  5. Updates the media row in Supabase with all AI results

POST /api/search
  Receives: { "query": "people dancing on stage" }
  Returns: Array of matching media IDs with similarity scores
"""
import io
import requests
from PIL import Image
from flask import Blueprint, request, jsonify

from config import supabase, SUPABASE_URL, SUPABASE_SERVICE_KEY
from services.tagger import generate_tags
from services.embedder import generate_embedding, generate_text_embedding
from services.deduplicator import generate_hash, check_duplicate
from services.classifier import classify_category

process_bp = Blueprint("process", __name__)


@process_bp.route("/api/process", methods=["POST"])
def process_media():
    """
    Main AI processing endpoint.
    Called by the React frontend after a user uploads an image.
    """
    data = request.get_json()
    
    if not data or "media_id" not in data or "file_path" not in data:
        return jsonify({"error": "Missing media_id or file_path"}), 400
    
    media_id = data["media_id"]
    file_path = data["file_path"]
    
    print(f"\n{'='*60}")
    print(f"[AI] Processing media: {media_id}")
    print(f"[AI] File path: {file_path}")
    print(f"{'='*60}")
    
    try:
        # ─── Step 1: Download the image from Supabase Storage ───
        print("[AI] Step 1/4: Downloading image from Supabase Storage...")
        
        # Build the download URL using Supabase Storage API
        download_url = f"{SUPABASE_URL}/storage/v1/object/vault-media/{file_path}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY
        }
        
        response = requests.get(download_url, headers=headers)
        
        if response.status_code != 200:
            print(f"[AI] ✗ Failed to download image: HTTP {response.status_code}")
            return jsonify({"error": f"Failed to download image: {response.status_code}"}), 500
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        print(f"[AI] ✓ Image downloaded ({image.width}x{image.height})")
        
        # ─── Step 2: Duplicate Detection ───
        print("[AI] Step 2/4: Running duplicate detection (pHash)...")
        
        phash = generate_hash(image)
        dup_result = check_duplicate(phash, exclude_media_id=media_id)
        
        if dup_result["is_duplicate"]:
            # Mark as duplicate in database and REJECT it so UI hides it
            supabase.table("media").update({
                "perception_hash": str(phash),
                "ai_tags": ["duplicate"],
                "status": "REJECTED",
                "metadata": {
                    "duplicate_of": str(dup_result["matching_id"]), 
                    "distance": int(dup_result["distance"])
                }
            }).eq("id", media_id).execute()
            
            return jsonify({
                "status": "duplicate",
                "message": "This image is a duplicate of an existing upload",
                "matching_id": str(dup_result["matching_id"]),
                "distance": int(dup_result["distance"])
            }), 200
        
        # ─── Step 3: Auto-Tagging ───
        print("[AI] Step 3/5: Running auto-tagging (CLIP)...")
        
        tags = generate_tags(image)
        
        # ─── Step 3b: Naive Bayes Category Classification (PGM) ───
        print("[AI] Step 4/5: Running Naive Bayes category classifier (PGM)...")
        
        classification = classify_category(tags)
        
        # ─── Step 5: Semantic Embedding ───
        print("[AI] Step 5/5: Generating semantic embedding (512-dim vector)...")
        
        embedding = generate_embedding(image)
        
        # ─── Save All Results to Database ───
        print("[AI] Saving all AI results to Supabase...")
        
        update_data = {
            "ai_tags": tags,
            "perception_hash": str(phash),
            "status": "APPROVED",
            "metadata": {
                "predicted_category": classification["predicted_category"],
                "category_confidence": classification["confidence"],
                "category_posteriors": classification["posteriors"]
            }
        }
        
        # pgvector is enabled — save embedding for semantic search
        update_data["image_embedding"] = embedding
        
        supabase.table("media").update(update_data).eq("id", media_id).execute()
        
        print(f"\n[AI] ════════════════════════════════════════")
        print(f"[AI] ✓ Processing complete for {media_id}")
        print(f"[AI]   Tags:  {tags}")
        print(f"[AI]   Category: {classification['predicted_category']} ({classification['confidence']:.1%})")
        print(f"[AI]   Hash:  {phash}")
        print(f"[AI]   Embed: {len(embedding)}-dim vector generated")
        print(f"[AI] ════════════════════════════════════════\n")
        
        return jsonify({
            "status": "success",
            "media_id": media_id,
            "ai_tags": tags,
            "classification": classification,
            "perception_hash": phash,
            "embedding_dimensions": len(embedding)
        }), 200
        
    except Exception as e:
        print(f"[AI] ✗ Error processing media: {str(e)}")
        return jsonify({"error": str(e)}), 500


@process_bp.route("/api/search", methods=["POST"])
def semantic_search():
    """
    Semantic search endpoint.
    Called by the Gallery search bar for AI-powered image search.
    """
    data = request.get_json()
    
    if not data or "query" not in data:
        return jsonify({"error": "Missing query"}), 400
    
    query = data["query"]
    print(f"[AI] Semantic search: '{query}'")
    
    try:
        # Convert the text query to a vector
        text_embedding = generate_text_embedding(query)
        
        # For now, we return the embedding so the frontend can do client-side comparison
        # Later, when pgvector is enabled, we'll do server-side cosine similarity
        return jsonify({
            "status": "success",
            "query": query,
            "embedding": text_embedding,
            "dimensions": len(text_embedding)
        }), 200
        
    except Exception as e:
        print(f"[AI] ✗ Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@process_bp.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check to verify the AI server is running."""
    return jsonify({
        "status": "healthy",
        "service": "VIT Media Vault AI Engine",
        "model": "openai/clip-vit-base-patch32",
        "features": ["auto-tagging", "duplicate-detection", "semantic-search", "naive-bayes-classifier"]
    }), 200
