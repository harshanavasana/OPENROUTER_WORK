import os
import json
from typing import Dict, List, Optional
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore, auth

def init_firebase():
    """Initialize Firebase Admin SDK using a service account."""
    if not firebase_admin._apps:
        # Load service account path from env or use default
        cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT", "serviceAccountKey.json")
        try:
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            else:
                # In Streamlit Cloud, you might load it from a dictionary in secrets
                print(f"Warning: Firebase Service account key not found at {cred_path}. Firebase Admin might not work if not using default credentials.")
                # We initialize with application default credentials as fallback
                firebase_admin.initialize_app(options={"projectId": os.getenv("FIREBASE_PROJECT_ID", "open-router-64d3c")})
        except Exception as e:
            print(f"Firebase initialization failed: {e}")

# Call init on module load
init_firebase()

def get_db():
    try:
        return firestore.client()
    except Exception as e:
        print(f"Firestore Client could not be initialized: {e}")
        return None

def verify_id_token(id_token: str) -> Optional[dict]:
    """Verify Firebase ID token and return decoded claims."""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

def create_chat_session(uid: str, title: str = "New Chat") -> str:
    """Creates a new chat session and returns its ID."""
    try:
        db = get_db()
        if not db: return "local-session-only"
        session_ref = db.collection("users").document(uid).collection("sessions").document()
        session_ref.set({
            "title": title,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        return session_ref.id
    except Exception as e:
        print(f"Failed to create session: {e}")
        return "local-session-only"

def add_message_to_session(uid: str, session_id: str, role: str, content: str):
    """Add a message to an existing chat session."""
    if session_id == "local-session-only": return
    try:
        db = get_db()
        if not db: return
        messages_ref = db.collection("users").document(uid).collection("sessions").document(session_id).collection("messages")
        messages_ref.add({
            "role": role,
            "content": content,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        # Update session's updated_at
        db.collection("users").document(uid).collection("sessions").document(session_id).update({
            "updated_at": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Failed to add message: {e}")

def get_user_sessions(uid: str) -> List[dict]:
    """Get all chat sessions for a user, ordered by updated_at descending."""
    try:
        db = get_db()
        if not db: return []
        sessions = db.collection("users").document(uid).collection("sessions").order_by("updated_at", direction=firestore.Query.DESCENDING).get()
        
        result = []
        for s in sessions:
            data = s.to_dict()
            data["id"] = s.id
            result.append(data)
        return result
    except Exception as e:
        print(f"Failed to load sessions (missing credentials?): {e}")
        return []

def get_session_messages(uid: str, session_id: str) -> List[dict]:
    """Get all messages in a session, ordered by timestamp ascending."""
    if session_id == "local-session-only": return []
    try:
        db = get_db()
        if not db: return []
        messages = db.collection("users").document(uid).collection("sessions").document(session_id).collection("messages").order_by("timestamp").get()
        return [m.to_dict() for m in messages]
    except Exception as e:
        print(f"Failed to get messages: {e}")
        return []

def save_user_api_keys(uid: str, api_keys: Dict[str, str]):
    """Save user-provided API keys (model_id -> key) to Firestore."""
    try:
        db = get_db()
        if not db: return
        db.collection("users").document(uid).set({"api_keys": api_keys}, merge=True)
    except Exception as e:
        print(f"Failed to save user API keys (missing credentials?): {e}")

def get_user_api_keys(uid: str) -> Dict[str, str]:
    """Get user-provided API keys from Firestore."""
    try:
        db = get_db()
        if not db: return {}
        doc = db.collection("users").document(uid).get()
        if doc.exists:
            return doc.to_dict().get("api_keys", {})
        return {}
    except Exception as e:
        print(f"Failed to get user API keys (missing credentials?): {e}")
        return {}
