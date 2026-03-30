from pathlib import Path
import streamlit.components.v1 as components

component_dir = str((Path(__file__).parent / "auth_component").resolve())
_component_func = components.declare_component(
    "firebase_auth_component",
    path=component_dir
)

def firebase_auth(key=None):
    """
    Renders the Firebase UI Google & Email/Password login.
    Returns a dict with {"token": ..., "uid": ..., "email": ..., "displayName": ...} upon successful login,
    or None if not logged in.
    """
    return _component_func(key=key, default=None)
