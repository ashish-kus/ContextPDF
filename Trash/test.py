# main.py

from dotenv import load_dotenv
import os

# ... other imports ...

load_dotenv()

# --- Debugging Code ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "Error: GOOGLE_API_KEY was not loaded from the .env file. Check the file location and format."
    )
    st.stop()  # Stops the app from running further if the key is missing
else:
    print(f"API Key successfully loaded (length: {len(api_key)})")
# --- End Debugging Code ---

# ... rest of your main function ...
