"""
Test Label Studio connection
"""
import os
from dotenv import load_dotenv
load_dotenv()

from utils.label_studio_client import LabelStudioClient

# Check environment variables
print("Checking environment variables...")
api_key = os.getenv("LABEL_STUDIO_API_KEY")
url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")

print(f"API Key: {'✓ Found' if api_key else '✗ Missing'}")
print(f"URL: {url}")

if not api_key:
    print("\n❌ LABEL_STUDIO_API_KEY not set!")
    print("Add it to your .env file:")
    print("LABEL_STUDIO_API_KEY=your-token-here")
    exit(1)

# Try to connect
print("\nTesting Label Studio connection...")
try:
    client = LabelStudioClient()
    print("✓ Client initialized successfully")
    
    # Try to fetch a task
    task_id = 35566  # Change this to a valid task ID
    print(f"\nFetching task {task_id}...")
    
    task = client.get_task(task_id)
    print("✓ Task fetched successfully!")
    print(f"\nTask data preview:")
    print(f"  Query: {task.get('data', {}).get('text', 'N/A')}")
    print(f"  Location: {task.get('data', {}).get('location', 'N/A')}")
    print(f"  Items: {len(task.get('data', {}).get('items', []))} documents")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
