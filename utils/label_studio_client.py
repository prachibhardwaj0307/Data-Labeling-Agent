"""
Label Studio Client - Fetches documents from Label Studio API
"""
import os
from typing import Dict, Any, Optional
from label_studio_sdk import LabelStudio


class LabelStudioClient:
    """
    Client for interacting with Label Studio API
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize Label Studio client
        
        Args:
            api_key: Label Studio API key (default from env var)
            base_url: Label Studio instance URL (default: http://localhost:8080)
        """
        self.api_key = api_key or os.getenv("LABEL_STUDIO_API_KEY")
        self.base_url = base_url or os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
        
        if not self.api_key:
            raise ValueError(
                "Label Studio API key not found. "
                "Set LABEL_STUDIO_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize Label Studio client
        self.client = LabelStudio(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"✓ Label Studio client initialized (URL: {self.base_url})")
    
    def get_task(self, task_id: int) -> Dict[str, Any]:
        """
        Fetch a single task from Label Studio by ID
        
        Args:
            task_id: Task ID in Label Studio
            
        Returns:
            Task data dictionary in the format expected by your system
        """
        try:
            # Fetch task from Label Studio
            task = self.client.tasks.get(id=task_id)
            
            # Convert to dict if needed
            if hasattr(task, 'dict'):
                task_data = task.dict()
            elif hasattr(task, 'to_dict'):
                task_data = task.to_dict()
            else:
                task_data = task
            
            print(f"✓ Fetched task {task_id} from Label Studio")
            
            # Transform to your system's format
            return self._transform_task_data(task_data, task_id)
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch task {task_id} from Label Studio: {e}")
    
    def _transform_task_data(self, task_data: Dict[str, Any], task_id: int) -> Dict[str, Any]:
        """
        Transform Label Studio task data to your system's expected format
        
        Args:
            task_data: Raw task data from Label Studio
            task_id: Task ID
            
        Returns:
            Transformed data in your system's format
        """
        # Extract the main data from Label Studio task
        data = task_data.get("data", {})
        annotations = task_data.get("annotations", [])
        
        # Build the expected format
        transformed = {
            "id": task_id,
            "data": {
                "items": data.get("items", []),
                "text": data.get("text", ""),
                "location": data.get("location", ""),
                "organisation": data.get("organisation", ""),
                "location_intent": data.get("location_intent", ""),
                "recency_intent": data.get("recency_intent", "")
            },
            "annotations": annotations if annotations else [
                {
                    "result": [
                        {
                            "value": {
                                "ranker": {
                                    "relevant": [],
                                    "somewhat_relevant": [],
                                    "acceptable": [],
                                    "irrelevant": [],
                                    "not_sure": [],
                                    "New Doc": [item.get("id") for item in data.get("items", [])]
                                }
                            }
                        }
                    ]
                }
            ]
        }
        
        return transformed
    
    def update_task_annotations(self, task_id: int, annotations: Dict[str, Any]) -> bool:
        """
        Update task annotations in Label Studio with labeling results
        
        Args:
            task_id: Task ID in Label Studio
            annotations: Updated annotations/labels
            
        Returns:
            True if successful
        """
        try:
            # Update the task with new annotations
            self.client.tasks.update(
                id=task_id,
                annotations=[{
                    "result": [{
                        "value": {
                            "ranker": annotations
                        }
                    }]
                }]
            )
            
            print(f"✓ Updated annotations for task {task_id} in Label Studio")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update task {task_id}: {e}")
            return False
    
    def get_project_tasks(self, project_id: int, limit: int = 100) -> list:
        """
        Fetch all tasks from a Label Studio project
        
        Args:
            project_id: Project ID in Label Studio
            limit: Maximum number of tasks to fetch
            
        Returns:
            List of task IDs
        """
        try:
            tasks = self.client.projects.list_tasks(
                id=project_id,
                page_size=limit
            )
            
            task_ids = [task.id for task in tasks]
            print(f"✓ Found {len(task_ids)} tasks in project {project_id}")
            
            return task_ids
            
        except Exception as e:
            print(f"❌ Failed to fetch tasks from project {project_id}: {e}")
            return []
