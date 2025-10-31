"""
Label Studio API Client
"""
import requests
from typing import Optional, Dict

class LabelStudioClient:
    """Client to interact with Label Studio API"""
    
    def __init__(self, base_url: str, api_key: str):
        """Initialize Label Studio client"""
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_task(self, task_id: int) -> Optional[Dict]:
        """Fetch a task from Label Studio (only ground_truth annotation)"""
        url = f"{self.base_url}/api/tasks/{task_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            raw_data = response.json()
            transformed_data = self._transform_task_data(raw_data)
            
            return transformed_data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch task {task_id}: {str(e)}")
    
    def _transform_task_data(self, raw_data: Dict) -> Dict:
        """Transform Label Studio response (only ground_truth annotation)"""
        
        ground_truth_annotation = None
        
        for annotation in raw_data.get("annotations", []):
            if annotation.get("ground_truth", False):
                ground_truth_annotation = annotation
                break
        
        transformed = {
            "id": raw_data["id"],
            "data": raw_data["data"],
            "annotations": [ground_truth_annotation] if ground_truth_annotation else []
        }
        
        return transformed
    
    def update_task_annotation(self, task_id: int, annotation_id: int, 
                               updated_ranker: Dict) -> bool:
        """Update task annotation with new labels (PATCH)"""
        url = f"{self.base_url}/api/tasks/{task_id}/annotations/{annotation_id}"
        
        payload = {
            "result": [
                {
                    "value": {
                        "ranker": updated_ranker
                    },
                    "from_name": "rank",
                    "to_name": "results",
                    "type": "ranker"
                }
            ]
        }
        
        try:
            response = requests.patch(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to update annotation: {str(e)}")
    
    def create_new_annotation(self, task_id: int, ranker: Dict, ground_truth: bool = False) -> Dict:
        """Create a new annotation for a task (POST)"""
        url = f"{self.base_url}/api/tasks/{task_id}/annotations/"
        
        payload = {
            "result": [
                {
                    "value": {
                        "ranker": ranker
                    },
                    "from_name": "rank",
                    "to_name": "results",
                    "type": "ranker"
                }
            ],
            "ground_truth": ground_truth
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to create annotation: {str(e)}")
