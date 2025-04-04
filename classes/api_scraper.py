import os
import sys
import requests
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

class NodeREDApiScraper:
    """
    Scraper for Node-RED API endpoints to get collection and flow data
    """
    
    def __init__(self):
        self.results = {
            "parsed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "collections": [],
            "flows": []
        }
        self.base_url = "https://flows.nodered.org"
        self.api_url = "https://flows.nodered.org/api"
    
    def scrape_collection(self, collection_id: str) -> Dict[str, Any]:
        """
        Get collection data directly from API
        
        Args:
            collection_id: ID of the collection to scrape
            
        Returns:
            Dictionary with collection data
        """
        print(f"Fetching collection data for ID: {collection_id}")
        
        # Get collection metadata
        collection_url = f"{self.api_url}/collection/{collection_id}"
        collection_data = self._fetch_json(collection_url)
        
        if not collection_data or "collection" not in collection_data:
            print(f"Error: Could not fetch collection data for {collection_id}")
            return self.results
        
        # Process collection metadata
        collection_info = collection_data.get("collection", {})
        
        # Format collection data
        formatted_collection = {
            "id": collection_info.get("_id"),
            "title": collection_info.get("name"),
            "description": collection_info.get("description", ""),
            "owners": [collection_info.get("owner", {}).get("name", "Unknown Owner")],
            "url": f"{self.base_url}/collection/{collection_id}",
            "updated": collection_info.get("updated_at")
        }
        
        self.results["collections"].append(formatted_collection)
        
        # Get flow list from the collection
        flow_ids = collection_info.get("items", [])
        flows = []
        
        print(f"Found {len(flow_ids)} flows in collection")
        
        # Process each flow
        for flow_id in flow_ids:
            flow_data = self.scrape_flow(flow_id)
            flow_data["collection_id"] = collection_id
            flows.append(flow_data)
            
        return {
            "collection": formatted_collection,
            "flows": flows
        }
    
    def scrape_flow(self, flow_id: str) -> Dict[str, Any]:
        """
        Get flow data directly from API
        
        Args:
            flow_id: ID of the flow to scrape
            
        Returns:
            Dictionary with flow data
        """
        print(f"Fetching flow data for ID: {flow_id}")
        
        # Get flow metadata
        flow_url = f"{self.api_url}/flow/{flow_id}"
        flow_data = self._fetch_json(flow_url)
        
        if not flow_data or "flow" not in flow_data:
            print(f"Error: Could not fetch flow data for {flow_id}")
            return {"id": flow_id, "error": "Failed to fetch flow data"}
        
        # Process flow metadata
        flow_info = flow_data.get("flow", {})
        
        # Format flow data
        formatted_flow = {
            "id": flow_id,
            "title": flow_info.get("description", "Untitled Flow"),
            "description": flow_info.get("long_description", ""),
            "owner": flow_info.get("owner", {}).get("name", "Unknown"),
            "tags": flow_info.get("tags", []),
            "updated": flow_info.get("updated_at"),
            "url": f"{self.base_url}/flow/{flow_id}"
        }
        
        # Add to results if not already present
        if not any(flow["id"] == flow_id for flow in self.results["flows"]):
            self.results["flows"].append(formatted_flow)
        
        return formatted_flow
    
    def _fetch_json(self, url: str) -> Dict[str, Any]:
        """
        Fetch JSON data from URL
        
        Args:
            url: URL to fetch JSON from
            
        Returns:
            Dictionary with JSON data
        """
        try:
            response = requests.get(url, headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 Node-RED Scraper"
            })
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return {}
    
    def print_json(self):
        """Print results as JSON"""
        print(json.dumps(self.results, indent=2))

def extract_id_from_url(url: str) -> Optional[str]:
    """Extract ID from URL"""
    import re
    if "/collection/" in url:
        match = re.search(r'/collection/([^/]+)', url)
        if match:
            return match.group(1)
    elif "/flow/" in url:
        match = re.search(r'/flow/([^/]+)', url)
        if match:
            return match.group(1)
    return None

def main():
    """Main function to scrape Node-RED using API"""
    parser = argparse.ArgumentParser(description="Scrape Node-RED data via API")
    parser.add_argument("--url", required=True, help="URL of collection or flow to scrape")
    
    args = parser.parse_args()
    
    try:
        # Create scraper
        scraper = NodeREDApiScraper()
        
        # Extract ID from URL
        item_id = extract_id_from_url(args.url)
        
        if not item_id:
            print(f"Error: Could not extract ID from URL: {args.url}")
            return 1
        
        # Determine if it's a collection or flow
        if "/collection/" in args.url:
            print(f"Scraping collection with ID: {item_id}")
            scraper.scrape_collection(item_id)
        elif "/flow/" in args.url:
            print(f"Scraping flow with ID: {item_id}")
            scraper.scrape_flow(item_id)
        else:
            print(f"URL type unknown, trying as collection ID: {item_id}")
            scraper.scrape_collection(item_id)
        
        # Print results
        scraper.print_json()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 