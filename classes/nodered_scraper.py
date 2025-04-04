import os
import sys
import requests
import json
import time
import re
from datetime import datetime
import argparse
from typing import Dict, Any, List, Optional, Union

class NodeREDScraper:
    """
    Scraper for Node-RED collections and flows using actual endpoints the website uses
    """
    
    def __init__(self):
        self.results = {
            "parsed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "collections": [],
            "flows": []
        }
        self.base_url = "https://flows.nodered.org"
        self.session = requests.Session()
        # Set headers to mimic a browser
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "X-Requested-With": "XMLHttpRequest"
        })
    
    def scrape_collection(self, collection_id: str) -> Dict[str, Any]:
        """
        Scrape a Node-RED collection
        
        Args:
            collection_id: ID of the collection
            
        Returns:
            Dictionary with collection data
        """
        print(f"Scraping collection with ID: {collection_id}")
        
        # First get the collection page to extract info
        collection_url = f"{self.base_url}/collection/{collection_id}"
        response = self.session.get(collection_url)
        if response.status_code != 200:
            print(f"Error: Failed to fetch collection page: {response.status_code}")
            return self.results
        
        # Extract collection title and description
        collection_html = response.text
        title_match = re.search(r'<h1 class="flow-title">(.*?)</h1>', collection_html)
        description_match = re.search(r'<div class="col-9-12 docs-content">.*?<p>(.*?)</p>', collection_html, re.DOTALL)
        updated_match = re.search(r'<div class="flowinfo">Updated (.*?)</div>', collection_html)
        owner_match = re.search(r'<div class="flowinfo"><a href="/user/(.*?)">(.*?)</a></div>', collection_html)
        
        # Format collection data
        collection_info = {
            "id": collection_id,
            "title": title_match.group(1).strip() if title_match else "Unknown Collection",
            "description": description_match.group(1).strip() if description_match else "",
            "updated": updated_match.group(1).strip() if updated_match else "",
            "owners": [owner_match.group(2).strip()] if owner_match else ["Unknown"],
            "url": collection_url
        }
        
        # Add to results
        self.results["collections"].append(collection_info)
        print(f"Parsed collection: {collection_info['title']}")
        
        # Now get the flows from the collection
        # The actual API endpoint used by the site for collection content
        api_url = f"{self.base_url}/things"
        params = {
            "format": "json",
            "collection": collection_id,
            "type": "flow",  # Get only flows
            "per_page": 50   # Get up to 50 items
        }
        
        response = self.session.get(api_url, params=params)
        if response.status_code != 200:
            print(f"Error: Failed to fetch collection items: {response.status_code}")
            return self.results
        
        try:
            collection_data = response.json()
            flows_data = collection_data.get("data", [])
            print(f"Found {len(flows_data)} flows in collection")
            
            # Process each flow
            flows = []
            for flow_data in flows_data:
                flow_id = flow_data.get("_id")
                if not flow_id:
                    continue
                    
                # Parse basic flow info from the list
                flow_info = {
                    "id": flow_id,
                    "title": flow_data.get("description", "Untitled Flow"),
                    "url": f"{self.base_url}/flow/{flow_id}",
                    "collection_id": collection_id,
                    "owner": flow_data.get("owner", {}).get("name", "Unknown"),
                    "updated": flow_data.get("updated_at")
                }
                
                print(f"Found flow: {flow_info['title']}")
                
                # Get detailed flow info
                detailed_flow = self.scrape_flow(flow_id, basic_info=flow_info)
                flows.append(detailed_flow)
                
                # Add a small delay to be nice to the server
                time.sleep(0.5)
            
            return {
                "collection": collection_info,
                "flows": flows
            }
            
        except json.JSONDecodeError:
            print("Error: Failed to parse collection data as JSON")
            return self.results
    
    def scrape_flow(self, flow_id: str, basic_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Scrape a Node-RED flow
        
        Args:
            flow_id: ID of the flow
            basic_info: Basic flow info if already available
            
        Returns:
            Dictionary with flow data
        """
        if not basic_info:
            print(f"Scraping flow with ID: {flow_id}")
            basic_info = {
                "id": flow_id,
                "url": f"{self.base_url}/flow/{flow_id}"
            }
        
        # Get the flow page
        flow_url = basic_info["url"]
        response = self.session.get(flow_url)
        if response.status_code != 200:
            print(f"Error: Failed to fetch flow page: {response.status_code}")
            return basic_info
        
        # Extract flow details
        flow_html = response.text
        
        # Only parse these if not already in basic_info
        if "title" not in basic_info:
            title_match = re.search(r'<h1 class="flow-title">(.*?)</h1>', flow_html)
            basic_info["title"] = title_match.group(1).strip() if title_match else "Untitled Flow"
        
        # Extract description and additional info
        description_match = re.search(r'<div class="col-9-12 docs-content">.*?<p>(.*?)</p>', flow_html, re.DOTALL)
        if description_match:
            basic_info["description"] = description_match.group(1).strip()
        
        # Extract tags
        tags = []
        tags_match = re.findall(r'<li><a href="/search\?term=.*?">(.*?)</a></li>', flow_html)
        if tags_match:
            tags = [tag.strip() for tag in tags_match]
        basic_info["tags"] = tags
        
        # Look for flow data (JSON)
        json_match = re.search(r'<pre id="flow".*?>(.*?)</pre>', flow_html, re.DOTALL)
        if json_match:
            try:
                flow_json = json.loads(json_match.group(1).strip())
                # Add some basic stats from the flow JSON
                node_count = len([n for n in flow_json if "type" in n and n["type"] != "tab"])
                tab_count = len([n for n in flow_json if "type" in n and n["type"] == "tab"])
                basic_info["node_count"] = node_count
                basic_info["tab_count"] = tab_count
            except json.JSONDecodeError:
                print(f"Warning: Could not parse flow JSON for {flow_id}")
        
        # Add to results if not already present
        if not any(flow["id"] == flow_id for flow in self.results["flows"]):
            self.results["flows"].append(basic_info)
        
        return basic_info

    def print_json(self):
        """Print results as JSON"""
        print(json.dumps(self.results, indent=2))

def extract_id_from_url(url: str) -> Optional[str]:
    """Extract ID from URL"""
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
    """Main function to scrape Node-RED collections and flows"""
    parser = argparse.ArgumentParser(description="Scrape Node-RED collections and flows")
    parser.add_argument("--url", required=True, help="URL of Node-RED collection or flow")
    
    args = parser.parse_args()
    
    try:
        # Create scraper
        scraper = NodeREDScraper()
        
        # Extract ID from URL
        item_id = extract_id_from_url(args.url)
        if not item_id:
            print(f"Error: Could not extract ID from URL: {args.url}")
            return 1
        
        # Determine type and scrape
        if "/collection/" in args.url:
            scraper.scrape_collection(item_id)
        elif "/flow/" in args.url:
            scraper.scrape_flow(item_id)
        else:
            print("URL type unknown, trying as collection")
            scraper.scrape_collection(item_id)
        
        # Print results
        scraper.print_json()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 