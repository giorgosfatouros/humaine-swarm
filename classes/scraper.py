import os
import sys
import requests
import time
import re
from bs4 import BeautifulSoup
import json
from datetime import datetime
import argparse
from typing import Dict, List, Any, Optional, Union
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class NodeREDParser:
    """
    Parser for Node-RED collections and flows
    """
    
    def __init__(self):
        self.results = {
            "parsed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "collections": [],
            "flows": []
        }
        self.driver = None
    
    def _setup_selenium(self):
        """Set up Selenium WebDriver for JavaScript rendering"""
        if self.driver is not None:
            return
            
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"Error setting up Chrome: {e}")
            print("Trying Firefox...")
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            from selenium.webdriver.firefox.service import Service as FirefoxService
            
            firefox_options = FirefoxOptions()
            firefox_options.add_argument("--headless")
            self.driver = webdriver.Firefox(options=firefox_options)
    
    def _close_selenium(self):
        """Close the Selenium WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def parse_collection(self, url: str) -> Dict[str, Any]:
        """
        Parse a Node-RED collection page using Selenium for JavaScript rendering
        
        Args:
            url: URL of the collection page
            
        Returns:
            Dictionary with collection information
        """
        # Set up Selenium
        self._setup_selenium()
        
        print(f"Loading page with Selenium: {url}")
        self.driver.get(url)
        
        # Wait for the page to load
        try:
            # Wait for either real content or "empty" message
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".gistlist li:not(.gistbox-placeholder),.thing-list-nav"))
            )
            
            # Additional wait to ensure JavaScript has finished
            time.sleep(2)
        except Exception as e:
            print(f"Warning: Timed out waiting for page to fully load: {e}")
        
        # Get the page source after JavaScript has executed
        html_content = self.driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. Extract collection metadata
        collection_info = {
            "title": self._safe_extract(soup, ".flow-title"),
            "description": self._safe_extract(soup, ".col-9-12.docs-content p"),
            "owners": [owner.text.strip() for owner in soup.select(".flowmeta:nth-of-type(2) .flowinfo a")],
            "id": self._extract_id_from_url(url),
            "url": url
        }
        
        print(f"Parsed collection: {collection_info['title']}")
        
        # 2. Find the gistlist class
        flow_urls = []
        
        # Look for actual flow items (not placeholders)
        flow_items = soup.select(".gistlist li:not(.gistbox-placeholder) a")
        
        if not flow_items:
            # Check if collection is empty
            empty_warning = soup.select_one("#collection-empty-warning")
            empty_count = soup.select_one("#collection-item-count-empty")
            
            if empty_warning and empty_warning.get_attribute_list("style") and "display: none" not in empty_warning.get_attribute_list("style"):
                print("Collection is explicitly marked as empty")
            elif empty_count and empty_count.get_attribute_list("style") and "display: none" not in empty_count.get_attribute_list("style"):
                print("Collection item count shows empty")
            else:
                print("No flow items found but collection may not be empty - JavaScript loading issue")
        
        print(f"Found {len(flow_items)} flow items in gistlist")
        
        # 3. Find href of each li in the gistlist
        for item in flow_items:
            href = item.get("href")
            if href and "/flow/" in href:
                if not href.startswith("http"):
                    full_url = "https://flows.nodered.org" + href
                else:
                    full_url = href
                flow_urls.append(full_url)
                print(f"Found flow URL: {full_url}")
        
        # Store the collection
        self.results["collections"].append(collection_info)
        
        # 4. Scrape each flow page
        flows = []
        for url in flow_urls:
            try:
                print(f"Fetching flow from: {url}")
                flow_data = self.parse_flow(url)
                flow_data["collection_id"] = collection_info["id"]
                flows.append(flow_data)
                print(f"Parsed flow: {flow_data.get('title', 'Unknown')}")
                time.sleep(0.5)  # Small delay
            except Exception as e:
                print(f"Error parsing flow from {url}: {str(e)}")
        
        print(f"Processed {len(flows)} flows in collection")
        return {
            "collection": collection_info,
            "flows": flows
        }
    
    def parse_flow(self, url: str) -> Dict[str, Any]:
        """
        Parse a Node-RED flow page using Selenium
        
        Args:
            url: URL of the flow page
            
        Returns:
            Dictionary with flow information
        """
        # Set up Selenium if needed
        self._setup_selenium()
        
        print(f"Loading flow page with Selenium: {url}")
        self.driver.get(url)
        
        # Wait for the page to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".flow-title"))
            )
            # Additional wait
            time.sleep(1)
        except Exception as e:
            print(f"Warning: Timed out waiting for flow page to load: {e}")
        
        # Get the page source after JavaScript has executed
        html_content = self.driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract flow metadata
        flow_info = {
            "id": self._extract_id_from_url(url),
            "title": self._safe_extract(soup, ".flow-title"),
            "description": self._extract_flow_description(soup),
            "owner": self._safe_extract(soup, ".flowmeta:contains('Owner') .flowinfo a"),
            "tags": [tag.text.strip() for tag in soup.select(".flow-tags li")],
            "url": url
        }
        
        # Add flow to results
        self.results["flows"].append(flow_info)
        
        return flow_info
    
    def _safe_extract(self, soup: BeautifulSoup, selector: str, index: int = 0) -> str:
        """Safely extract text from a BS4 selector"""
        elements = soup.select(selector)
        if elements and len(elements) > index:
            return elements[index].text.strip()
        return ""
    
    def _extract_id_from_url(self, url: str) -> Optional[str]:
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
    
    def _extract_flow_description(self, soup: BeautifulSoup) -> str:
        """Extract flow description from the page"""
        paragraphs = soup.select(".docs-content > p")
        if paragraphs:
            return "\n".join([p.text.strip() for p in paragraphs])
        return ""
    
    def get_results_as_dict(self) -> Dict[str, Any]:
        """Return the results as a dictionary"""
        return self.results
    
    def print_json(self):
        """Print the results as JSON to stdout"""
        print(json.dumps(self.results, indent=2))
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self._close_selenium()

def main():
    """Main function to parse Node-RED collection and flow pages"""
    parser = argparse.ArgumentParser(description="Parse Node-RED collection and flow pages")
    parser.add_argument("--url", required=True, help="URL to scrape (Node-RED collection or flow)")
    
    args = parser.parse_args()
    
    try:
        # Create parser instance
        node_red_parser = NodeREDParser()
        
        # Determine if it's a collection or flow page
        if "/collection/" in args.url:
            print("Parsing collection page...")
            node_red_parser.parse_collection(args.url)
        elif "/flow/" in args.url:
            print("Parsing flow page...")
            node_red_parser.parse_flow(args.url)
        else:
            print("URL type unknown, trying as collection...")
            node_red_parser.parse_collection(args.url)
        
        # Print results as JSON
        node_red_parser.print_json()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        # Clean up Selenium
        if hasattr(node_red_parser, '_close_selenium'):
            node_red_parser._close_selenium()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

