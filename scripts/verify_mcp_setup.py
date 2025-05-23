#!/usr/bin/env python
"""
Verify MCP Server Configuration for GitHub Copilot
This script checks your VS Code settings and verifies that the MCP server is properly configured.
"""

import os
import sys
import json
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger()

def check_mcp_json():
    """Check for .mcp.json file in the project directory"""
    mcp_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mcp.json")
    
    if not os.path.exists(mcp_file):
        logger.error("Missing .mcp.json file in project directory")
        return False
        
    try:
        with open(mcp_file, 'r') as f:
            mcp_config = json.load(f)
            
        # Validate required fields
        if "servers" not in mcp_config:
            logger.error(".mcp.json is missing 'servers' configuration")
            return False
            
        if not mcp_config["servers"] or len(mcp_config["servers"]) == 0:
            logger.error(".mcp.json has empty 'servers' configuration")
            return False
            
        # Check first server config
        server = mcp_config["servers"][0]
        if "id" not in server or "url" not in server:
            logger.error(".mcp.json server configuration is missing required fields")
            return False
            
        logger.info(f"MCP configuration found: Server ID: {server['id']}, URL: {server['url']}")
        return True
    except json.JSONDecodeError:
        logger.error(".mcp.json is not valid JSON")
        return False
    except Exception as e:
        logger.error(f"Error reading .mcp.json: {e}")
        return False

def check_vscode_settings():
    """Check VS Code settings for MCP configuration"""
    home_dir = os.path.expanduser("~")
    settings_path = os.path.join(home_dir, "Library", "Application Support", "Code", "User", "settings.json")
    
    if not os.path.exists(settings_path):
        logger.warning(f"VS Code settings not found at {settings_path}")
        return False
        
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            
        # Check for GitHub Copilot MCP configuration
        if "github.copilot.advanced" not in settings:
            logger.error("VS Code settings missing GitHub Copilot advanced configuration")
            return False
            
        copilot_settings = settings["github.copilot.advanced"]
        if "mcp" not in copilot_settings or "servers" not in copilot_settings["mcp"]:
            logger.error("VS Code settings missing MCP servers configuration")
            return False
            
        servers = copilot_settings["mcp"]["servers"]
        if not servers or len(servers) == 0:
            logger.error("No MCP servers configured in VS Code settings")
            return False
            
        # Find our server
        found = False
        for server in servers:
            if "name" in server and server["name"] == "cursor-templates-vss":
                logger.info(f"Found MCP server configuration in VS Code settings: {server}")
                found = True
                break
                
        if not found:
            logger.error("MCP server 'cursor-templates-vss' not found in VS Code settings")
            return False
            
        return True
    except json.JSONDecodeError:
        logger.error("VS Code settings.json is not valid JSON")
        return False
    except Exception as e:
        logger.error(f"Error reading VS Code settings: {e}")
        return False

def check_mcp_server():
    """Check if MCP server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("MCP server is running and healthy")
            return True
        else:
            logger.error(f"MCP server health check failed with status code: {response.status_code}")
            return False
    except requests.ConnectionError:
        logger.error("Failed to connect to MCP server - is it running?")
        return False
    except Exception as e:
        logger.error(f"Error checking MCP server: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print(" MCP SERVER CONFIGURATION CHECK ".center(60, "="))
    print("=" * 60)
    
    mcp_json_ok = check_mcp_json()
    vscode_settings_ok = check_vscode_settings()
    server_running = check_mcp_server()
    
    print("\n" + "-" * 60)
    print(" SUMMARY ".center(60, "-"))
    print("-" * 60)
    print(f"✓ .mcp.json configuration:         {'OK' if mcp_json_ok else 'MISSING/INVALID'}")
    print(f"✓ VS Code settings:               {'OK' if vscode_settings_ok else 'MISSING/INVALID'}")
    print(f"✓ MCP server running:             {'OK' if server_running else 'NOT RUNNING'}")
    
    if mcp_json_ok and vscode_settings_ok and server_running:
        print("\n✅ MCP server is properly configured and running!")
        print("GitHub Copilot should be able to access your vector search data.")
    else:
        print("\n❌ MCP server configuration is incomplete.")
        print("Please check the errors above and fix the issues.")
    
    print("-" * 60)

if __name__ == "__main__":
    main()
