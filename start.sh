#!/bin/bash
# Startup script for HumAIne Swarm Assistant with OAuth

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting HumAIne Swarm Assistant...${NC}"

# Check if env.sh exists
if [ ! -f "env.sh" ]; then
    echo -e "${RED}Error: env.sh not found!${NC}"
    echo -e "${YELLOW}Please copy env.sh.example to env.sh and configure your environment variables.${NC}"
    exit 1
fi

# Source environment variables
echo -e "${GREEN}Loading environment variables from env.sh...${NC}"
source env.sh

# Verify OAuth variables are set
if [ -z "$OAUTH_KEYCLOAK_CLIENT_ID" ]; then
    echo -e "${RED}Error: OAUTH_KEYCLOAK_CLIENT_ID is not set!${NC}"
    echo -e "${YELLOW}Please check your env.sh configuration.${NC}"
    exit 1
fi

if [ -z "$OAUTH_KEYCLOAK_CLIENT_SECRET" ]; then
    echo -e "${RED}Error: OAUTH_KEYCLOAK_CLIENT_SECRET is not set!${NC}"
    echo -e "${YELLOW}Please check your env.sh configuration.${NC}"
    exit 1
fi

if [ -z "$OAUTH_KEYCLOAK_BASE_URL" ]; then
    echo -e "${RED}Error: OAUTH_KEYCLOAK_BASE_URL is not set!${NC}"
    echo -e "${YELLOW}Please check your env.sh configuration.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ OAuth configuration loaded${NC}"
echo -e "${GREEN}✓ Client ID: $OAUTH_KEYCLOAK_CLIENT_ID${NC}"
echo -e "${GREEN}✓ Realm: $OAUTH_KEYCLOAK_REALM${NC}"
echo -e "${GREEN}✓ Base URL: $OAUTH_KEYCLOAK_BASE_URL${NC}"
echo ""

# Start Chainlit
echo -e "${GREEN}Starting Chainlit server...${NC}"
chainlit run app.py -w

