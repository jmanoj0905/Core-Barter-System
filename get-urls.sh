#!/bin/bash
# Get local IP and print access URLs for Alice and Bob

echo "==============================================="
echo "  Core Barter System - Server URLs"
echo "==============================================="
echo ""

# Get local IP (prefer en0, fallback to en6)
IP=$(ifconfig en0 2>/dev/null | grep "inet " | awk '{print $2}')
if [ -z "$IP" ]; then
    IP=$(ifconfig en6 2>/dev/null | grep "inet " | awk '{print $2}')
fi

if [ -z "$IP" ]; then
    echo "Error: Could not find local IP address"
    echo "Make sure you're connected to WiFi/Ethernet"
    exit 1
fi

echo "Your server IP: $IP"
echo ""
echo "==============================================="
echo "  Open on device 1 (Alice):"
echo "==============================================="
echo "  https://$IP"
echo ""
echo "==============================================="
echo "  Open on device 2 (Bob):"
echo "==============================================="
echo "  https://$IP"
echo ""
echo "==============================================="
echo ""
echo "NOTE: Accept the self-signed cert warning"
echo "      when opening in browser."
echo ""
echo "To check IP anytime: ./get-urls.sh"
echo "==============================================="