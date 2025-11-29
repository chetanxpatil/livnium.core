#!/bin/bash
# Launch Livnium Geometry Explorer - The REAL 3D Structure

cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════════════════"
echo "  LIVNIUM GEOMETRY EXPLORER"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  The REAL 3D Force Field Inside the Cube"
echo "  Not a projection. Not a sphere. The actual geometry."
echo ""
echo "  Three views:"
echo "    • Vector Field - Force directions as arrows"
echo "    • Volume MRI - Density slices like a CT scan"
echo "    • Isosurface - The actual topological shape"
echo ""

# Check if force field data exists
FORCE_FIELD_DATA="planet_output/livnium_3d_force_field.json"
if [ ! -f "$FORCE_FIELD_DATA" ]; then
    echo "Generating 3D force field data..."
    python3 planet/compute_3d_force_field.py --resolution 20
    echo ""
fi

# Find available port
PORT=8000
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; do
    echo "Port $PORT is in use, trying next port..."
    PORT=$((PORT + 1))
done

# Start server from nli_v5 directory
echo "Starting web server on port $PORT..."
echo "Open your browser to: http://localhost:$PORT/viewer/livnium_geometry_explorer.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m http.server $PORT

