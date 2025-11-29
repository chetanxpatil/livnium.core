# Livnium Planet v2 - Web Viewer

Interactive 3D visualization of Livnium Planet in your browser.

## Quick Start

1. **Export planet data:**
   ```bash
   python3 experiments/nli_v5/export_planet_data.py
   ```
   This creates `planet_data.json` with the planet geometry.

2. **Open in browser:**
   ```bash
   # Option 1: Simple HTTP server
   cd experiments/nli_v5
   python3 -m http.server 8000
   # Then open http://localhost:8000/livnium_planet_web.html
   
   # Option 2: Just open the HTML file directly
   open livnium_planet_web.html
   ```

## Features

- **Interactive 3D View**: Rotate, zoom, pan with mouse
- **Auto-rotation**: Toggle automatic planet rotation
- **Wireframe Mode**: Toggle wireframe view
- **Real-time Stats**: FPS and vertex count
- **Multi-layered Visualization**: Shows all Livnium fields

## Controls

- **Mouse Drag**: Rotate camera
- **Scroll**: Zoom in/out
- **Right-click Drag**: Pan camera
- **Toggle Rotation**: Button to start/stop auto-rotation
- **Reset View**: Return to default camera position
- **Toggle Wireframe**: Switch between solid and wireframe view

## Files

- `livnium_planet_web.html` - Main web viewer
- `planet_data.json` - Planet geometry data (generated)
- `export_planet_data.py` - Script to generate planet data

## Browser Compatibility

Works best in:
- Chrome/Edge (recommended)
- Firefox
- Safari

Requires WebGL support.

