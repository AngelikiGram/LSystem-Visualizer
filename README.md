# L-System Tree Generation and Visualization

This project provides tools for generating, processing, and visualizing tree structures using L-systems, with support for realistic bark and leaf rendering in Blender.

## Project Overview
- **L-system Parsing & Growth Simulation**: Scripts to parse symbolic L-system strings and simulate tree growth.
- **Branch & Leaf Visualization**: Blender scripts to render trees with textured bark and leaves, including advanced placement and scaling logic.
- **Data Processing**: Utilities for converting, simplifying, and analyzing tree structure data.

## Setup
1. **Sync Data from Remote (if needed):**
   ```sh
   rsync -avz -e "ssh -p 31415" --exclude='' grammatikakis1@dgxa100.icsd.hmu.gr:~/P3/inference_data/exports/ /mnt/c/Users/angie/Documents/P3/inference_data/exports
   ```
2. **Python Requirements:**
   - Python 3.x
   - numpy
   - (Optional) bmesh, if running Blender scripts outside Blender

3. **Blender:**
   - Recommended: Blender 3.x or newer
   - Place bark and leaf textures in the directories specified in `viz_polished.py`

## Usage

### Growth Simulation
Run the growth simulation and adjust parameters as needed:
```sh
python growth.py
```
- Edit `growth.py` to set:
  - `TARGET_LEVEL = 1`  # Depth: 0=Trunk, 1=L1, 2=L2, 3=L3
  - `DENSIFY_SCALE = 5`  # Number of additional L2 branches to spawn
  - `HEAL_ENABLED = False`  # If False, Densify Scale is ignored for L2

### Tree Visualization in Blender
Render the tree with branches, bark textures, and leaves:
```sh
blender --python viz_polished.py
```

### Blender Add-on: L-System Visualizer & Seasons
**The L-System Visualizer & Seasons Blender Add-on procedurally generates, animates, and textures customizable 3D trees from L-System text files with real-time feedback for structural growth, geometry tuning, and dynamic seasonal foliage.**

To install and use:
1. Zip the `lsystem_viz_addon` folder or install the `__init__.py` script directly from Blender (`Edit > Preferences > Add-ons > Install`).
2. Once enabled, open the **3D Viewport > Sidebar (N) > L-System** tab.
3. Configure your text file path, material directories, and tweak the real-time sliders for precise control over tree thickness, seasonal leaf properties, and procedural growth pruning/enrichment.

### Grease Pencil Visualization
- Smooth render:
  ```sh
  blender --python growth_viz_smooth2.py
  ```
- Standard render:
  ```sh
  blender --python growth_viz.py
  ```

## File Descriptions
- `growth.py`: L-system growth simulation and symbolic string generation.
- `tokens_counter.py`: Analyze and count L-system tokens.
- `viz_symbolic.py`, `viz_symbolic_smooth.py.py`: Grease pencil visualizations for tree growth.
- `viz_rendered.py`: Main Blender script for realistic tree and leaf rendering.
- `visualize.py`: Additional visualization tools for A->RF type of representations (non-symbolic, but simplified).

## Notes
- Ensure texture paths in `viz_polished.py` are correct for your system.
- Leaves are placed at the deepest (terminal) branches, and each is a separate object for easy inspection.
- For large trees, Blender performance may be affected by the number of leaf objects.

---