import numpy as np
import matplotlib.pyplot as plt
from sdf_utils import MultiplePolygons, Polygon
import os
import glob
import re
from pathlib import Path

class MultiElementGeometryProcessor:
    """
    Process multi-element airfoil geometries from .hug files and generate SDF database
    """
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.geometry_files = []
        self.database = {}
        
    def find_geometry_files(self):
        """Find all .hug files in the data directory"""
        pattern = os.path.join(self.data_directory, "**/*.hug")
        self.geometry_files = glob.glob(pattern, recursive=True)
        print(f"Found {len(self.geometry_files)} geometry files")
        return self.geometry_files
    
    def extract_y_location(self, filename):
        """Extract y location from filename"""
        # Pattern to match y3.321 or similar
        match = re.search(r'y([\d.]+)', os.path.basename(filename))
        if match:
            return float(match.group(1))
        return None
    
    def extract_sweep_angle(self, filename):
        """Extract sweep angle from filename"""
        # Pattern to match SWEEP=0.00 or similar (avoiding trailing periods)
        match = re.search(r'SWEEP=([-]?[\d.]+)', os.path.basename(filename))
        if match:
            # Remove any trailing periods that might have been captured
            sweep_str = match.group(1).rstrip('.')
            try:
                return float(sweep_str)
            except ValueError:
                return None
        return None
    
    def load_hug_file(self, filepath):
        """
        Load a .hug file and parse multi-element geometry
        Format: 
        - Line 1: number of elements
        - Line 2: number of points in first element
        - Next N lines: coordinates for first element
        - Line N+3: number of points in second element
        - Next M lines: coordinates for second element
        - etc.
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            num_elements = int(lines[0].strip())
            
            elements = []
            line_idx = 1
            
            for elem_idx in range(num_elements):
                # Read number of points for this element
                if line_idx >= len(lines):
                    break
                    
                num_points = int(lines[line_idx].strip())
                line_idx += 1
                
                # Read coordinates for this element
                element_coords = []
                for i in range(num_points):
                    if line_idx >= len(lines):
                        break
                        
                    line = lines[line_idx].strip()
                    if line and not line.startswith('#'):
                        coords = [float(x) for x in line.split()]
                        if len(coords) >= 2:
                            element_coords.append([coords[0], coords[1]])
                    line_idx += 1
                
                if element_coords:
                    element_coords = np.array(element_coords)
                    elements.append(self._reorder_polygon(element_coords))
            
            return elements, num_elements
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return [], 0
    
    def _reorder_polygon(self, pts):
        """
        Reorder a set of 2D points to form a closed polygon.
        Assumes the upper surface is from leading edge to trailing edge.
        """
        if len(pts) < 3:
            return pts
            
        # Find the trailing edge (largest x coordinate)
        trailing_edge_index = np.argmax(pts[:, 0])
        
        # Split into upper and lower surfaces
        upper = pts[:trailing_edge_index+1]
        lower = pts[trailing_edge_index+1:][::-1]
        
        # Combine surfaces
        poly = np.vstack((upper, lower))
        
        # Remove duplicate endpoint if present
        if len(poly) > 1 and np.linalg.norm(poly[0] - poly[-1]) < 1e-6:
            poly = poly[:-1]
            
        return poly
    
    def generate_sdf_for_geometry(self, elements, num_points=10000, bbox_extension=(0.5, 0.5)):
        """
        Generate SDF data for a multi-element geometry
        """
        if not elements:
            return None, None
            
        try:
            # Create MultiplePolygons object
            multi_poly = MultiplePolygons(elements)
            
            # Sample points and compute SDF
            points, sdf = multi_poly.sample_points(num_points=num_points, bbox_extension=bbox_extension)
            
            return points, sdf
            
        except Exception as e:
            print(f"Error generating SDF: {e}")
            return None, None
    
    def process_all_geometries(self, num_points=10000, bbox_extension=(0.5, 0.5)):
        """
        Process all geometry files and generate SDF database
        """
        self.find_geometry_files()
        
        for idx, filepath in enumerate(self.geometry_files):
            print(f"Processing {idx+1}/{len(self.geometry_files)}: {os.path.basename(filepath)}")
            
            # Extract metadata from filename
            y_location = self.extract_y_location(filepath)
            sweep_angle = self.extract_sweep_angle(filepath)
            
            # Load geometry
            elements, num_elements = self.load_hug_file(filepath)
            
            if not elements:
                print(f"  Skipping - no valid elements found")
                continue
            
            # Print element info for debugging
            element_sizes = [len(elem) for elem in elements]
            print(f"  Elements detected: {len(elements)} with sizes {element_sizes}")
            
            # Generate SDF
            points, sdf = self.generate_sdf_for_geometry(elements, num_points, bbox_extension)
            
            if points is not None and sdf is not None:
                # Store in database
                self.database[idx] = {
                    'filename': os.path.basename(filepath),
                    'filepath': filepath,
                    'y_location': y_location,
                    'sweep_angle': sweep_angle,
                    'num_elements': num_elements,
                    'element_sizes': element_sizes,  # Added for debugging
                    'elements': elements,
                    'points': points,
                    'sdf': sdf,
                    'num_points': len(points)
                }
                print(f"  Successfully processed - {num_elements} elements, {len(points)} SDF points")
            else:
                print(f"  Failed to generate SDF")
        
        print(f"\nProcessed {len(self.database)} geometries successfully")
        return self.database
    
    def save_database(self, output_filename="multi_element_sdf_database.npy"):
        """Save the database to a numpy file"""
        np.save(output_filename, self.database)
        print(f"Database saved to {output_filename}")
    
    def visualize_geometry(self, idx, save_plots=True, output_dir="visualizations"):
        """
        Visualize a specific geometry with its SDF field
        """
        if idx not in self.database:
            print(f"Geometry {idx} not found in database")
            return
        
        entry = self.database[idx]
        elements = entry['elements']
        points = entry['points']
        sdf = entry['sdf']
        
        # Create output directory if it doesn't exist
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot of SDF field
        scatter = ax1.scatter(points[:, 0], points[:, 1], c=sdf, 
                             cmap='viridis', s=1, alpha=0.6)
        
        # Plot each element
        colors = ['red', 'blue', 'green']
        element_names = ['Element 1', 'Element 2', 'Element 3']
        
        for i, element in enumerate(elements):
            color = colors[i % len(colors)]
            name = element_names[i] if i < len(element_names) else f'Element {i+1}'
            ax1.plot(element[:, 0], element[:, 1], color=color, 
                    linewidth=2, label=name)
        
        plt.colorbar(scatter, ax=ax1, label='SDF')
        ax1.set_title(f'SDF Field - {entry["filename"]}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.set_aspect('equal')
        
        # Contour plot
        try:
            contour = ax2.tricontour(points[:, 0], points[:, 1], sdf, 
                                   levels=20, cmap='viridis')
            for i, element in enumerate(elements):
                color = colors[i % len(colors)]
                ax2.plot(element[:, 0], element[:, 1], color=color, linewidth=2)
            
            plt.colorbar(contour, ax=ax2, label='SDF')
            ax2.set_title(f'SDF Contours - {entry["filename"]}')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_aspect('equal')
        except Exception as e:
            print(f"Could not create contour plot: {e}")
            ax2.text(0.5, 0.5, 'Contour plot failed', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{output_dir}/geometry_{idx}_{entry['filename'].replace('.hug', '.png')}"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        plt.show()
    
    def get_database_summary(self):
        """Print a summary of the processed database"""
        if not self.database:
            print("Database is empty")
            return
        
        print(f"\n=== Database Summary ===")
        print(f"Total geometries: {len(self.database)}")
        
        # Count by number of elements
        element_counts = {}
        y_locations = []
        sweep_angles = []
        
        for entry in self.database.values():
            num_elem = entry['num_elements']
            element_counts[num_elem] = element_counts.get(num_elem, 0) + 1
            
            if entry['y_location'] is not None:
                y_locations.append(entry['y_location'])
            if entry['sweep_angle'] is not None:
                sweep_angles.append(entry['sweep_angle'])
        
        print("\nElement distribution:")
        for num_elem, count in sorted(element_counts.items()):
            print(f"  {num_elem} elements: {count} geometries")
        
        if y_locations:
            print(f"\nY locations: {min(y_locations):.3f} to {max(y_locations):.3f}")
            print(f"  Unique y locations: {len(set(y_locations))}")
        
        if sweep_angles:
            print(f"\nSweep angles: {min(sweep_angles):.2f}° to {max(sweep_angles):.2f}°")
            print(f"  Unique sweep angles: {len(set(sweep_angles))}")


def main():
    """Main function to process geometries and generate database"""
    
    # Configuration
    data_directories = {
        'LDG': "/home/dmsm/gi.catalani/Projects/enf-torch/enf-torch-main/multi_airfoil/data/Geom2.5D_HUGS_LDG",
        'CLEAN': "/home/dmsm/gi.catalani/Projects/enf-torch/enf-torch-main/multi_airfoil/data/Geom2.5D_HUGS_CLEAN", 
        'TO': "/home/dmsm/gi.catalani/Projects/enf-torch/enf-torch-main/multi_airfoil/data/Geom2.5D_HUGS_TO"
    }
    output_filename = "/home/dmsm/gi.catalani/Projects/enf-torch/enf-torch-main/multi_airfoil/data/multi_element_sdf_database.npy"
    num_points = 5000
    bbox_extension = (0.5, 0.5)
    
    # Combined database
    combined_database = {}
    global_idx = 0
    
    # Process each directory
    for config_type, data_directory in data_directories.items():
        print(f"\n{'='*60}")
        print(f"Processing {config_type} geometries from {data_directory}")
        print(f"{'='*60}")
        
        # Initialize processor for this directory
        processor = MultiElementGeometryProcessor(data_directory)
        
        # Process all geometries in this directory
        database = processor.process_all_geometries(
            num_points=num_points, 
            bbox_extension=bbox_extension
        )
        
        # Add to combined database with config_type key
        for local_idx, entry in database.items():
            entry['config_type'] = config_type  # Add the configuration type
            combined_database[global_idx] = entry
            global_idx += 1
        
        print(f"Added {len(database)} {config_type} geometries to combined database")
    
    # Save combined database
    print(f"\n{'='*60}")
    print(f"Saving combined database with {len(combined_database)} total geometries...")
    np.save(output_filename, combined_database)
    print(f"Combined database saved to {output_filename}")
    
    # Print combined summary
    print(f"\n=== Combined Database Summary ===")
    print(f"Total geometries: {len(combined_database)}")
    
    # Count by configuration type and elements
    config_counts = {}
    element_counts = {}
    
    for entry in combined_database.values():
        config_type = entry['config_type']
        num_elem = entry['num_elements']
        
        config_counts[config_type] = config_counts.get(config_type, 0) + 1
        
        key = f"{config_type}_{num_elem}elem"
        element_counts[key] = element_counts.get(key, 0) + 1
    
    print("\nConfiguration distribution:")
    for config_type, count in sorted(config_counts.items()):
        print(f"  {config_type}: {count} geometries")
    
    print("\nDetailed element distribution:")
    for key, count in sorted(element_counts.items()):
        print(f"  {key}: {count} geometries")
    
    # Visualize samples from each configuration
    if len(combined_database) > 0:
        print("\nGenerating sample visualizations...")
        
        # Get one sample from each configuration type
        samples_by_config = {}
        for idx, entry in combined_database.items():
            config_type = entry['config_type']
            if config_type not in samples_by_config:
                samples_by_config[config_type] = idx
        
        # Visualize one from each configuration
        for config_type, idx in samples_by_config.items():
            print(f"Visualizing sample from {config_type}...")
            # Create a temporary processor just for visualization
            temp_processor = MultiElementGeometryProcessor("")
            temp_processor.database = {idx: combined_database[idx]}
            temp_processor.visualize_geometry(idx, save_plots=True, 
                                            output_dir='/home/dmsm/gi.catalani/Projects/enf-torch/enf-torch-main/multi_airfoil/data/visualizations')


if __name__ == "__main__":
    main()