# DBSCAN vs HDBSCAN Clustering Comparison

A comprehensive comparison of density-based clustering algorithms applied to Canadian museum location data.

## Overview

This project demonstrates the differences between two popular density-based clustering algorithms:
- **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)
- **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

## Dataset

**Source**: The Open Database of Cultural and Art Facilities (ODCAF) from Statistics Canada
- Contains names, types, and locations of cultural facilities across Canada
- Focus: Museum locations for geographic clustering analysis
- License: Open Government License - Canada

## Key Learning Objectives

✅ Implement DBSCAN and HDBSCAN clustering using scikit-learn  
✅ Compare clustering performance on real geographic data  
✅ Understand scaling requirements for coordinate data  
✅ Visualize clustering results on geographic maps  
✅ Analyze density variations in spatial data  

## Algorithm Comparison

| Feature | DBSCAN | HDBSCAN |
|---------|---------|----------|
| **Approach** | Fixed density threshold | Hierarchical density analysis |
| **Parameters** | eps, min_samples | min_cluster_size, min_samples |
| **Cluster Shape** | Uniform density | Variable density |
| **Noise Handling** | Fixed threshold | Adaptive threshold |
| **Best For** | Known density patterns | Varying density patterns |

## Project Structure

```
DBSCAN_HDBSCAN_Clustering/
├── README.md                           # This file
├── dbscan_hdbscan_clustering.ipynb     # Main analysis notebook
└── requirements.txt                    # Required packages
```

## Key Findings

### DBSCAN Results
- **Clusters Found**: 2 large clusters + 79 noise points
- **Characteristics**: 
  - Forms clusters based on fixed neighborhood radius (eps=1.0)
  - Aggregates neighboring regions within distance threshold
  - Less granular clustering approach

### HDBSCAN Results  
- **Clusters Found**: Multiple uniformly-sized clusters + more noise points
- **Characteristics**:
  - Adapts to local density variations
  - More conservative cluster assignment
  - Better captures geographic density patterns

## Technical Implementation

### Data Preprocessing
1. **Data Filtering**: Extract museum-only records
2. **Missing Data**: Remove records with missing coordinates (..)
3. **Type Conversion**: Convert coordinates to float values
4. **Geographic Scaling**: Account for lat/lng coordinate differences

### Coordinate Scaling Strategy
```python
# Why multiply latitude by 2?
# - Latitude range: -90° to +90° (180° total)
# - Longitude range: 0° to 360° (360° total)
# - Scaling makes distance calculations geographically accurate
coords_scaled["Latitude"] = 2 * coords_scaled["Latitude"]
```

### Visualization
- Uses **geopandas** for geographic data handling
- **contextily** for Canadian basemap integration
- Color-coded clusters with noise points highlighted

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
hdbscan
geopandas
contextily
shapely
requests
```

## Usage

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   - Open `dbscan_hdbscan_clustering.ipynb` in Jupyter
   - Execute cells sequentially
   - View clustering results on Canadian map

3. **Experiment**:
   - Adjust DBSCAN parameters (eps, min_samples)
   - Modify HDBSCAN parameters (min_cluster_size)
   - Compare results across different parameter sets

## Key Insights

### Geographic Clustering Considerations
- **Variable Density**: Canadian museums show varying density patterns
- **Urban vs Rural**: Major cities have dense clusters, rural areas sparse
- **Algorithm Choice**: HDBSCAN better suited for geographic data

### Practical Applications
- **Market Analysis**: Identify business location clusters
- **Urban Planning**: Analyze facility distribution patterns
- **Resource Allocation**: Optimize service coverage areas

## Future Enhancements

- [ ] Parameter optimization using silhouette analysis
- [ ] Integration with population density data
- [ ] Comparison with other clustering algorithms (K-means, GMM)
- [ ] Interactive visualization with folium maps
- [ ] Cluster quality metrics evaluation

## References

- [Statistics Canada ODCAF Database](https://www.statcan.gc.ca/en/lode/databases/odcaf)
- [DBSCAN Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- [GeoPandas Documentation](https://geopandas.org/)


