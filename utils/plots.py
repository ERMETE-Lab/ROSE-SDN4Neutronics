# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 09 July 2025
# Latest Doc  Update: 09 July 2025

from matplotlib.patches import Rectangle
from matplotlib import cm
import pyvista as pv

class PlotTriga:
    """
    A utility class for slicing and visualizing scalar fields on a VTK mesh, 
    specifically tailored for TRIGA reactor geometries.

    Parameters
    ----------
    vtk_mesh : pyvista.UnstructuredGrid
        A VTK mesh representing the TRIGA reactor geometry.
    """
    
    def __init__(self, vtk_mesh):
        self.grid = vtk_mesh

    def get_slice(self, snap, name, plane_normal='z', plane_origin=(0, 0, 47)):
        """
        Extracts a 2D slice from the 3D VTK mesh based on a scalar field.

        Parameters
        ----------
        snap : np.ndarray
            Array of scalar values to assign to the mesh (e.g., flux or power distribution).
        name : str
            The name of the scalar field to associate with the mesh data.
        plane_normal : {'x', 'y', 'z'}, optional
            The normal direction of the slicing plane. Default is 'z'.
        plane_origin : tuple of float, optional
            The origin of the slicing plane. Default is (0, 0, 47).

        Returns
        -------
        slice_mesh : list of np.ndarray
            A list of two arrays representing the 2D coordinates of the slice in the plane.
        slice_field : np.ndarray
            Scalar values associated with the slice.

        Raises
        ------
        ValueError
            If `plane_normal` is not one of {'x', 'y', 'z'}.
        """
        self.grid[name] = snap
        _slice = self.grid.slice(normal=plane_normal, origin=plane_origin)
        points = _slice.cell_centers().points

        axis_map = {
            'x': (1, 2),
            'y': (0, 2),
            'z': (0, 1)
        }

        if plane_normal not in axis_map:
            raise ValueError("Invalid plane normal. Use 'x', 'y', or 'z'.")

        idx1, idx2 = axis_map[plane_normal]
        _slice_mesh = [points[:, idx1], points[:, idx2]]
        _slice_field = _slice[name]
        
        # Clear the data from the grid to avoid memory issues
        self.grid.clear_data()

        return _slice_mesh, _slice_field
        
    def plot_slice(self, ax, snap, 
                   normal='z', origin=(0, 0, 47),
                   levels=20, cmap=cm.jet, name='u'):
        """
        Plots a filled contour of a 2D slice from the VTK mesh.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis on which to draw the plot.
        snap : np.ndarray
            Array of scalar values to assign to the mesh.
        normal : {'x', 'y', 'z'}, optional
            The normal direction of the slicing plane. Default is 'z'.
        origin : tuple of float, optional
            The origin of the slicing plane. Default is (0, 0, 47).
        levels : int, optional
            Number of contour levels to draw. Default is 20.
        cmap : matplotlib.colors.Colormap, optional
            Colormap used for the filled contour plot. Default is `cm.jet`.
        name : str, optional
            Name of the scalar field to plot. Default is 'u'.

        Returns
        -------
        matplotlib.contour.QuadContourSet
            The contour set object returned by `ax.tricontourf`.
        """
        _slice_mesh, _slice_field = self.get_slice(snap=snap, name=name, plane_normal=normal, plane_origin=origin)
        cont = ax.tricontourf(*_slice_mesh, _slice_field, levels=levels, cmap=cmap)

        return cont