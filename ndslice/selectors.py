"""
ndslice GUI for choosing datasets from multi-dataset files.
Handles mat, h5, and npz files.
"""
import numpy as np


def _view_dataset_in_subprocess(filepath, dataset_path, interpret_as_complex):
    from pyqtgraph.Qt import QtWidgets
    from .ndslice import ndslice
    import sys
    
    # Import appropriate selector based on file extension
    if filepath.suffix.lower() == '.h5':
        from .selectors import H5DatasetSelector
        selector_class = H5DatasetSelector
    elif filepath.suffix.lower() == '.npz':
        from .selectors import NpzDatasetSelector
        selector_class = NpzDatasetSelector
    elif filepath.suffix.lower() == '.mat':
        from .selectors import MatDatasetSelector
        selector_class = MatDatasetSelector
    else:
        return

    # Create selector and load data locally
    selector = selector_class(filepath)
    data = selector.load_data(dataset_path)
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    ndslice(data=data,
            title=f"{filepath.name} - {dataset_path}",
            block=True,
            complex_dim=interpret_as_complex)


class DatasetSelector:
    """Base class for GUI dialog to select datasets from multi-dataset files."""
    
    COLOR_INCOMPATIBLE = (180, 180, 180) # Light grey for incompatible items
    
    def __init__(self, filepath, compatible_datasets=None):
        from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
        
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.QtGui = QtGui
        
        self.filepath = filepath
        self.compatible_datasets = compatible_datasets or []
        self.selected_dataset = None
        self.interpret_as_complex = None  # Dimension to interpret as complex (real/imag split)
    
    def requires_gui(self):
        """Check if GUI selector is needed (more than one compatible dataset)."""
        return len(self.compatible_datasets) > 1
    
    def get_single_data(self):
        """Get data if there's only one compatible dataset. Returns (name, data) or None."""
        raise NotImplementedError("Subclasses must implement get_single_data()")
    
    def load_data(self, path):
        """Load data for a given dataset path/name. Returns numpy array."""
        raise NotImplementedError("Subclasses must implement load_data()")
    
    def _build_tree(self):
        """Build the tree widget. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_tree()")
    
    def show(self):
        """Show the selector dialog and return the selected dataset name."""
        dialog = self._create_dialog()
        result = dialog.exec()
        
        if result == self.QtWidgets.QDialog.Accepted and self.selected_dataset:
            return self.selected_dataset
        return None
    
    def view(self, block=False):
        """
        Select dataset (with GUI if needed) and open in ndslice.
        Returns True if successful, False otherwise.
        """
        from .ndslice import ndslice
        
        # Auto-load if single compatible dataset
        if not self.requires_gui():
            result = self.get_single_data()
            if result:
                name, data = result
                ndslice(data=data,
                        title=f"{self.filepath.name} - {name}",
                        block=block,
                        complex_dim=self.interpret_as_complex)
                return True
            else:
                return False
        
        # Multiple datasets
        # Ensure QApplication exists
        import sys
        app = self.QtWidgets.QApplication.instance()
        if app is None:
            app = self.QtWidgets.QApplication(sys.argv)
        
        if selected_path := self.show():
            if block:
                data = self.load_data(selected_path)
                ndslice(data=data,
                        title=f"{self.filepath.name} - {selected_path}",
                        block=True,
                        complex_dim=self.interpret_as_complex)
            else:
                import multiprocessing as mp
                mp.Process(target=_view_dataset_in_subprocess,
                          args=(self.filepath, selected_path, self.interpret_as_complex)).start()
            return True
        
        return False
    
    def _create_dialog(self):
        """Create and configure the dialog widget."""
        dialog = self.QtWidgets.QDialog()
        dialog.setWindowTitle(f"Select Dataset - {self.filepath.name}")
        dialog.resize(600, 400)
        
        layout = self.QtWidgets.QVBoxLayout()
        
        # Tree widget
        self.tree = self.QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Shape", "Type"])
        self.tree.setColumnWidth(0, 225)
        self.tree.setColumnWidth(1, 175)
        self.tree.itemDoubleClicked.connect(lambda item, col: self._on_item_double_clicked(item, col, dialog))
        
        # Build tree (implemented by subclasses)
        self._build_tree()
        
        layout.addWidget(self.tree)
                
        dialog.setLayout(layout)
        return dialog
    
    
    def _on_item_double_clicked(self, item, column, dialog):
        """Handle double-click on tree item."""
        compatible = item.data(0, self.QtCore.Qt.UserRole + 1)
        if compatible:
            dataset_name = item.data(0, self.QtCore.Qt.UserRole)
            self.selected_dataset = dataset_name
            dialog.accept()
    
    def _add_group(self, parent, name, path=None, item_type="Group"):
        """Add a group/struct item to the tree (non-selectable, greyed out, expanded)."""
        item = self.QtWidgets.QTreeWidgetItem([name, "", item_type])
        item.setData(0, self.QtCore.Qt.UserRole, path)
        item.setData(0, self.QtCore.Qt.UserRole + 1, False)  # Not compatible
        
        # Customize group appearance
        item.setFlags(item.flags() & ~self.QtCore.Qt.ItemIsSelectable)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setFont(1, font)
        item.setFont(2, font)
        
        if isinstance(parent, self.QtWidgets.QTreeWidget):
            parent.addTopLevelItem(item)
        else:
            parent.addChild(item)
        
        item.setExpanded(True)
        return item
    
    def _add_item(self, parent, name, shape, dtype, path, compatible=True):
        # Format shape
        if isinstance(shape, tuple):
            shape_str = ' × '.join(map(str, shape))
        else:
            shape_str = str(shape)
        
        # Create item
        prefix = ""# "✓ " if compatible else "  "
        item = self.QtWidgets.QTreeWidgetItem([f"{prefix}{name}"])
        item.setText(1, f"[{shape_str}]" if shape_str else "")
        item.setText(2, str(dtype))
        item.setData(0, self.QtCore.Qt.UserRole, path)
        item.setData(0, self.QtCore.Qt.UserRole + 1, compatible)

        if not compatible: # Mark incompatible items
            item.setFlags(item.flags() & ~self.QtCore.Qt.ItemIsEnabled)
            item.setForeground(0, self.QtGui.QBrush(self.QtGui.QColor(*self.COLOR_INCOMPATIBLE)))
            item.setForeground(1, self.QtGui.QBrush(self.QtGui.QColor(*self.COLOR_INCOMPATIBLE)))
            item.setForeground(2, self.QtGui.QBrush(self.QtGui.QColor(*self.COLOR_INCOMPATIBLE)))
        
        if isinstance(parent, self.QtWidgets.QTreeWidget):
            parent.addTopLevelItem(item)
        else:
            parent.addChild(item)
        
        return item


class H5DatasetSelector(DatasetSelector):
    
    def __init__(self, filepath):
        try:
            import h5py as h5
        except ImportError:
            raise ImportError("h5py is required to read HDF5 files. Install it with: pip install h5py")
        
        self.h5 = h5
        self.h5_file = h5.File(filepath, 'r')
        self.compatible_datasets = self._find_compatible_datasets()
        self.compatible_paths = {path for path, _ in self.compatible_datasets}
        super().__init__(filepath, self.compatible_datasets)

        self._detect_complex_interpretation()
    
    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

    def _detect_complex_interpretation(self):
        """
        If there's a single compatible dataset with 'complex' in name and a size-2 dimension,
        auto-set interpret_as_complex to that dimension.
        This is how ISMRMD stores complex data.
        """
        if len(self.compatible_datasets) != 1:
            return
        
        dataset_path, shape = self.compatible_datasets[0]
        
        if 'complex' not in dataset_path.lower():
            return
        
        # Find first dimension with size 2
        for dim_idx, size in enumerate(shape):
            if size == 2:
                self.interpret_as_complex = dim_idx
                print(f"Auto-detected complex interpretation: dimension {dim_idx} of '{dataset_path}' (shape {shape})")
                break

    def _is_compound_dataset(self, dataset):
        """Check if dataset is compound type with array fields."""
        return hasattr(dataset, 'dtype') and dataset.dtype.names is not None

    def _get_compound_field_info(self, dataset, field_name):
        """Get shape and dtype for a compound field. Returns (shape, dtype, is_array)."""
        field_dtype = dataset.dtype.fields[field_name][0]
        
        # Check if field has subdtype (array within compound)
        if hasattr(field_dtype, 'subdtype') and field_dtype.subdtype is not None:
            base_dtype, shape = field_dtype.subdtype
            is_array = isinstance(shape, tuple) and len(shape) >= 1 and np.issubdtype(base_dtype, np.number)
            return (shape, base_dtype, is_array)
        # Scalar field
        return ((1,), field_dtype, False)

    def _find_compatible_datasets(self):
        """Find all compatible datasets and compound array fields."""
        import h5py as h5
        compatible = []

        def visit_func(name, obj):
            if isinstance(obj, h5.Dataset) and hasattr(obj, 'ndim') and obj.ndim >= 1:
                if self._is_compound_dataset(obj):
                    # Only add dataset itself if it has both real and imag fields
                    names = obj.dtype.names or ()
                    real_name = next((n for n in ['real', 'realdata', 'r'] if n in names), None)
                    imag_name = next((n for n in ['imag', 'imagdata', 'i'] if n in names), None)
                    if real_name and imag_name:
                        compatible.append((name, obj.shape))
                    else:
                        # Still allow array fields (e.g. for ISMRMRD style)
                        for field_name in names:
                            shape, dtype, is_array = self._get_compound_field_info(obj, field_name)
                            if is_array:
                                field_path = f"{name}/{field_name}"
                                compatible.append((field_path, shape))
                elif np.issubdtype(obj.dtype, np.number):
                    compatible.append((name, obj.shape))

        self.h5_file.visititems(visit_func)
        return compatible
    
    def get_single_data(self):
        if len(self.compatible_datasets) == 1:
            path = self.compatible_datasets[0][0]
            data = self.load_data(path)
            return (path, data)
        return None
    
    def load_data(self, path):
        # Handle compound field path (dataset/field)
        parts = path.split('/')
        if len(parts) == 2:
            dataset_path, field_name = parts
            dset = self.h5_file[dataset_path]
            
            if field_name in (dset.dtype.names or []):
                field_data = dset[field_name]
                # Remove singleton dataset dimension if present
                if dset.shape == (1,):
                    field_data = field_data[0]
                return np.array(field_data)
        
        # Regular dataset
        dset = self.h5_file[path]

        # Handle native complex arrays
        if np.iscomplexobj(dset):
            return np.array(dset)
        
        if names := dset.dtype.names:
            # Find first matching real and imaginary field names
            real_name = next((n for n in ['real', 'realdata', 'r'] if n in names), None)
            imag_name = next((n for n in ['imag', 'imagdata', 'i'] if n in names), None)
            
            if real_name and imag_name:
                return dset[real_name] + 1j * dset[imag_name]
            elif real_name:
                return np.array(dset[real_name])
        
        return np.array(dset)
    
    def _build_tree(self):
        
        def add_items(parent_item, h5_group, parent_path=''):
            for key in h5_group:
                item = h5_group[key]
                full_path = f"{parent_path}/{key}" if parent_path else key
                
                if isinstance(item, self.h5.Group):
                    group_item = self._add_group(parent_item, key, full_path, "Group")
                    add_items(group_item, item, full_path)
                    
                elif isinstance(item, self.h5.Dataset):
                    is_compatible = full_path in self.compatible_paths
                    
                    # Add compound datasets as groups (like MATLAB structs)
                    if self._is_compound_dataset(item):
                        dataset_item = self._add_group(parent_item, key, full_path, "Compound")
                        dataset_item.setExpanded(True)
                        
                        # Add all compound fields as children
                        for field_name in item.dtype.names:
                            field_path = f"{full_path}/{field_name}"
                            shape, dtype, is_array = self._get_compound_field_info(item, field_name)
                            field_compatible = field_path in self.compatible_paths
                            # Show all fields (arrays as enabled, scalars as disabled)
                            self._add_item(dataset_item, field_name, shape, dtype, 
                                         field_path, field_compatible)
                    else:
                        # Regular dataset
                        self._add_item(parent_item, key, item.shape, item.dtype, 
                                      full_path, is_compatible)
        
        add_items(self.tree, self.h5_file)



class NpzDatasetSelector(DatasetSelector):
    
    def __init__(self, filepath):
        self.npz_file = np.load(filepath)
        compatible_datasets = self._find_compatible_datasets()
        super().__init__(filepath, compatible_datasets)
    
    def __del__(self):
        """Close the NPZ file when selector is destroyed."""
        if hasattr(self, 'npz_file'):
            self.npz_file.close()
    
    def _find_compatible_datasets(self):
        """Find all compatible datasets in the NPZ file."""
        compatible = []
        for key in self.npz_file.keys():
            arr = self.npz_file[key]
            if hasattr(arr, 'ndim') and arr.ndim >= 1:
                compatible.append((key, arr.shape, arr.dtype))
        return compatible
    
    def get_single_data(self):
        if len(self.compatible_datasets) == 1:
            name = self.compatible_datasets[0][0]
            data = self.load_data(name)
            return (name, data)
        return None
    
    def load_data(self, name):
        """Load data for a given array name."""
        return np.asarray(self.npz_file[name]) # npz can be read-only memmap
    
    def _build_tree(self):
        for name, shape, dtype in self.compatible_datasets: # flat and simple
            self._add_item(self.tree, name, shape, dtype, name, compatible=True)


class MatDatasetSelector(DatasetSelector):
    def __init__(self, filepath):
        try:
            import scipy.io
        except ImportError:
            raise ImportError("scipy is required to read .mat files. Install it with: pip install scipy")
        
        mat_data = scipy.io.loadmat(filepath)
        self.mat_dict = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        
        compatible_datasets = self._find_compatible_datasets()
        super().__init__(filepath, compatible_datasets)
    
    def _is_struct(self, val):
        return isinstance(val, np.ndarray) and hasattr(val.dtype, 'names') and val.dtype.names is not None
    
    def _is_numeric_array(self, val):
        return isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number) and val.ndim >= 1
    
    def _has_numeric_data(self, var):
        if not isinstance(var, np.ndarray):
            return False
        
        if self._is_struct(var):
            for field_name in var.dtype.names:
                field_val = var[field_name].item()  # .item() extracts scalar struct
                if self._has_numeric_data(field_val): # recursion
                    return True
            return False
        
        return self._is_numeric_array(var)
    
    def _find_compatible_datasets(self):
        compatible = []
        for key, var in self.mat_dict.items():
            if self._has_numeric_data(var):
                compatible.append((key, var.shape, var.dtype))
        return compatible
    
    def get_single_data(self):
        numeric_vars = [(k, v) for k, v in self.mat_dict.items() if self._is_numeric_array(v)]
        
        if len(numeric_vars) == 1 and len(self.mat_dict) == 1:
            name, data = numeric_vars[0]
            return (name, np.squeeze(data))
        return None
    
    def load_data(self, path):
        """Doesn't actually load, because that's done in ctor. Just navigate the struct."""""
        parts = path.split('/')
        data = self.mat_dict[parts[0]]
        
        # Navigate through struct hierarchy
        for field_name in parts[1:]:
            data = data[field_name].item()  # .item() extracts from (1,1) struct array
        
        return np.atleast_1d(np.squeeze(data))  # Ensure at least 1D
    
    def _build_tree(self):
        for name in sorted(self.mat_dict.keys()):
            self._add_variable(name, self.mat_dict[name], self.tree)
    
    def _add_variable(self, name, val, parent):
        """Add variable to tree (handles structs recursively)."""
        if not isinstance(val, np.ndarray):
            return
        
        if self._is_struct(val):
            # Add struct as group
            item = self._add_group(parent, name, None, "struct")
            # Add all struct fields
            for field_name in val.dtype.names:
                field_val = val[field_name].item()
                field_path = f"{name}/{field_name}"
                self._add_field(field_name, field_val, field_path, item)
        
        elif self._is_numeric_array(val):
            # Add numeric array
            self._add_item(parent, name, val.shape, val.dtype, name, compatible=True)
    
    def _add_field(self, field_name, field_val, field_path, parent):
        """Add struct field to tree (handles nested structs)."""
        if not isinstance(field_val, np.ndarray):
            return
        
        if self._is_struct(field_val):
            # Nested struct
            item = self._add_group(parent, field_name, None, "struct")
            for nested_field in field_val.dtype.names:
                nested_val = field_val[nested_field].item()
                nested_path = f"{field_path}/{nested_field}"
                self._add_field(nested_field, nested_val, nested_path, item)
        
        elif self._is_numeric_array(field_val):
            # Numeric field - compatible
            self._add_item(parent, field_name, field_val.shape, field_val.dtype, field_path, compatible=True)
        
        else:
            # Non-numeric field (string, cell, etc.) - show but disable
            self._add_item(parent, field_name, field_val.shape, field_val.dtype, None, compatible=False)