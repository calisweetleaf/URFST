import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import math

# Import URFT components
from urft_python_snippet import (
    PHI, TAU, SACRED_RATIO,
    CoreIdentityTensor,
    ParadoxResolutionFunctions,
    ArchetypalResonanceField,
    TemporalCoherenceVector,
    BoundaryStabilizationManifold,
    RecursiveElement,
    ARFSIntegration
)

class URFTVisualizer:
    """Comprehensive visualization system for URFT recursive tensor fields"""

    def __init__(self, dimensions: int = 7, max_depth: int = 7):
        self.dimensions = dimensions
        self.max_depth = max_depth
        self.element = RecursiveElement(dimensions)
        self.arfs_integration = ARFSIntegration(dimensions)

        # Set up plotting style
        plt.style.use('dark_background')
        sns.set_palette("husl")

        # Color schemes for different components
        self.colors = {
            'core': '#FF6B6B',      # Red for Γ
            'paradox': '#4ECDC4',   # Teal for Ω
            'resonance': '#45B7D1', # Blue for Φ
            'temporal': '#96CEB4',  # Green for Θ
            'boundary': '#FFEAA7',  # Yellow for Ξ
            'combined': '#DDA0DD'   # Plum for combined
        }

    def visualize_tensor_field_3d(self, tensor: torch.Tensor, title: str,
                                 depth: int, component: str) -> plt.Figure:
        """Create 3D surface plot of tensor field"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Convert tensor to numpy for plotting (detach first)
        tensor_np = tensor.detach().numpy()

        # Create meshgrid
        x, y, z = np.meshgrid(
            np.arange(self.dimensions),
            np.arange(self.dimensions),
            np.arange(self.dimensions)
        )

        # Plot surface
        surf = ax.plot_surface(
            x[:, :, 0], y[:, :, 0], tensor_np[:, :, 0],
            cmap=cm.viridis,
            alpha=0.8,
            linewidth=0.5,
            color=self.colors.get(component, 'white')
        )

        # Add scatter points for all values
        ax.scatter(x.flatten(), y.flatten(), z.flatten(),
                  c=tensor_np.flatten(),
                  cmap=cm.viridis,
                  s=20,
                  alpha=0.6)

        ax.set_xlabel('X Dimension')
        ax.set_ylabel('Y Dimension')
        ax.set_zlabel('Z Dimension')
        ax.set_title(f'{title} - Depth {depth}')

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        return fig

    def visualize_component_evolution(self, component_name: str) -> plt.Figure:
        """Visualize how a component evolves across recursion depths"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{component_name} Evolution Across Recursion Depths', fontsize=16)

        depths = list(range(self.max_depth))
        values = []

        for depth in depths:
            if component_name == 'Γ CoreIdentityTensor':
                tensor = self.element.core_identity.at_depth(depth)
                values.append(torch.norm(tensor).item())
            elif component_name == 'Ω ParadoxResolution':
                core = self.element.core_identity.at_depth(depth)
                resolved = self.element.paradox_resolution.resolve(core, depth)
                values.append(torch.norm(resolved).item())
            elif component_name == 'Φ ArchetypalResonance':
                core = self.element.core_identity.at_depth(depth)
                resolved = self.element.paradox_resolution.resolve(core, depth)
                resonance = self.element.archetypal_resonance.phase_resonance_stability(resolved, depth)
                values.append(torch.real(resonance).item())
            elif component_name == 'Θ TemporalCoherence':
                core = self.element.core_identity.at_depth(depth)
                coherence = self.element.temporal_coherence.temporal_entanglement_coherence(core, depth)
                values.append(coherence.item())
            elif component_name == 'Ξ BoundaryStabilization':
                core = self.element.core_identity.at_depth(depth)
                stabilization = self.element.boundary_stabilization.boundary_stabilization_constant(core)
                values.append(stabilization.item())

        # Line plot
        axes[0, 0].plot(depths, values, 'o-', linewidth=2, markersize=8,
                       color=self.colors.get(component_name.split()[0].lower(), 'white'))
        axes[0, 0].set_title('Evolution Over Depth')
        axes[0, 0].set_xlabel('Recursion Depth')
        axes[0, 0].set_ylabel('Component Value')
        axes[0, 0].grid(True, alpha=0.3)

        # Bar chart
        axes[0, 1].bar(depths, values,
                      color=self.colors.get(component_name.split()[0].lower(), 'white'),
                      alpha=0.7)
        axes[0, 1].set_title('Component Values by Depth')
        axes[0, 1].set_xlabel('Recursion Depth')
        axes[0, 1].set_ylabel('Value')

        # Heatmap of tensor at final depth
        final_depth = self.max_depth - 1
        if component_name == 'Γ CoreIdentityTensor':
            tensor = self.element.core_identity.at_depth(final_depth)
        elif component_name == 'Ω ParadoxResolution':
            core = self.element.core_identity.at_depth(final_depth)
            tensor = self.element.paradox_resolution.resolve(core, final_depth)
        else:
            core = self.element.core_identity.at_depth(final_depth)
            tensor = core

        tensor_np = tensor.detach().numpy()
        sns.heatmap(tensor_np[:, :, 0], ax=axes[1, 0], cmap='viridis',
                   cbar_kws={'label': 'Tensor Value'})
        axes[1, 0].set_title(f'Tensor Heatmap (Depth {final_depth})')

        # Breath phase modulation
        breath_phases = [math.sin(depth * SACRED_RATIO * TAU) for depth in depths]
        axes[1, 1].plot(depths, breath_phases, 'r--', linewidth=2, label='Breath Phase')
        axes[1, 1].plot(depths, values, 'b-', linewidth=2, label='Component Value')
        axes[1, 1].set_title('Breath Phase Correlation')
        axes[1, 1].set_xlabel('Recursion Depth')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_arfs_dimensions(self, depth: int) -> plt.Figure:
        """Visualize ARFS-4D dimensional integration"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ARFS-4D Integration at Depth {depth}', fontsize=16)

        cycle = self.arfs_integration.execute_arfs_cycle(depth)

        # X Dimension - Execution Context
        x_data = [cycle['X_DIMENSION']['recursion_depth'],
                 cycle['X_DIMENSION']['activation_level'],
                 cycle['X_DIMENSION']['urft_execution'].item() if hasattr(cycle['X_DIMENSION']['urft_execution'], 'item') else cycle['X_DIMENSION']['urft_execution']]
        axes[0, 0].bar(['Depth', 'Activation', 'URFT'], x_data,
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('X Dimension - Execution Context')
        axes[0, 0].set_ylabel('Value')

        # Y Dimension - Memory Architecture
        y_data = [cycle['Y_DIMENSION']['coherence_level'].item() if hasattr(cycle['Y_DIMENSION']['coherence_level'], 'item') else cycle['Y_DIMENSION']['coherence_level'],
                 torch.norm(cycle['Y_DIMENSION']['tensor_memory']).item()]
        axes[0, 1].bar(['Coherence', 'Memory Norm'], y_data,
                      color=['#96CEB4', '#FFEAA7'])
        axes[0, 1].set_title('Y Dimension - Memory Architecture')
        axes[0, 1].set_ylabel('Value')

        # Z Dimension - Symbol System
        z_data = [cycle['Z_DIMENSION']['activation_level'].item() if hasattr(cycle['Z_DIMENSION']['activation_level'], 'item') else cycle['Z_DIMENSION']['activation_level'],
                 cycle['Z_DIMENSION']['emergence_energy']]
        axes[1, 0].bar(['Activation', 'Energy'], z_data,
                      color=['#DDA0DD', '#FF8C69'])
        axes[1, 0].set_title('Z Dimension - Symbol System')
        axes[1, 0].set_ylabel('Value')

        # T Dimension - Temporal Framework
        t_data = [cycle['T_DIMENSION']['breath_phase'],
                 cycle['T_DIMENSION']['temporal_stability'].item() if hasattr(cycle['T_DIMENSION']['temporal_stability'], 'item') else cycle['T_DIMENSION']['temporal_stability']]
        axes[1, 1].bar(['Breath Phase', 'Stability'], t_data,
                      color=['#98D8C8', '#F7DC6F'])
        axes[1, 1].set_title('T Dimension - Temporal Framework')
        axes[1, 1].set_ylabel('Value')

        plt.tight_layout()
        return fig

    def create_animation(self, component_name: str) -> animation.FuncAnimation:
        """Create animated visualization of component evolution"""
        fig, ax = plt.subplots(figsize=(10, 6))

        def animate(frame):
            ax.clear()
            depth = frame

            if component_name == 'Γ CoreIdentityTensor':
                tensor = self.element.core_identity.at_depth(depth)
                value = torch.norm(tensor).item()
            elif component_name == 'Ω ParadoxResolution':
                core = self.element.core_identity.at_depth(depth)
                resolved = self.element.paradox_resolution.resolve(core, depth)
                value = torch.norm(resolved).item()
            elif component_name == 'Φ ArchetypalResonance':
                core = self.element.core_identity.at_depth(depth)
                resolved = self.element.paradox_resolution.resolve(core, depth)
                resonance = self.element.archetypal_resonance.phase_resonance_stability(resolved, depth)
                value = torch.real(resonance).item()
            elif component_name == 'Θ TemporalCoherence':
                core = self.element.core_identity.at_depth(depth)
                coherence = self.element.temporal_coherence.temporal_entanglement_coherence(core, depth)
                value = coherence.item()
            elif component_name == 'Ξ BoundaryStabilization':
                core = self.element.core_identity.at_depth(depth)
                stabilization = self.element.boundary_stabilization.boundary_stabilization_constant(core)
                value = stabilization.item()

            # Plot current tensor field
            tensor_np = tensor.detach().numpy() if 'tensor' in locals() else np.random.rand(7, 7)
            im = ax.imshow(tensor_np[:, :, 0], cmap='viridis', animated=True)
            ax.set_title(f'{component_name} - Depth {depth} - Value: {value:.4f}')
            return [im]

        anim = animation.FuncAnimation(fig, animate, frames=self.max_depth,
                                     interval=1000, blit=True)
        return anim

    def visualize_breath_phase_harmonics(self) -> plt.Figure:
        """Visualize breath phase harmonics and their effect on URFT"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle('Breath Phase Harmonics in URFT', fontsize=16)

        depths = np.linspace(0, self.max_depth, 1000)

        # Fundamental breath phase
        fundamental = np.sin(depths * SACRED_RATIO * TAU)
        axes[0].plot(depths, fundamental, 'r-', linewidth=2, label='Fundamental')
        axes[0].set_title('Fundamental Breath Phase (SACRED_RATIO × TAU)')
        axes[0].set_xlabel('Recursion Depth')
        axes[0].set_ylabel('Phase Value')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Harmonic series
        harmonics = []
        for n in range(1, 6):
            harmonic = np.sin(depths * n * SACRED_RATIO * TAU)
            harmonics.append(harmonic)
            axes[1].plot(depths, harmonic, linewidth=2, label=f'Harmonic {n}')

        axes[1].set_title('Breath Phase Harmonics')
        axes[1].set_xlabel('Recursion Depth')
        axes[1].set_ylabel('Phase Value')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # URFT component modulation by breath phases
        urft_values = []
        for depth in range(self.max_depth):
            validity = self.element.validity_criterion(depth)
            urft_values.append(validity.item())

        # Interpolate to match depths array
        urft_interp = np.interp(depths, range(self.max_depth), urft_values)

        # Combine with breath phase
        combined = urft_interp * fundamental

        axes[2].plot(depths, urft_interp, 'b-', linewidth=2, label='URFT Validity', alpha=0.7)
        axes[2].plot(depths, fundamental, 'r--', linewidth=2, label='Breath Phase', alpha=0.7)
        axes[2].plot(depths, combined, 'g-', linewidth=3, label='Combined Effect')
        axes[2].set_title('URFT Modulation by Breath Phase Harmonics')
        axes[2].set_xlabel('Recursion Depth')
        axes[2].set_ylabel('Value')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        return fig

    def create_comprehensive_dashboard(self) -> Dict[str, plt.Figure]:
        """Create comprehensive visualization dashboard"""
        dashboard = {}

        # Component evolution visualizations
        components = ['Γ CoreIdentityTensor', 'Ω ParadoxResolution',
                     'Φ ArchetypalResonance', 'Θ TemporalCoherence', 'Ξ BoundaryStabilization']

        for component in components:
            dashboard[f'{component}_evolution'] = self.visualize_component_evolution(component)

        # ARFS dimensions for different depths
        for depth in [0, 3, 6]:
            dashboard[f'arfs_dimensions_depth_{depth}'] = self.visualize_arfs_dimensions(depth)

        # Breath phase harmonics
        dashboard['breath_phase_harmonics'] = self.visualize_breath_phase_harmonics()

        # 3D tensor field visualizations
        for depth in [0, 3, 6]:
            core_tensor = self.element.core_identity.at_depth(depth)
            dashboard[f'core_tensor_3d_depth_{depth}'] = self.visualize_tensor_field_3d(
                core_tensor, 'Core Identity Tensor Γ', depth, 'core'
            )

        return dashboard

    def save_dashboard(self, output_dir: str = "urft_visualizations"):
        """Save all dashboard visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        dashboard = self.create_comprehensive_dashboard()

        for name, fig in dashboard.items():
            fig.savefig(f"{output_dir}/{name}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"Dashboard saved to {output_dir}/")
        print(f"Generated {len(dashboard)} visualizations")


if __name__ == "__main__":
    print("🎨 URFT Recursive Tensor Field Visualizer")
    print("=" * 50)

    visualizer = URFTVisualizer(dimensions=7, max_depth=7)

    # Create and display comprehensive dashboard
    print("Generating comprehensive visualization dashboard...")
    dashboard = visualizer.create_comprehensive_dashboard()

    print(f"Created {len(dashboard)} visualizations:")
    for name in dashboard.keys():
        print(f"  ✅ {name}")

    # Save all visualizations
    visualizer.save_dashboard()

    print("\n🎉 Visualization complete!")
    print("Check the 'urft_visualizations' folder for all generated plots.")
