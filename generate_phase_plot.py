
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure aesthetics for Nature/NeurIPS standard
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.linewidth'] = 1.5

def generate_hero_image(output_path: str):
    # 1. Setup Data
    t = np.linspace(0, 50, 2000)
    
    # Parameters
    gamma_benign = 0.005  # Very slow decay (Stable Memory)
    gamma_attack = 0.15   # Fast decay (Spectral Collapse)
    t_defense = 20        # Defense kicks in at t=20
    
    # Benign Trajectory (Blue Spiral)
    # Helix moving up Z, maintaining radius
    r_benign = np.exp(-gamma_benign * t)
    x_benign = r_benign * np.cos(t)
    y_benign = r_benign * np.sin(t)
    z_benign = t
    
    # Attack Trajectory (Red Collapse)
    # Spirals quickly into the center
    r_attack = np.exp(-gamma_attack * t)
    x_attack = r_attack * np.cos(t)
    y_attack = r_attack * np.sin(t)
    z_attack = t
    
    # Defense Trajectory (Green Shield)
    # Follows attack until threshold, then stops (gating)
    defense_mask = t <= t_defense
    x_defense = x_attack[defense_mask]
    y_defense = y_attack[defense_mask]
    z_defense = z_attack[defense_mask]
    
    # 2. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Trajectories
    # Benign
    ax.plot(x_benign, y_benign, z_benign, 
            color='#2980b9', alpha=0.6, linewidth=1.5, 
            label=r'Benign Dynamics ($\rho \approx 1$)')
            
    # Attack
    ax.plot(x_attack, y_attack, z_attack, 
            color='#c0392b', linestyle='--', alpha=0.8, linewidth=1.5,
            label=r'Adversarial Attack ($\rho \to 0$)')
            
    # Defense (Active Shield)
    # Draw a marker at the cutoff point
    ax.plot(x_defense, y_defense, z_defense, 
            color='#27ae60', linewidth=2.5, 
            label=r'SpectralGuard (Defense)')
            
    ax.scatter(x_defense[-1], y_defense[-1], z_defense[-1], 
               color='#27ae60', s=50, marker='o', edgecolors='white', zorder=10)

    # 3. Aesthetics
    
    # Remove pane backgrounds for clean look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    
    # Axis labels
    ax.set_xlabel('State Dim 1 ($h_1$)', labelpad=10)
    ax.set_ylabel('State Dim 2 ($h_2$)', labelpad=10)
    ax.set_zlabel('Time / Sequence Level ($t$)', labelpad=10)
    
    # View Angle
    ax.view_init(elev=25, azim=-45)
    
    # Legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
    legend.get_frame().set_edgecolor('gray')
    
    # Clean Axis ticks (optional, makes it look more abstract/theoretical)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    
    plt.tight_layout()
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, transparent=False, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"Image saved to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 3D collapse plot without undefined legend terms.")
    parser.add_argument(
        "--output",
        default="mamba_spectral/Estado_atual/Jj3rAqE4.png",
        help="Output image path (defaults to canonical paper filename).",
    )
    args = parser.parse_args()
    generate_hero_image(output_path=args.output)
