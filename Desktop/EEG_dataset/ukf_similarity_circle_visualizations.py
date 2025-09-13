# UKF Similarity Visualizations - U: The Mind Company
# Focused on similarity circle visualizations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.signal import welch
import os

# Import existing brand styling functions
from digital_twin_using_neuralmass_model_3 import (
    BrandColors, get_font_weights, load_and_prepare_logo
)

def create_ukf_similarity_circle(real_data, filtered_data, channel, csv_filename, band_name=None, freq_range=None, save_path=None, logo_path="U_logo.png"):
    """
    Create brand-compliant circular similarity visualization for UKF model with data source display
    
    Args:
        real_data: Real EEG data (channels x samples)
        filtered_data: UKF filtered EEG data (channels x samples)
        channel: Channel number to analyze
        csv_filename: Name of the CSV file (e.g., "s15.csv")
        band_name: Name of the frequency band (e.g., "Delta (0.5-4 Hz)")
        freq_range: Tuple of (low_freq, high_freq)
        save_path: Path to save the visualization
        logo_path: Path to company logo
        
    Returns:
        Dictionary containing calculated metrics
    """
    
    # Setup brand-compliant fonts
    get_font_weights()
    
    # Load logo
    logo_img = load_and_prepare_logo(logo_path) if logo_path else None
    
    # Calculate UKF performance metrics
    if channel >= real_data.shape[0] or channel >= filtered_data.shape[0]:
        raise ValueError(f"Channel {channel} out of range")
    
    real_ch = real_data[channel]
    filtered_ch = filtered_data[channel]
    
    # Calculate correlation (main similarity metric)
    correlation, _ = pearsonr(real_ch, filtered_ch)
    similarity_percentage = abs(correlation) * 100
    
    # Calculate additional metrics
    mse = mean_squared_error(real_ch, filtered_ch)
    
    # Power spectrum analysis
    freqs_real, psd_real = welch(real_ch, fs=256, nperseg=512)
    freqs_filt, psd_filtered = welch(filtered_ch, fs=256, nperseg=512)
    
    # Extract data source number from filename (e.g., "s15.csv" -> "s15")
    data_source = csv_filename
    
    # Create figure with brand white background
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BrandColors.WHITE)
    ax.set_facecolor(BrandColors.WHITE)
    
    # Remove axes
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis('off')
    
    # Calculate angle for the arc (similarity percentage)
    angle = (similarity_percentage / 100) * 360
    
    # Draw background circle (brand dark gray)
    bg_circle = Circle((0, 0), 0.75, fill=False, linewidth=20, color=BrandColors.DARK_GRAY, alpha=0.3)
    ax.add_patch(bg_circle)
    
    # Draw similarity arc using brand blue
    if angle > 0:
        wedge = Wedge((0, 0), 0.80, -90, -90 + angle, width=0.095, 
                     facecolor=BrandColors.BLUE, edgecolor=BrandColors.BLUE)
        ax.add_patch(wedge)
    
    # Add percentage text (center, large) - Brand Blue
    ax.text(0, 0.2, f'{similarity_percentage:.0f}%', 
            fontsize=80, fontweight='bold', ha='center', va='center',
            color=BrandColors.BLUE)
    
    # Add "SIMILARITY" text - Brand Black
    ax.text(0, -0.1, 'SIMILARITY', 
            fontsize=24, fontweight='bold', ha='center', va='center',
            color=BrandColors.BLACK)
    
    # Add "TO REAL BRAIN" text - Brand Dark Gray
    ax.text(0, -0.35, 'TO REAL BRAIN SIGNALS', 
            fontsize=16, fontweight='normal', ha='center', va='center',
            color=BrandColors.DARK_GRAY)
    
    # Add quality indicator with brand colors
    if similarity_percentage >= 70:
        quality = "EXCELLENT"
        quality_color = BrandColors.GREEN
    elif similarity_percentage >= 50:
        quality = "GOOD" 
        quality_color = BrandColors.BLUE
    elif similarity_percentage >= 30:
        quality = "FAIR"
        quality_color = BrandColors.ORANGE
    else:
        quality = "POOR"
        quality_color = BrandColors.RED
    
    ax.text(0, -0.6, quality, 
            fontsize=18, fontweight='bold', ha='center', va='center',
            color=quality_color)
    
    # Add band information if provided
    if band_name:
        ax.text(0, 0.93, f'UKF Model Analysis - {band_name}', 
                fontsize=14, fontweight='bold', ha='center', va='center',
                color=BrandColors.BLACK)
    
    if channel is not None:
        ax.text(0, 0.84, f'Channel {channel}', 
                fontsize=12, fontweight='normal', ha='center', va='center',
                color=BrandColors.BLACK)
    
    # Add logo and company name
    if logo_img is not None:
        # Add logo
        imagebox = OffsetImage(logo_img, zoom=0.45)
        ab_logo = AnnotationBbox(imagebox, (-0.95, -1.1), 
                               xycoords='data', frameon=False)
        ax.add_artist(ab_logo)
        
        # Add company name without "U:"
        ax.text(-0.85, -1.1, 'THE MIND COMPANY | Advancing Neurostimulation Technology', 
                fontsize=14, fontweight='normal', ha='left', va='center',
                color=BrandColors.DARK_GRAY)
    else:
        # Fallback to text
        ax.text(0, -1.1, 'U: The Mind Company | Advancing Neurostimulation Technology', 
                fontsize=14, fontweight='normal', ha='center', va='center',
                color=BrandColors.DARK_GRAY)
    
    # Company info with data source - Brand Blue
    ax.text(0, -1.3, f'Data Source: {data_source} | Ohio, USA', 
            fontsize=12, fontweight='normal', ha='center', va='center',
            color=BrandColors.BLUE)
    
    # Save the visualization
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor=BrandColors.WHITE, edgecolor='none')
        print(f"Saved UKF similarity circle: {save_path}")
    
    plt.show()
    plt.close()
    
    return {
        'similarity_percentage': similarity_percentage,
        'correlation': abs(correlation),
        'mse': mse,
        'quality': quality,
        'data_source': data_source,
        'channel': channel,
        'band_name': band_name
    }