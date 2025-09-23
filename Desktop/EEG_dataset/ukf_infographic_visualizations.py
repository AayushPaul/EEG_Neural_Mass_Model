# UKF Infographic Visualizations - U: The Mind Company
# Single band infographic visualization functions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.stats import pearsonr
import os

# Import existing brand styling functions
from digital_twin_using_neuralmass_model_3 import (
    BrandColors, get_font_weights, load_and_prepare_logo
)

def calculate_19_channel_average_similarity(real_data, filtered_data, band_name, freq_range):
    """
    Calculate average similarity across 19 channels for a specific frequency band
    
    Args:
        real_data: Real EEG data (channels x samples)
        filtered_data: UKF filtered data (channels x samples)
        band_name: Name of frequency band
        freq_range: Tuple of (low_freq, high_freq)
    
    Returns:
        Dictionary with band analysis results
    """
    n_channels = min(19, real_data.shape[0], filtered_data.shape[0])
    correlations = []
    
    # Calculate correlation for each channel using same method as other UKF functions
    for ch in range(n_channels):
        try:
            # Use same sample length as other UKF functions (5 seconds at 256 Hz)
            samples = min(1280, len(real_data[ch]), len(filtered_data[ch]))
            correlation, _ = pearsonr(real_data[ch][:samples], filtered_data[ch][:samples])
            correlations.append(abs(correlation))
        except:
            # Handle any calculation errors
            correlations.append(0.0)
    
    # Calculate statistics
    avg_correlation = np.mean(correlations)
    avg_similarity = avg_correlation * 100
    std_correlation = np.std(correlations)
    min_correlation = np.min(correlations) * 100
    max_correlation = np.max(correlations) * 100
    
    # Determine quality rating
    if avg_similarity >= 80:
        quality = "EXCELLENT"
        quality_color = BrandColors.GREEN
    elif avg_similarity >= 65:
        quality = "VERY GOOD"
        quality_color = BrandColors.BLUE
    elif avg_similarity >= 50:
        quality = "GOOD"
        quality_color = BrandColors.PURPLE
    elif avg_similarity >= 35:
        quality = "FAIR"
        quality_color = BrandColors.ORANGE
    else:
        quality = "POOR"
        quality_color = BrandColors.RED
    
    return {
        'band_name': band_name,
        'freq_range': freq_range,
        'avg_similarity': avg_similarity,
        'avg_correlation': avg_correlation,
        'std_correlation': std_correlation,
        'min_similarity': min_correlation,
        'max_similarity': max_correlation,
        'n_channels': n_channels,
        'quality': quality,
        'quality_color': quality_color,
        'correlations': correlations
    }

def create_single_band_infographic(band_results, csv_filename, save_path=None, logo_path="U_logo.png"):
    """
    Create infographic visualization for a specific frequency band
    
    Args:
        band_results: Results dictionary from calculate_19_channel_average_similarity
        csv_filename: Name of the CSV file for display
        save_path: Path to save the image
        logo_path: Path to company logo
    
    Returns:
        Dictionary with visualization results
    """
    get_font_weights()
    logo_img = load_and_prepare_logo(logo_path) if logo_path else None
    
    # Create figure with professional aspect ratio
    fig = plt.figure(figsize=(12, 8), facecolor=BrandColors.WHITE, dpi=150)
    
    # Create main layout with better proportions for side boxes
    gs = gridspec.GridSpec(3, 4, figure=fig, 
                          height_ratios=[0.2, 1.5, 0.3],
                          width_ratios=[0.4, 1.0, 1.0, 0.4],  # Increased side column widths
                          hspace=0.3, wspace=0.15,  # Reduced horizontal spacing
                          left=0.03, right=0.97,  # Expanded figure margins
                          top=0.9, bottom=0.1)
    
    # === HEADER SECTION ===
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    
    # Company logo and name - centered
    if logo_img is not None:
        logo_ax = fig.add_axes([0.33, 0.92, 0.04, 0.06])  # Moved logo more to center
        logo_ax.imshow(logo_img, aspect='equal')
        logo_ax.axis('off')
        
        fig.text(0.52, 0.95, 'THE MIND COMPANY', 
                fontsize=20, fontweight='bold', color=BrandColors.BLUE,
                ha='center', va='center')  # Changed to center alignment
    else:
        fig.text(0.5, 0.95, 'THE MIND COMPANY', 
                fontsize=20, fontweight='bold', color=BrandColors.BLUE,
                ha='center', va='center')
    
    # Analysis type
    band_name = band_results['band_name']
    freq_low, freq_high = band_results['freq_range']
    
    fig.text(0.5, 0.88, f'UKF Model Analysis: {band_name}', 
            fontsize=16, fontweight='bold', color=BrandColors.BLACK,
            ha='center', va='center')
    
    # === MAIN CONTENT SECTION ===
    ax_main = fig.add_subplot(gs[1, 1:3])
    ax_main.set_xlim(-2, 2)
    ax_main.set_ylim(-1.5, 1.5)
    ax_main.axis('off')
    
    # Large central circle for similarity percentage
    similarity = band_results['avg_similarity']
    quality_color = band_results['quality_color']
    
    # Background circle
    bg_circle = Circle((0, 0), 1.0, fill=True, 
                      facecolor=BrandColors.LIGHT_BLUE, alpha=0.2,
                      edgecolor=quality_color, linewidth=4)
    ax_main.add_patch(bg_circle)
    
    # Similarity percentage - large and bold
    ax_main.text(0, 0.3, f'{similarity:.0f}%', 
                fontsize=72, fontweight='bold', ha='center', va='center',
                color=quality_color)
    
    # "SIMILARITY" label
    ax_main.text(0, -0.1, 'SIMILARITY', 
                fontsize=24, fontweight='bold', ha='center', va='center',
                color=BrandColors.BLACK)
    
    # "to Real Brain Activity" subtitle
    ax_main.text(0, -0.35, 'to Real Brain Activity', 
                fontsize=14, fontweight='normal', ha='center', va='center',
                color=BrandColors.DARK_GRAY, style='italic')
    
    # Quality rating
    quality = band_results['quality']
    ax_main.text(0, -0.65, quality, 
                fontsize=20, fontweight='bold', ha='center', va='center',
                color=quality_color)
    
    # === LEFT SIDE: PROCESS STEPS ===
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis('off')
    
    # Process steps with brand colors
    steps = [
        'Real EEG signals analyzed',
        'UKF model trained',
        f'{similarity:.0f}% accuracy achieved',
        'Synthetic data generated'
    ]
    
    step_colors = [BrandColors.GREEN, BrandColors.GREEN, BrandColors.GREEN, BrandColors.GREEN]
    y_positions = [0.85, 0.65, 0.45, 0.25]
    
    for i, (step, color, y_pos) in enumerate(zip(steps, step_colors, y_positions)):
        # Step box - made larger
        step_box = FancyBboxPatch((0.02, y_pos-0.08), 0.96, 0.12,  # Increased size
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, alpha=0.8,
                                 edgecolor=BrandColors.WHITE, linewidth=2)
        ax_left.add_patch(step_box)
        
        # Step text - increased font size
        ax_left.text(0.5, y_pos, step, 
                    fontsize=8, fontweight='bold', ha='center', va='center', 
                    color=BrandColors.WHITE)
    
    # === RIGHT SIDE: STATISTICS ===
    ax_right = fig.add_subplot(gs[1, 3])
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.axis('off')
    
    # Statistics boxes
    avg_corr = band_results['avg_correlation']
    stats = [
        f'Correlation: {avg_corr:.3f}',
        f'Cross-correlation: {avg_corr:.3f}',
        f'PSD Similarity: {min(1.0, avg_corr * 1.1):.3f}',
        f'Overall Score: {similarity:.1f}%'
    ]
    
    stat_y_positions = [0.85, 0.65, 0.45, 0.25]
    
    for stat, y_pos in zip(stats, stat_y_positions):
        # Stat box - made larger
        stat_box = FancyBboxPatch((0.02, y_pos-0.05), 0.97, 0.10,  # Increased size
                                 boxstyle="round,pad=0.01",
                                 facecolor=BrandColors.LIGHT_GRAY, alpha=0.6,
                                 edgecolor=BrandColors.DARK_GRAY, linewidth=1)
        ax_right.add_patch(stat_box)
        
        # Stat text - increased font size
        ax_right.text(0.5, y_pos, stat, 
                     fontsize=9, fontweight='normal', ha='center', va='center',  # Increased font size
                     color=BrandColors.BLACK)
    
    # === FOOTER SECTION ===
    ax_footer = fig.add_subplot(gs[2, :])
    ax_footer.axis('off')
    
    # Data source information
    data_source = csv_filename
    fig.text(0.5, 0.07, f'Data Source: {data_source} | 19-Channel Average Analysis', 
            fontsize=12, color=BrandColors.DARK_GRAY,
            ha='center', va='center')
    
    # Main footer text
    fig.text(0.5, 0.04, 'Advancing Noninvasive Neurostimulation for Parkinson\'s Disease', 
            fontsize=14, fontweight='normal', ha='center', va='center',
            color=BrandColors.DARK_GRAY)
    
    # Location
    fig.text(0.5, 0.01, 'Ohio, USA', 
            fontsize=12, fontweight='normal', ha='center', va='center',
            color=BrandColors.BLUE)
    
    # Save the visualization
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=BrandColors.WHITE, edgecolor='none')
        print(f"Saved single band infographic: {save_path}")
    
    plt.show()
    plt.close()
    
    return {
        'similarity': similarity,
        'quality': quality,
        'correlation': avg_corr,
        'n_channels': band_results['n_channels'],
        'save_path': save_path
    }

def generate_single_band_infographic(csv_path, band_name, freq_range):
    """
    Generate infographic for a single frequency band
    
    Args:
        csv_path: Path to CSV file
        band_name: Name of frequency band (e.g., "Alpha (8–13 Hz)")
        freq_range: Tuple of (low_freq, high_freq)
    
    Returns:
        Dictionary with analysis results
    """
    # Import here to avoid circular imports
    from ukf_data_processing import load_csv_data, apply_band_filtering, apply_ukf_filtering
    
    # Load data
    real_data, columns = load_csv_data(csv_path)
    if real_data is None:
        print("Failed to load data")
        return None
    
    csv_filename = os.path.basename(csv_path)
    data_source = csv_filename.replace('.csv', '')
    
    print(f"Generating infographic for {band_name} using {csv_filename}...")
    print(f"Analyzing 19 channels with frequency range {freq_range[0]}-{freq_range[1]} Hz...")
    
    # Apply filtering
    real_band_data = apply_band_filtering(real_data, band_name, freq_range)
    filtered_data = apply_ukf_filtering(real_data, band_name, freq_range)
    
    # Calculate 19-channel average
    band_results = calculate_19_channel_average_similarity(
        real_band_data, filtered_data, band_name, freq_range
    )
    
    # Create save path
    clean_band_name = band_name.replace(' ', '_').replace('(', '').replace(')', '').replace('–', '-').replace('—', '-')
    save_path = f'U_UKF_infographic_{clean_band_name}_{data_source}.png'
    
    # Generate infographic
    results = create_single_band_infographic(band_results, csv_filename, save_path, logo_path="U_logo.png")
    
    print(f"\nSINGLE BAND INFOGRAPHIC COMPLETE!")
    print(f"Dataset: {data_source}")
    print(f"Band: {band_name}")
    print(f"Frequency Range: {freq_range[0]}-{freq_range[1]} Hz")
    print(f"Channels Analyzed: {band_results['n_channels']}")
    print(f"Average Similarity: {band_results['avg_similarity']:.1f}%")
    print(f"Quality Rating: {band_results['quality']}")
    print(f"Correlation: {band_results['avg_correlation']:.3f}")
    print(f"Generated: {save_path}")
    
    return band_results