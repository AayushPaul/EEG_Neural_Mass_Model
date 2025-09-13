# UKF Channel Comparison Visualizations - U: The Mind Company
# Focused on channel comparison visualizations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import os

# Import existing brand styling functions
from digital_twin_using_neuralmass_model_3 import (
    BrandColors, get_font_weights, load_and_prepare_logo
)

def create_ukf_channel_comparison(real_data, filtered_data, channel_range, csv_filename,
                                 save_path=None, logo_path="U_logo.png", band_name=None):
    """Create multi-channel UKF comparison visualization in neural mass model style"""
    get_font_weights()
    logo_img = load_and_prepare_logo(logo_path) if logo_path else None
    
    start_ch, end_ch = channel_range
    n_channels = end_ch - start_ch + 1
    
    # DYNAMIC LAYOUT CALCULATION
    # Calculate header height needed based on content
    base_header_height = 0.15  # Base header size
    text_elements = 4  # Title, subtitle, data source, correlation
    text_spacing = 0.03  # Space between text elements
    needed_header_height = base_header_height + (text_elements * text_spacing)
    
    # Ensure minimum spacing regardless of channel count
    header_height = max(needed_header_height, 0.25)
    
    # Calculate top margin to accommodate header
    top_margin = 1.0 - header_height
    
    # Create figure with better proportions for neural mass model style
    fig = plt.figure(figsize=(14, 3 + n_channels * 1.2), facecolor=BrandColors.WHITE, dpi=150)
    
    # Create grid layout with dynamic top margin
    gs = gridspec.GridSpec(n_channels, 2, figure=fig,
                          hspace=0.35, wspace=0.25,
                          left=0.12, right=0.95,
                          top=top_margin, bottom=0.15)
    
    # === ENHANCED HEADER SECTION ===
    # Create header area that scales with content
    header_ax = fig.add_axes([0, top_margin, 1, header_height])
    header_ax.axis('off')
    
    # Logo placement - adjust based on header size and number of channels
    if logo_img is not None:
        logo_y = top_margin + (header_height * 0.8)
        
        # Dynamic logo size based on number of channels
        if n_channels <= 3:
            logo_width, logo_height = 0.055, 0.055  # Larger for fewer channels
        elif n_channels <= 5:
            logo_width, logo_height = 0.045, 0.045    # Medium size
        else:
            logo_width, logo_height = 0.035, 0.035    # Original size for many channels
        
        logo_ax = fig.add_axes([0.34, logo_y, logo_width, logo_height])
        logo_ax.imshow(logo_img, aspect='equal')
        logo_ax.axis('off')
        
        # Use relative positioning within header area
        title_y = 0.85
        subtitle_y = 0.65
        data_source_y = 0.45
        correlation_y = 0.25
    else:
        # Without logo, use slightly different positioning
        title_y = 0.85
        subtitle_y = 0.65
        data_source_y = 0.45
        correlation_y = 0.25
    
    # Main title with company logo
    if logo_img is not None:
        header_ax.text(0.5, title_y, 'THE MIND COMPANY', transform=header_ax.transAxes,
                    fontsize=18, fontweight='bold', color=BrandColors.BLUE,
                    ha='center', va='center')
    else: 
        header_ax.text(0.5, title_y, 'U: THE MIND COMPANY', transform=header_ax.transAxes,
                    fontsize=18, fontweight='bold', color=BrandColors.BLUE,
                    ha='center', va='center')
    
    # Subtitle with model and analysis info
    if band_name:
        header_ax.text(0.5, subtitle_y, f'UKF Model Analysis - {band_name} (Channels {start_ch}-{end_ch})', 
                      transform=header_ax.transAxes,
                      fontsize=14, fontweight='bold', color=BrandColors.BLACK,
                      ha='center', va='center')
    else:
        header_ax.text(0.5, subtitle_y, f'UKF Model Analysis (Channels {start_ch}-{end_ch})', 
                      transform=header_ax.transAxes,
                      fontsize=14, fontweight='bold', color=BrandColors.BLACK,
                      ha='center', va='center')
    
    # Data source info
    header_ax.text(0.5, data_source_y, f'Data Source: {csv_filename}', 
                  transform=header_ax.transAxes,
                  fontsize=12, color=BrandColors.DARK_GRAY,
                  ha='center', va='center')
    
    # Time axis - show 5 seconds like neural mass model
    time_axis = np.arange(min(1280, real_data.shape[1])) / 256  # 5 seconds at 256 Hz
    
    overall_correlations = []
    
    # CALCULATE GLOBAL Y-AXIS LIMITS FOR CONSISTENT SCALING
    all_real_values = []
    all_filtered_values = []
    
    for ch in range(start_ch, end_ch + 1):
        if ch >= real_data.shape[0]:
            break
        samples = min(1280, len(real_data[ch]))
        all_real_values.extend(real_data[ch][:samples])
        all_filtered_values.extend(filtered_data[ch][:samples])
    
    # Calculate global y-limits with some padding
    global_min = min(min(all_real_values), min(all_filtered_values))
    global_max = max(max(all_real_values), max(all_filtered_values))
    y_range = global_max - global_min
    y_padding = y_range * 0.1  # 10% padding
    global_ylim = (global_min - y_padding, global_max + y_padding)
    
    # Create subplots for each channel - 2 columns like neural mass model
    for i, ch in enumerate(range(start_ch, end_ch + 1)):
        if ch >= real_data.shape[0]:
            break
            
        # === LEFT COLUMN: Original EEG (Real) ===
        ax_orig = fig.add_subplot(gs[i, 0])
        samples = min(1280, len(real_data[ch]))
        ax_orig.plot(time_axis[:samples], real_data[ch][:samples], 
                    color=BrandColors.BLUE, linewidth=1.5, alpha=0.9)
        
        # Channel label in top-left corner
        ax_orig.text(0.05, 0.95, f'Original Ch{ch}', transform=ax_orig.transAxes, 
                    fontsize=11, fontweight='bold', color=BrandColors.BLUE,
                    ha='left', va='top', 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=BrandColors.WHITE, 
                             edgecolor=BrandColors.BLUE, alpha=0.9, linewidth=1))
        
        # Set axis limits and labels
        ax_orig.set_xlim(0, 5)
        ax_orig.set_ylim(global_ylim)
        
        # Add y-axis label only to the middle channel
        if i == n_channels // 2:
            ax_orig.set_ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')
        
        # === RIGHT COLUMN: UKF Filtered (AI) ===
        ax_filt = fig.add_subplot(gs[i, 1])
        filtered_samples = min(1280, len(filtered_data[ch]))
        ax_filt.plot(time_axis[:filtered_samples], filtered_data[ch][:filtered_samples], 
                    color=BrandColors.PURPLE, linewidth=1.5, alpha=0.9)
        
        # Calculate correlation
        corr, _ = pearsonr(real_data[ch][:samples], filtered_data[ch][:samples])
        overall_correlations.append(abs(corr))
        
        # Channel label with correlation in box like neural mass model
        ax_filt.text(0.05, 0.95, f'UKF Ch{ch} (Correlation={corr:.3f})', 
                    transform=ax_filt.transAxes, 
                    fontsize=11, fontweight='bold', color=BrandColors.PURPLE,
                    ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=BrandColors.WHITE, 
                             edgecolor=BrandColors.PURPLE, alpha=0.9, linewidth=1))
        
        # Set axis limits and labels
        ax_filt.set_xlim(0, 5)
        ax_filt.set_ylim(global_ylim)
        
        # Styling for both plots
        for ax in [ax_orig, ax_filt]:
            ax.set_facecolor(BrandColors.WHITE)
            ax.spines['left'].set_color(BrandColors.DARK_GRAY)
            ax.spines['bottom'].set_color(BrandColors.DARK_GRAY)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors=BrandColors.BLACK, labelsize=9)
            ax.grid(True, alpha=0.3, color=BrandColors.LIGHT_GRAY)
        
        # X-axis labels only on bottom row
        if i == n_channels - 1:
            ax_orig.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
            ax_filt.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        else:
            ax_orig.set_xticklabels([])
            ax_filt.set_xticklabels([])
    
    # === OVERALL CORRELATION DISPLAY ===
    avg_correlation = np.mean(overall_correlations)
    
    # Add overall correlation info in header area at reserved position
    header_ax.text(0.5, correlation_y, f'Overall Correlation: {avg_correlation:.3f} ({avg_correlation*100:.1f}%)', 
                  transform=header_ax.transAxes,
                  fontsize=14, fontweight='bold', 
                  color=BrandColors.GREEN if avg_correlation > 0.7 else BrandColors.ORANGE if avg_correlation > 0.3 else BrandColors.RED,
                  ha='center', va='center',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=BrandColors.LIGHT_GRAY, 
                           edgecolor=BrandColors.DARK_GRAY, alpha=0.8, linewidth=1))
    
    # Brand footer
    fig.text(0.5, 0.02, 'U: The Mind Company | Ohio, USA | Advancing Neurostimulation Technology', 
             ha='center', fontsize=10, color=BrandColors.DARK_GRAY, style='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=BrandColors.WHITE, edgecolor='none')
        print(f"Saved UKF channel comparison: {save_path}")
    
    plt.show()
    plt.close()
    
    return avg_correlation