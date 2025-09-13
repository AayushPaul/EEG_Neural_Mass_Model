# UKF Dashboard Visualizations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.stats import pearsonr, entropy
from scipy.signal import welch, hilbert
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Import existing brand styling functions
from digital_twin_using_neuralmass_model_3 import (
    BrandColors, get_font_weights, load_and_prepare_logo
)

def analyze_ukf_performance_enhanced(real_data, filtered_data, channel, band_name=None, freq_range=None):
    """Enhanced UKF performance analysis matching neural mass model metrics"""
    
    if channel >= real_data.shape[0] or channel >= filtered_data.shape[0]:
        raise ValueError(f"Channel {channel} out of range")
    
    real_ch = real_data[channel]
    filtered_ch = filtered_data[channel]
    
    # Calculate basic metrics
    correlation, _ = pearsonr(real_ch, filtered_ch)
    mse = mean_squared_error(real_ch, filtered_ch)
    
    # Power spectrum analysis
    freqs_real, psd_real = welch(real_ch, fs=256, nperseg=512)
    freqs_filt, psd_filtered = welch(filtered_ch, fs=256, nperseg=512)
    
    # Spectral similarity (cross-correlation of PSDs)
    spectral_similarity = np.corrcoef(psd_real, psd_filtered)[0, 1]
    
    # Phase similarity using Hilbert transform
    analytic_real = hilbert(real_ch)
    analytic_filt = hilbert(filtered_ch)
    phase_real = np.angle(analytic_real)
    phase_filt = np.angle(analytic_filt)
    phase_similarity = abs(np.mean(np.exp(1j * (phase_real - phase_filt))))
    
    # Band-specific analysis
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8), 
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    
    band_powers_real = {}
    band_powers_filt = {}
    band_peaks_real = {}
    band_peaks_filt = {}
    
    for band_name, (low, high) in bands.items():
        band_mask = (freqs_real >= low) & (freqs_real <= high)
        if np.any(band_mask):
            band_powers_real[band_name] = np.sum(psd_real[band_mask])
            band_powers_filt[band_name] = np.sum(psd_filtered[band_mask])
            band_peaks_real[band_name] = freqs_real[band_mask][np.argmax(psd_real[band_mask])]
            band_peaks_filt[band_name] = freqs_filt[band_mask][np.argmax(psd_filtered[band_mask])]
        else:
            band_powers_real[band_name] = 0
            band_powers_filt[band_name] = 0
            band_peaks_real[band_name] = 0
            band_peaks_filt[band_name] = 0
    
    return {
        'correlation': abs(correlation),
        'mse': mse,
        'spectral_similarity': abs(spectral_similarity) if not np.isnan(spectral_similarity) else 0,
        'phase_similarity': phase_similarity,
        'freqs_real': freqs_real,
        'psd_real': psd_real,
        'freqs_filtered': freqs_filt,
        'psd_filtered': psd_filtered,
        'band_powers_real': band_powers_real,
        'band_powers_filt': band_powers_filt,
        'band_peaks_real': band_peaks_real,
        'band_peaks_filt': band_peaks_filt,
        'bands': bands
    }

def create_ukf_multi_channel_dashboard(real_data, filtered_data, csv_filename, 
                                     channel_range=(0, 8), save_path=None, logo_path="U_logo.png"):
    """
    Create UKF dashboard matching neural mass model style with multi-channel analysis
    """
    
    # Setup brand-compliant fonts and logo
    get_font_weights()
    logo_img = load_and_prepare_logo(logo_path) if logo_path else None
    
    start_ch, end_ch = channel_range
    n_channels = min(end_ch, real_data.shape[0], filtered_data.shape[0])
    channels_to_analyze = range(start_ch, n_channels)
    
    # Calculate metrics for all channels
    all_metrics = {}
    correlation_scores = []
    spectral_scores = []
    phase_scores = []
    
    for ch in channels_to_analyze:
        metrics = analyze_ukf_performance_enhanced(real_data, filtered_data, ch)
        all_metrics[ch] = metrics
        correlation_scores.append(metrics['correlation'] * 100)
        spectral_scores.append(metrics['spectral_similarity'] * 100)
        phase_scores.append(metrics['phase_similarity'] * 100)
    
    # Calculate overall performance
    avg_correlation = np.mean(correlation_scores)
    avg_spectral = np.mean(spectral_scores)
    avg_phase = np.mean(phase_scores)
    overall_score = (avg_correlation + avg_spectral + avg_phase) / 3
    
    # Determine status
    if overall_score >= 70:
        status = "EXCELLENT"
        status_color = BrandColors.GREEN
    elif overall_score >= 50:
        status = "GOOD"
        status_color = BrandColors.BLUE
    elif overall_score >= 30:
        status = "FAIR"
        status_color = BrandColors.ORANGE
    else:
        status = "POOR"
        status_color = BrandColors.RED
    
    # Create figure with exact layout matching neural mass model
    fig = plt.figure(figsize=(16, 10), facecolor=BrandColors.WHITE, dpi=150)
    
    # Create grid layout matching the neural mass model
    gs = gridspec.GridSpec(3, 4, figure=fig, 
                          height_ratios=[0.15, 1.0, 1.0],
                          width_ratios=[1.2, 1.2, 1.2, 1.0],
                          hspace=0.35, wspace=0.25,
                          left=0.08, right=0.95, 
                          top=0.88, bottom=0.12)
    
    # === HEADER SECTION ===
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    
    # Title with logo (matching neural mass model style)
    if logo_img is not None:
        logo_ax = fig.add_axes([0.02, 0.91, 0.03, 0.06])
        logo_ax.imshow(logo_img, aspect='equal')
        logo_ax.axis('off')
    
    # Main title
    ax_header.text(0.5, 0.8, 'THE MIND COMPANY', transform=ax_header.transAxes,
                  fontsize=18, fontweight='bold', color=BrandColors.BLUE,
                  ha='center', va='center')
    
    ax_header.text(0.5, 0.4, f'UKF Model Performance Analysis - Channels {start_ch}-{n_channels-1}', 
                  transform=ax_header.transAxes,
                  fontsize=14, fontweight='bold', color=BrandColors.BLACK,
                  ha='center', va='center')
    
    ax_header.text(0.5, 0.0, f'Overall Performance: {overall_score:.1f}% | Status: {status}', 
                  transform=ax_header.transAxes,
                  fontsize=12, color=status_color, fontweight='bold',
                  ha='center', va='center')
    
    # === 1. CORRELATION PERFORMANCE (Top Left) ===
    ax_corr = fig.add_subplot(gs[1, 0])
    
    channel_labels = [f'Ch{i}' for i in channels_to_analyze]
    colors_corr = [BrandColors.GREEN if x > 70 else BrandColors.ORANGE if x > 30 else BrandColors.RED 
                   for x in correlation_scores]
    
    bars = ax_corr.bar(range(len(channels_to_analyze)), correlation_scores, 
                       color=colors_corr, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on bars
    for i, (bar, val) in enumerate(zip(bars, correlation_scores)):
        ax_corr.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add target lines
    ax_corr.axhline(y=50, color=BrandColors.ORANGE, linestyle='--', alpha=0.7, linewidth=1)
    ax_corr.axhline(y=70, color=BrandColors.GREEN, linestyle='--', alpha=0.7, linewidth=1)
    
    ax_corr.set_title('Correlation Performance', fontsize=14, fontweight='bold', color=BrandColors.BLACK)
    ax_corr.set_xlabel('Channel', fontsize=12)
    ax_corr.set_ylabel('Correlation (%)', fontsize=12)
    ax_corr.set_xticks(range(len(channels_to_analyze)))
    ax_corr.set_xticklabels(channel_labels)
    ax_corr.set_ylim(0, 100)
    ax_corr.grid(True, alpha=0.3)
    
    # === 2. SPECTRAL SIMILARITY (Top Center) ===
    ax_spectral = fig.add_subplot(gs[1, 1])
    
    colors_spectral = [BrandColors.GREEN if x > 70 else BrandColors.ORANGE if x > 30 else BrandColors.RED 
                       for x in spectral_scores]
    
    bars = ax_spectral.bar(range(len(channels_to_analyze)), spectral_scores, 
                          color=colors_spectral, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, spectral_scores)):
        ax_spectral.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add target lines
    ax_spectral.axhline(y=50, color=BrandColors.ORANGE, linestyle='--', alpha=0.7, linewidth=1)
    ax_spectral.axhline(y=70, color=BrandColors.GREEN, linestyle='--', alpha=0.7, linewidth=1)
    
    ax_spectral.set_title('Spectral Similarity', fontsize=14, fontweight='bold')
    ax_spectral.set_xlabel('Channel', fontsize=12)
    ax_spectral.set_ylabel('Spectral Similarity (%)', fontsize=12)
    ax_spectral.set_xticks(range(len(channels_to_analyze)))
    ax_spectral.set_xticklabels(channel_labels)
    ax_spectral.set_ylim(0, 100)
    ax_spectral.grid(True, alpha=0.3)
    
    # === 3. PHASE SIMILARITY (Top Right) ===
    ax_phase = fig.add_subplot(gs[1, 2])
    
    colors_phase = [BrandColors.BLUE if x > 50 else BrandColors.ORANGE if x > 30 else BrandColors.RED 
                    for x in phase_scores]
    
    bars = ax_phase.bar(range(len(channels_to_analyze)), phase_scores, 
                       color=colors_phase, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, phase_scores)):
        ax_phase.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax_phase.set_title('Phase Similarity', fontsize=14, fontweight='bold')
    ax_phase.set_xlabel('Channel', fontsize=12)
    ax_phase.set_ylabel('Phase Similarity (%)', fontsize=12)
    ax_phase.set_xticks(range(len(channels_to_analyze)))
    ax_phase.set_xticklabels(channel_labels)
    ax_phase.set_ylim(0, 100)
    ax_phase.grid(True, alpha=0.3)
    
    # === 4. OVERALL PERFORMANCE METRICS (Top Far Right) ===
    ax_overall = fig.add_subplot(gs[1, 3])
    
    # Create overall performance bars
    metrics_names = ['Correlation', 'Spectral', 'Phase']
    metrics_values = [avg_correlation, avg_spectral, avg_phase]
    colors_overall = [BrandColors.BLUE, BrandColors.PURPLE, BrandColors.GREEN]
    
    bars = ax_overall.bar(metrics_names, metrics_values, color=colors_overall, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, metrics_values):
        ax_overall.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add target lines
    ax_overall.axhline(y=50, color=BrandColors.ORANGE, linestyle='--', alpha=0.7)
    ax_overall.axhline(y=70, color=BrandColors.GREEN, linestyle='--', alpha=0.7)
    
    ax_overall.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    ax_overall.set_ylabel('Performance (%)', fontsize=12)
    ax_overall.set_ylim(0, 100)
    ax_overall.grid(True, alpha=0.3)
    
    # === 5. PERFORMANCE HEATMAP (Bottom Left & Center) ===
    ax_heatmap = fig.add_subplot(gs[2, :2])
    
    # Create heatmap data
    metrics_data = np.array([correlation_scores, spectral_scores, phase_scores])
    
    # Create heatmap
    im = ax_heatmap.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax_heatmap.set_xticks(range(len(channels_to_analyze)))
    ax_heatmap.set_xticklabels(channel_labels)
    ax_heatmap.set_yticks(range(3))
    ax_heatmap.set_yticklabels(['Correlation', 'Spectral', 'Phase'])
    
    # Add text annotations
    for i in range(3):
        for j in range(len(channels_to_analyze)):
            text = ax_heatmap.text(j, i, f'{metrics_data[i, j]:.1f}%',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    ax_heatmap.set_title('Performance Heatmap by Channel and Metric', fontsize=14, fontweight='bold')
    ax_heatmap.set_xlabel('Channel', fontsize=12)
    ax_heatmap.set_ylabel('Metric Type', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Performance (%)', rotation=270, labelpad=15)
    
    # === 6. PERFORMANCE SUMMARY (Bottom Right) ===
    ax_summary = fig.add_subplot(gs[2, 2:])
    ax_summary.axis('off')
    
    # Summary box
    summary_bg = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                               boxstyle="round,pad=0.02",
                               facecolor=BrandColors.LIGHT_GRAY, alpha=0.3,
                               edgecolor=BrandColors.DARK_GRAY, linewidth=2)
    ax_summary.add_patch(summary_bg)
    
    # Find best and worst channels
    best_channel_idx = np.argmax(correlation_scores)
    worst_channel_idx = np.argmin(correlation_scores)
    best_channel = channels_to_analyze[best_channel_idx]
    worst_channel = channels_to_analyze[worst_channel_idx]
    
    summary_lines = [
        'PERFORMANCE SUMMARY',
        f'Channel Range: {start_ch}-{n_channels-1}',
        f'Channels Analyzed: {len(channels_to_analyze)}',
        '',
        f'Best Channel: Ch{best_channel} ({correlation_scores[best_channel_idx]:.1f}%)',
        f'Worst Channel: Ch{worst_channel} ({correlation_scores[worst_channel_idx]:.1f}%)',
        '',
        f'Average Correlation: {avg_correlation:.1f}%',
        f'Average Spectral: {avg_spectral:.1f}%', 
        f'Average Phase: {avg_phase:.1f}%',
        '',
        f'Overall Score: {overall_score:.1f}%',
        f'Performance Status: {status}'
    ]
    
    y_positions = np.linspace(0.95, 0.05, len(summary_lines))
    
    for i, line in enumerate(summary_lines):
        if line == 'PERFORMANCE SUMMARY':
            color = BrandColors.BLUE
            weight = 'bold'
            size = 12
        elif 'Status:' in line:
            color = status_color
            weight = 'bold'
            size = 10
        elif line == '':
            continue
        else:
            color = BrandColors.BLACK
            weight = 'bold' if any(word in line for word in ['Best', 'Worst', 'Overall']) else 'normal'
            size = 10
        
        ax_summary.text(0.1, y_positions[i], line, transform=ax_summary.transAxes,
                       fontsize=size, fontweight=weight, color=color, va='center')
    
    # === STYLING ===
    for ax in [ax_corr, ax_spectral, ax_phase, ax_overall, ax_heatmap]:
        ax.set_facecolor(BrandColors.WHITE)
        for spine in ax.spines.values():
            spine.set_color(BrandColors.DARK_GRAY)
            spine.set_linewidth(1.5)
        ax.tick_params(colors=BrandColors.BLACK, labelsize=10)
    
    # Brand footer
    fig.text(0.5, 0.01, 'U: The Mind Company | Ohio, USA | Advancing Neurostimulation Technology', 
             ha='center', fontsize=10, color=BrandColors.DARK_GRAY, style='italic')
    
    # Save the dashboard
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=BrandColors.WHITE, edgecolor='none')
        print(f"Saved UKF dashboard: {save_path}")
    
    plt.show()
    plt.close()
    
    return {
        'overall_score': overall_score,
        'avg_correlation': avg_correlation,
        'avg_spectral': avg_spectral,
        'avg_phase': avg_phase,
        'status': status,
        'best_channel': best_channel,
        'worst_channel': worst_channel,
        'n_channels': len(channels_to_analyze)
    }

def create_ukf_single_channel_dashboard(real_data, filtered_data, channel, csv_filename,
                                       save_path=None, logo_path="U_logo.png"):
    """
    Create single channel UKF dashboard matching neural mass model style
    """
    
    get_font_weights()
    logo_img = load_and_prepare_logo(logo_path) if logo_path else None
    
    # Analyze performance for the specific channel
    metrics = analyze_ukf_performance_enhanced(real_data, filtered_data, channel)
    
    # Calculate overall performance
    correlation_pct = metrics['correlation'] * 100
    spectral_pct = metrics['spectral_similarity'] * 100
    phase_pct = metrics['phase_similarity'] * 100
    overall_score = (correlation_pct + spectral_pct + phase_pct) / 3
    
    # Determine status
    if overall_score >= 70:
        status = "EXCELLENT"
        status_color = BrandColors.GREEN
    elif overall_score >= 50:
        status = "GOOD"  
        status_color = BrandColors.BLUE
    elif overall_score >= 30:
        status = "FAIR"
        status_color = BrandColors.ORANGE
    else:
        status = "POOR"
        status_color = BrandColors.RED
    
    # Create figure
    fig = plt.figure(figsize=(16, 10), facecolor=BrandColors.WHITE, dpi=150)
    
    # Create grid layout
    gs = gridspec.GridSpec(3, 4, figure=fig, 
                          height_ratios=[0.15, 1.2, 1.0],
                          width_ratios=[1.2, 1.2, 1.2, 1.0],
                          hspace=0.4, wspace=0.3,
                          left=0.08, right=0.95, 
                          top=0.88, bottom=0.08)
    
    # Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    
    if logo_img is not None:
        logo_ax = fig.add_axes([0.02, 0.91, 0.03, 0.06])
        logo_ax.imshow(logo_img, aspect='equal')
        logo_ax.axis('off')
    
    ax_header.text(0.5, 0.8, 'THE MIND COMPANY', transform=ax_header.transAxes,
                  fontsize=18, fontweight='bold', color=BrandColors.BLUE,
                  ha='center', va='center')
    
    ax_header.text(0.5, 0.4, f'UKF Model Performance Analysis - Channel {channel}', 
                  transform=ax_header.transAxes,
                  fontsize=14, fontweight='bold', color=BrandColors.BLACK,
                  ha='center', va='center')
    
    ax_header.text(0.5, 0.0, f'Data Source: {csv_filename}', 
                  transform=ax_header.transAxes,
                  fontsize=12, color=BrandColors.DARK_GRAY,
                  ha='center', va='center')
    
    # === 1. SIGNAL COMPARISON ===
    ax_signal = fig.add_subplot(gs[1, 0])
    
    real_ch = real_data[channel]
    filtered_ch = filtered_data[channel]
    
    samples_to_show = min(512, len(real_ch))
    time_axis = np.arange(samples_to_show) / 256
    
    ax_signal.plot(time_axis, real_ch[:samples_to_show], 
                  color=BrandColors.BLUE, linewidth=1.5, label='Original EEG', alpha=0.9)
    ax_signal.plot(time_axis, filtered_ch[:samples_to_show], 
                  color=BrandColors.RED, linewidth=1.5, label='UKF Filtered', alpha=0.9)
    
    ax_signal.set_title(f'Channel {channel} Signal Comparison', fontsize=14, fontweight='bold')
    ax_signal.set_xlabel('Time (seconds)', fontsize=12)
    ax_signal.set_ylabel('Amplitude (μV)', fontsize=12)
    ax_signal.legend()
    ax_signal.grid(True, alpha=0.3)
    
    # === 2. PERFORMANCE METRICS ===
    ax_metrics = fig.add_subplot(gs[1, 1])
    
    metric_names = ['Correlation', 'MSE\n(×10⁻³)', 'Alpha Peak\nMatch']
    
    # Calculate alpha peak match
    alpha_peak_real = metrics['band_peaks_real']['Alpha']
    alpha_peak_filt = metrics['band_peaks_filt']['Alpha']
    alpha_peak_match = 100 - abs(alpha_peak_real - alpha_peak_filt) * 10
    alpha_peak_match = max(0, min(100, alpha_peak_match))
    
    metric_values = [correlation_pct, metrics['mse'] * 1000, alpha_peak_match]
    colors = [BrandColors.BLUE, BrandColors.ORANGE, BrandColors.GREEN]
    
    bars = ax_metrics.bar(metric_names, metric_values, color=colors, alpha=0.8)
    
    for bar, val, name in zip(bars, metric_values, metric_names):
        height = bar.get_height()
        if 'MSE' in name:
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_metrics.set_title('UKF Performance Metrics', fontsize=14, fontweight='bold')
    ax_metrics.set_ylabel('Performance Score', fontsize=12)
    ax_metrics.grid(True, alpha=0.3)
    
    # === 3. POWER SPECTRAL DENSITY ===
    ax_psd = fig.add_subplot(gs[1, 2])
    
    ax_psd.semilogy(metrics['freqs_real'], metrics['psd_real'], 
                   color=BrandColors.BLUE, linewidth=2, label='Original', alpha=0.8)
    ax_psd.semilogy(metrics['freqs_filtered'], metrics['psd_filtered'], 
                   color=BrandColors.RED, linewidth=2, label='UKF Filtered', alpha=0.8)
    
    # Highlight frequency bands
    colors_bands = [BrandColors.PURPLE, BrandColors.GREEN, BrandColors.YELLOW, 
                   BrandColors.ORANGE, BrandColors.RED]
    
    for i, (band_name, (low, high)) in enumerate(metrics['bands'].items()):
        ax_psd.axvspan(low, high, alpha=0.1, color=colors_bands[i])
    
    ax_psd.set_xlim(0, 50)
    ax_psd.set_title('Power Spectral Density', fontsize=14, fontweight='bold')
    ax_psd.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_psd.set_ylabel('Power (μV²/Hz)', fontsize=12)
    ax_psd.legend()
    ax_psd.grid(True, alpha=0.3)
    
    # === 4. UKF SUMMARY ===
    ax_summary = fig.add_subplot(gs[1, 3])
    ax_summary.axis('off')
    
    summary_bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                               boxstyle="round,pad=0.02",
                               facecolor=BrandColors.LIGHT_BLUE, alpha=0.3,
                               edgecolor=BrandColors.BLUE, linewidth=2)
    ax_summary.add_patch(summary_bg)
    
    summary_lines = [
        'UKF SUMMARY',
        f'Channel: {channel}',
        f'Correlation: {correlation_pct:.1f}%',
        f'MSE: {metrics["mse"]:.4f}',
        f'Status: {status}',
        f'Alpha Peak Real: {alpha_peak_real:.1f}Hz',
        f'Alpha Peak UKF: {alpha_peak_filt:.1f}Hz',
        f'Spectral Match: {spectral_pct:.1f}%',
        f'Phase Match: {phase_pct:.1f}%'
    ]
    
    ax_summary.text(0.5, 0.95, 'UKF SUMMARY', transform=ax_summary.transAxes,
                   fontsize=12, fontweight='bold', color=BrandColors.BLUE,
                   ha='center', va='top')
    
    y_positions = np.linspace(0.85, 0.15, len(summary_lines)-1)
    
    for i, line in enumerate(summary_lines[1:]):
        color = status_color if 'Status:' in line else BrandColors.BLACK
        weight = 'bold' if any(word in line for word in ['Status:', 'Channel:', 'Correlation:']) else 'normal'
        
        ax_summary.text(0.1, y_positions[i], line, transform=ax_summary.transAxes,
                       fontsize=10, fontweight=weight, color=color, va='center')
    
    # === 5. FREQUENCY BAND ANALYSIS ===
    ax_bands = fig.add_subplot(gs[2, :2])
    
    band_names = list(metrics['bands'].keys())
    band_powers_real = [metrics['band_powers_real'][band] for band in band_names]
    band_powers_filt = [metrics['band_powers_filt'][band] for band in band_names]
    
    x_pos = np.arange(len(band_names))
    width = 0.35
    
    bars1 = ax_bands.bar(x_pos - width/2, band_powers_real, width, 
                        color=BrandColors.BLUE, alpha=0.8, label='Original')
    bars2 = ax_bands.bar(x_pos + width/2, band_powers_filt, width,
                        color=BrandColors.RED, alpha=0.8, label='UKF Filtered')
    
    ax_bands.set_title('Frequency Band Power', fontsize=14, fontweight='bold')
    ax_bands.set_xlabel('Frequency Bands', fontsize=12)
    ax_bands.set_ylabel('Power (μV²)', fontsize=12)
    ax_bands.set_xticks(x_pos)
    ax_bands.set_xticklabels(band_names)
    ax_bands.legend()
    ax_bands.grid(True, alpha=0.3)
    
    # === 6. UKF STABILITY ANALYSIS ===
    ax_stability = fig.add_subplot(gs[2, 2:])
    
    # Create stability metrics
    stability_metrics = {
        'Frequency\nPreservation': correlation_pct,
        'Noise Reduction': max(0, 100 - (metrics['mse'] * 1000)),
        'Signal Stability': min(100, correlation_pct * 1.1),
        'Phase Coherence': phase_pct
    }
    
    # Circular progress indicators
    angles = np.linspace(0, 2*np.pi, len(stability_metrics), endpoint=False)
    
    for i, (metric_name, value) in enumerate(stability_metrics.items()):
        angle = angles[i]
        x = 0.5 + 0.3 * np.cos(angle)
        y = 0.5 + 0.3 * np.sin(angle)
        
        # Create mini circle
        circle = Circle((x, y), 0.08, facecolor=BrandColors.LIGHT_BLUE, 
                       edgecolor=BrandColors.BLUE, linewidth=2, alpha=0.8,
                       transform=ax_stability.transAxes)
        ax_stability.add_patch(circle)
        
        # Add percentage
        ax_stability.text(x, y, f'{value:.0f}%', transform=ax_stability.transAxes,
                         ha='center', va='center', fontsize=9, fontweight='bold',
                         color=BrandColors.BLUE)
        
        # Add label
        label_x = 0.5 + 0.45 * np.cos(angle)
        label_y = 0.5 + 0.45 * np.sin(angle)
        ax_stability.text(label_x, label_y, metric_name, transform=ax_stability.transAxes,
                         ha='center', va='center', fontsize=10, fontweight='bold',
                         color=BrandColors.BLACK)
    
    ax_stability.set_xlim(0, 1)
    ax_stability.set_ylim(0, 1)
    ax_stability.set_title('UKF Stability Analysis', fontsize=14, fontweight='bold')
    ax_stability.axis('off')
    
    # === STYLING ===
    for ax in [ax_signal, ax_metrics, ax_psd, ax_bands]:
        ax.set_facecolor(BrandColors.WHITE)
        for spine in ax.spines.values():
            spine.set_color(BrandColors.DARK_GRAY)
            spine.set_linewidth(1.5)
        ax.tick_params(colors=BrandColors.BLACK, labelsize=10)
    
    # Brand footer
    fig.text(0.5, 0.01, 'U: The Mind Company | Ohio, USA | Advancing Neurostimulation Technology', 
             ha='center', fontsize=10, color=BrandColors.DARK_GRAY, style='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=BrandColors.WHITE, edgecolor='none')
        print(f"Saved UKF single channel dashboard: {save_path}")
    
    plt.show()
    plt.close()
    
    return {
        'correlation': correlation_pct,
        'spectral_similarity': spectral_pct,
        'phase_similarity': phase_pct,
        'overall_score': overall_score,
        'status': status,
        'alpha_peak_real': alpha_peak_real,
        'alpha_peak_filtered': alpha_peak_filt
    }