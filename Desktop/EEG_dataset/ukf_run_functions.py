# UKF Run Functions - U: The Mind Company
# Interactive analysis functions for UKF visualizations

import os
from ukf_data_processing import (
    get_csv_input, get_channel_input, get_band_input,
    apply_band_filtering, apply_ukf_filtering
)
from ukf_similarity_circle_visualizations import create_ukf_similarity_circle
from ukf_channel_comparison_visualizations import create_ukf_channel_comparison
from ukf_dashboard_visualizations import (
    create_ukf_multi_channel_dashboard, create_ukf_single_channel_dashboard
)
from digital_twin_using_neuralmass_model_3 import get_channel_range_input

def run_ukf_similarity_circle_analysis():
    """Interactive UKF similarity circle analysis"""
    print("\n" + "="*80)
    print("U: THE MIND COMPANY - UKF SIMILARITY CIRCLE ANALYSIS")
    print("Interactive Neural Signal Processing Analysis")
    print("="*80)
    
    # Get CSV input
    csv_path, real_data, columns = get_csv_input()
    if real_data is None:
        return
    
    # Get channel input
    channel = get_channel_input(real_data.shape[0])
    if channel is None:
        return
    
    # Get frequency band input
    band_name, freq_range = get_band_input()
    if band_name is None:
        return
    
    print(f"\nProcessing UKF similarity analysis...")
    print(f"    CSV File: {csv_path}")
    print(f"    Channel: {channel}")
    print(f"    Band: {band_name}")
    
    # Extract filename from path
    csv_filename = os.path.basename(csv_path)
    
    # Process single band
    real_band_data = apply_band_filtering(real_data, band_name, freq_range)
    filtered_data = apply_ukf_filtering(real_data, band_name, freq_range)
    
    # Create clean filename
    clean_band_name = band_name.replace(' ', '_').replace('(', '').replace(')', '').replace('â€"', '-')
    data_source = csv_filename.replace('.csv', '')
    save_path = f'U_UKF_similarity_{clean_band_name}_ch{channel}_{data_source}_circle.png'
    
    # Generate single similarity circle
    metrics = create_ukf_similarity_circle(
        real_band_data, filtered_data, channel, csv_filename, 
        band_name, freq_range, save_path, logo_path="U_logo.png"
    )
    
    print(f"\nUKF SIMILARITY CIRCLE ANALYSIS COMPLETE!")
    print(f"Channel: {channel}")
    print(f"Band: {band_name}")
    print(f"Similarity: {metrics['similarity_percentage']:.1f}%")
    print(f"Quality: {metrics['quality']}")
    print(f"Generated: {save_path}")
    
    return metrics

def run_ukf_single_channel_dashboard():
    """Interactive single channel UKF dashboard analysis"""
    print("\n" + "="*80)
    print("U: THE MIND COMPANY - UKF SINGLE CHANNEL DASHBOARD")
    print("Interactive Neural Signal Processing Analysis")
    print("="*80)
    
    # Get CSV input
    csv_path, real_data, columns = get_csv_input()
    if real_data is None:
        return
    
    # Get channel input
    channel = get_channel_input(real_data.shape[0])
    if channel is None:
        return
    
    print(f"\nProcessing UKF single channel dashboard for channel {channel}...")
    
    # Process full spectrum data (no band filtering for comprehensive single channel analysis)
    filtered_data = apply_ukf_filtering(real_data)
    
    # Create filename
    csv_name = os.path.basename(csv_path).replace('.csv', '')
    save_path = f'U_UKF_single_channel_{channel}_{csv_name}_dashboard.png'
    
    # Generate single channel dashboard
    results = create_ukf_single_channel_dashboard(
        real_data, filtered_data, channel, csv_name, save_path, logo_path="U_logo.png"
    )
    
    print(f"\nUKF SINGLE CHANNEL DASHBOARD COMPLETE!")
    print(f"Dataset: {csv_name}")
    print(f"Channel: {channel}")
    print(f"Correlation: {results['correlation']:.1f}%")
    print(f"Spectral Similarity: {results['spectral_similarity']:.1f}%")
    print(f"Phase Similarity: {results['phase_similarity']:.1f}%")
    print(f"Overall Score: {results['overall_score']:.1f}%")
    print(f"Status: {results['status']}")
    print(f"Alpha Peak Real: {results['alpha_peak_real']:.1f}Hz")
    print(f"Alpha Peak Filtered: {results['alpha_peak_filtered']:.1f}Hz")
    print(f"Generated: {save_path}")
    
    return results

def run_ukf_multi_channel_dashboard():
    """Interactive multi-channel UKF dashboard analysis"""
    print("\n" + "="*80)
    print("U: THE MIND COMPANY - UKF MULTI-CHANNEL DASHBOARD")
    print("Interactive Neural Signal Processing Analysis")
    print("="*80)
    
    # Get CSV input
    csv_path, real_data, columns = get_csv_input()
    if real_data is None:
        return
    
    # Get channel range input
    channel_range = get_channel_range_input(real_data.shape[0])
    if channel_range is None:
        return
    
    print(f"\nProcessing UKF multi-channel dashboard for channels {channel_range[0]}-{channel_range[1]}...")
    
    # Process full spectrum data (no band filtering for multi-channel dashboard)
    band_name, freq_range = get_band_input()
    filtered_data = apply_ukf_filtering(real_data, band_name, freq_range)
    
    # Create filename
    csv_name = csv_path
    save_path = f'U_UKF_multi_channel_{channel_range[0]}-{channel_range[1]}_{csv_name}_dashboard.png'
    
    # Generate multi-channel dashboard
    results = create_ukf_multi_channel_dashboard(
        real_data, filtered_data, csv_name, channel_range, save_path, logo_path="U_logo.png"
    )
    
    print(f"\nUKF MULTI-CHANNEL DASHBOARD COMPLETE!")
    print(f"Dataset: {csv_name}")
    print(f"Channels: {channel_range[0]}-{channel_range[1]}")
    print(f"Channels Analyzed: {results['n_channels']}")
    print(f"Average Correlation: {results['avg_correlation']:.1f}%")
    print(f"Average Spectral: {results['avg_spectral']:.1f}%")
    print(f"Average Phase: {results['avg_phase']:.1f}%")
    print(f"Overall Score: {results['overall_score']:.1f}%")
    print(f"Best Channel: Ch{results['best_channel']}")
    print(f"Worst Channel: Ch{results['worst_channel']}")
    print(f"Status: {results['status']}")
    print(f"Generated: {save_path}")
    
    return results

def run_ukf_multi_channel_comparison():
    """Interactive multi-channel UKF comparison analysis"""
    print("\n" + "="*80)
    print("U: THE MIND COMPANY - UKF MULTI-CHANNEL COMPARISON")
    print("Interactive Neural Signal Processing Analysis")
    print("="*80)
    
    # Get CSV input
    csv_path, real_data, columns = get_csv_input()
    if real_data is None:
        return
    
    # Get channel range input
    channel_range = get_channel_range_input(real_data.shape[0])
    if channel_range is None:
        return
    
    # Get frequency band input
    band_name, freq_range = get_band_input()
    if band_name is None:
        return
    
    print(f"\nProcessing UKF comparison for channels {channel_range[0]}-{channel_range[1]}, band: {band_name}...")
    
    # Process single band
    real_band_data = apply_band_filtering(real_data, band_name, freq_range)
    filtered_data = apply_ukf_filtering(real_data, band_name, freq_range)
    
    # Create filename
    csv_name = os.path.basename(csv_path)
    clean_band_name = band_name.replace(' ', '_').replace('(', '').replace(')', '').replace('â€"', '-')
    save_path = f'U_UKF_channels_{channel_range[0]}-{channel_range[1]}_{clean_band_name}_{csv_name.replace(".csv", "")}_comparison.png'
    
    # Generate comparison
    avg_correlation = create_ukf_channel_comparison(
        real_band_data, filtered_data, channel_range, csv_name, save_path, logo_path="U_logo.png", band_name=band_name
    )
    
    print(f"\nUKF MULTI-CHANNEL COMPARISON COMPLETE!")
    print(f"Channels: {channel_range[0]}-{channel_range[1]}")
    print(f"Band: {band_name}")
    print(f"Average Correlation: {avg_correlation:.3f}")
    print(f"Generated: {save_path}")
    
    return avg_correlation

# def run_ukf_comprehensive_analysis():
#     """Interactive comprehensive UKF analysis"""
#     print("\n" + "="*80)
#     print("U: THE MIND COMPANY - UKF COMPREHENSIVE ANALYSIS")
#     print("Interactive Neural Signal Processing Analysis")
#     print("="*80)
    
#     # Get CSV input
#     csv_path, real_data, columns = get_csv_input()
#     if real_data is None:
#         return
    
#     # Get frequency band input
#     band_name, freq_range = get_band_input()
#     if band_name is None:
#         return
    
#     print(f"\nProcessing comprehensive UKF analysis for band: {band_name}...")
    
#     # Process single band
#     real_band_data = apply_band_filtering(real_data, band_name, freq_range)
#     filtered_data = apply_ukf_filtering(real_data, band_name, freq_range)
    
#     # Create filename
#     csv_name = os.path.basename(csv_path).replace('.csv', '')
#     clean_band_name = band_name.replace(' ', '_').replace('(', '').replace(')', '').replace('â€"', '-')
#     save_path = f'U_UKF_comprehensive_{clean_band_name}_{csv_name}_dashboard.png'
    
#     # Generate dashboard
#     results = create_ukf_comprehensive_dashboard(
#         real_band_data, filtered_data, csv_name, save_path, logo_path="U_logo.png"
#     )
    
#     print(f"\nUKF COMPREHENSIVE ANALYSIS COMPLETE!")
#     print(f"Dataset: {csv_name}")
#     print(f"Band: {band_name}")
#     print(f"Channels Analyzed: {results['n_channels']}")
#     print(f"Average Correlation: {results['avg_correlation']:.3f}")
#     print(f"Average MSE: {results['avg_mse']:.4f}")
#     print(f"Best Channel: {results['best_channel']}")
#     print(f"Status: {results['status']}")
#     print(f"Generated: {save_path}")
    
#     return results

def main_ukf_visualization_suite():
    """Main function to run UKF visualization suite"""
    
    print("\n" + "="*80)
    print("U: THE MIND COMPANY - UKF VISUALIZATION SUITE")
    print("Advanced Neural Signal Processing Analysis")
    print("Ohio, USA")
    print("="*80)
    
    while True:
        try:
            print("\nUKF ANALYSIS OPTIONS:")
            print("1. Similarity Circle Analysis")
            print("2. Single Channel Dashboard")
            print("3. Multi-Channel Dashboard")
            print("4. Multi-Channel Comparison")
            print("5. Comprehensive Analysis")
            print("6. Exit")
            print("\nFREQUENCY BANDS SUPPORTED:")
            print("   • Delta (0.5–4 Hz)")
            print("   • Theta (4–8 Hz)")
            print("   • Alpha (8–13 Hz)")
            print("   • Beta (13–30 Hz)")
            print("   • Gamma (30–45 Hz)")
            print("   • Custom Band (user-defined range)")
            
            choice = input("\nSelect analysis type (1-6): ").strip()
            
            if choice == '1':
                run_ukf_similarity_circle_analysis()
            elif choice == '2':
                run_ukf_single_channel_dashboard()
            elif choice == '3':
                run_ukf_multi_channel_dashboard()
            elif choice == '4':
                run_ukf_multi_channel_comparison()
            elif choice == '5':
                print("\nExiting UKF Visualization Suite")
                print("U: The Mind Company | Advancing Neurostimulation Technology")
                break
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nExiting UKF Visualization Suite")
            print("U: The Mind Company | Advancing Neurostimulation Technology")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main_ukf_visualization_suite()