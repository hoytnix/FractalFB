"""FractalFB - Facebook Ad Analysis and Reporting Tool

This module provides functionality for analyzing Facebook ad performance data
and generating comprehensive PDF reports with visualizations and recommendations.

The tool processes CSV data from Facebook Ads Manager and generates insights based on
user-defined KPI ranges specified in a YAML configuration file.


Example settings.yaml:

```
organization: "Michael Hoyt"
kpi_ranges:
  "Quality ranking":
    min: 7
    max: 10
  "CTR (link click-through rate)":
    min: 0.5
    max: 3
  "Cost per results":
    min: 0.01
    max: 30
```

"""

import pandas as pd
import yaml
from fpdf import FPDF
from PIL import Image, ImageDraw
from datetime import datetime
from sphinx.cmd.quickstart import generate
from sphinx.ext.autodoc import setup
import argparse
import os

def read_facebook_data(csv_file: str) -> list:
    """Read and parse Facebook ad data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing Facebook ad data

    Returns:
        list: List of dictionaries containing ad data, or None if file read fails
    """
    try:
        df = pd.read_csv(csv_file)
        return df.to_dict('records')
    except Exception as e:
        print(f'Error reading CSV file: {e}')
        return None

def read_kpi_settings(yaml_file: str) -> dict:
    """Read KPI settings from a YAML configuration file.

    Args:
        yaml_file (str): Path to the YAML file containing KPI settings

    Returns:
        dict: Dictionary containing KPI settings, or None if file read fails
    """
    try:
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f'Error reading YAML file: {e}')
        return None

def get_gradient_color(value: float, min_val: float, max_val: float) -> tuple:
    """Calculate gradient color based on value position between min and max.

    Args:
        value: The value to calculate color for.
        min_val: Minimum value in range.
        max_val: Maximum value in range.

    Returns:
        Tuple of RGB color values as (red, green, blue).
    """
    if max_val == min_val:
        norm_value = 0.5
    else:
        norm_value = (value - min_val) / (max_val - min_val)
        # Clamp norm_value between 0 and 1
        norm_value = max(0, min(1, norm_value))
    
    red = (255, 0, 0)
    yellow = (255, 255, 0)
    green = (0, 255, 0)
    
    if norm_value <= 0.5:
        t = norm_value * 2
        r = int(red[0] + (yellow[0] - red[0]) * t)
        g = int(red[1] + (yellow[1] - red[1]) * t)
        b = int(red[2] + (yellow[2] - red[2]) * t)
    else:
        t = (norm_value - 0.5) * 2
        r = int(yellow[0] + (green[0] - yellow[0]) * t)
        g = int(yellow[1] + (green[1] - yellow[1]) * t)
        b = int(yellow[2] + (green[2] - yellow[2]) * t)
    
    return (r, g, b)

def create_funnel_image(metrics: dict, kpi_results: list, width: int = 800, height: int = 600) -> Image:
	# Validate and clean input metrics using safe conversion
	validated_metrics = {}
	for key, value in metrics.items():
		if pd.isna(value):
			validated_metrics[key] = 0
		elif key in ['Link Clicks (#)', 'Leads (#)']:
			validated_metrics[key] = safe_convert_int(value)
		else:
			validated_metrics[key] = safe_convert_float(value)

	# Create image
	image = Image.new('RGB', (width, height), 'white')
	draw = ImageDraw.Draw(image)

	# Calculate funnel dimensions
	top_width = 600
	bottom_width = 200
	section_height = height // len(validated_metrics)

	# Draw funnel sections
	for i, (metric_name, value) in enumerate(validated_metrics.items()):
		y1 = i * section_height
		y2 = (i + 1) * section_height
		current_width = top_width - (i * ((top_width - bottom_width) / (len(validated_metrics) - 1)))
		x1 = int((width - current_width) // 2)
		x2 = int(x1 + current_width)

		# Map metric names
		kpi_mapping = {
			'CPM ($)': 'CPM (cost per 1,000 impressions) (USD)',
			'CTR (%)': 'CTR (link click-through rate)',
			'CPC ($)': 'CPC (cost per link click) (USD)',
			'Link Clicks (#)': 'Link clicks',
			'Results (%)': 'Result rate',
			'Leads (#)': 'Leads'
		}

		kpi_name = kpi_mapping.get(metric_name)
		fill_color = 'lightblue'

		# Calculate gradient color
		if metric_name == 'Link Clicks (#)':
			clicks_values = [result['Link clicks']['value'] for result in kpi_results if 'Link clicks' in result]
			min_clicks = min(clicks_values) if clicks_values else 0
			max_clicks = max(clicks_values) if clicks_values else 1
			fill_color = get_gradient_color(value, min_clicks, max_clicks)
		elif metric_name == 'Leads (#)':
			leads_values = [result['Leads']['value'] for result in kpi_results if 'Leads' in result]
			min_leads = min(leads_values) if leads_values else 0
			max_leads = max(leads_values) if leads_values else 1
			fill_color = get_gradient_color(value, min_leads, max_leads)
		elif kpi_name in ['CPM (cost per 1,000 impressions) (USD)', 'CTR (link click-through rate)', 'CPC (cost per link click) (USD)', 'Result rate']:
			for result in kpi_results:
				if kpi_name in result:
					kpi_data = result[kpi_name]
					if isinstance(kpi_data, dict) and 'value' in kpi_data:
						min_val = float(kpi_data.get('min', 0))
						max_val = float(kpi_data.get('max', 1))
						fill_color = get_gradient_color(value, min_val, max_val)
						break

		# Draw funnel section
		draw.polygon([
			(x1, y1),
			(x2, y1),
			(int(x2 - ((top_width - bottom_width) / (len(validated_metrics) - 1)) // 2), y2),
			(int(x1 + ((top_width - bottom_width) / (len(validated_metrics) - 1)) // 2), y2)
		], outline='black', fill=fill_color)

		# Format text
		if metric_name in ['Leads (#)', 'Link Clicks (#)']:
			text = f'{metric_name}: {int(value)}'
			text_x = int((width - draw.textlength(text)) // 2)
		else:
			text = f'{metric_name}: {value:.3f}'
			text_x = int((width - draw.textlength(text)) // 2)

		# Get text dimensions for vertical positioning
		text_bbox = draw.textbbox((0, 0), text)
		text_height = text_bbox[3] - text_bbox[1]
		text_y = int(y1 + (section_height - text_height) // 2)

		# Draw text
		draw.text(
			(text_x, text_y),
			text,
			fill='black'
		)

	return image

def check_kpi_ranges(data: list, settings: dict) -> list:
    results = []
    for entry in data:
        entry_results = {
            'Ad name': entry.get('Ad name', 'Unnamed Ad'),
            'Ad Set Name': entry.get('Ad Set Name', 'Unnamed Ad Set'),
            'Ad set budget': entry.get('Ad set budget', 'N/A'),
            'Ad set budget type': entry.get('Ad set budget type', 'N/A')
        }
        for metric, value in entry.items():
            if metric in settings['kpi_ranges']:
                value = safe_convert_float(value)
                min_val = settings['kpi_ranges'][metric]['min']
                max_val = settings['kpi_ranges'][metric]['max']
                status_explanation = ''
                
                if value < min_val:
                    status_explanation = f'Value {value:.2f} is below minimum threshold of {min_val:.2f}'
                elif value > max_val:
                    status_explanation = f'Value {value:.2f} exceeds maximum threshold of {max_val:.2f}'
                else:
                    status_explanation = f'Value {value:.2f} is within expected range ({min_val:.2f} - {max_val:.2f})'
                
                entry_results[metric] = {
                    'value': value,
                    'within_range': min_val <= value <= max_val,
                    'explanation': status_explanation
                }
        results.append(entry_results)
    return results

def analyze_metrics(entry: dict, data_entry: dict) -> dict:
    """Analyze ad metrics and generate insights.

    Args:
        entry: Dictionary containing ad entry data.
        data_entry: Dictionary containing metric data for analysis.

    Returns:
        Dictionary containing analysis results with strengths, weaknesses, and recommendations.
    """
    # Create analysis dictionary
    analysis = {
        'strengths': [],
        'weaknesses': [],
        'recommendations': []
    }
    
    # Analyze CPM
    cpm = float(data_entry.get('CPM (cost per 1,000 impressions) (USD)', 0))
    if cpm < 10:
        analysis['strengths'].append('Low CPM indicates cost-effective reach')
        analysis['strengths'].append('Efficient budget utilization for audience exposure')
    elif cpm > 20:
        analysis['weaknesses'].append('High CPM suggests targeting may need refinement')
        analysis['weaknesses'].append('Budget inefficiency in reaching target audience')
        analysis['recommendations'].append('Consider narrowing audience targeting to reduce CPM')
        analysis['recommendations'].append('Test different placements to find more cost-effective options')
        analysis['recommendations'].append('Review and optimize ad scheduling for better CPM')
    
    # Analyze CTR
    ctr = float(data_entry.get('CTR (link click-through rate)', 0))
    if ctr > 1:
        analysis['strengths'].append('Strong CTR shows engaging ad content')
        analysis['strengths'].append('Effective message-to-audience match')
        analysis['strengths'].append('Creative elements resonating well with viewers')
    elif ctr < 0.5:
        analysis['weaknesses'].append('Low CTR indicates ad content may not resonate with audience')
        analysis['weaknesses'].append('Possible mismatch between ad creative and target audience')
        analysis['weaknesses'].append('Weak call-to-action performance')
        analysis['recommendations'].append('Test different ad creatives and copy to improve engagement')
        analysis['recommendations'].append('Analyze successful competitor ads for insights')
        analysis['recommendations'].append('Implement A/B testing for headlines and images')
        analysis['recommendations'].append('Strengthen call-to-action elements')
    
    # Analyze CPC
    cpc = float(data_entry.get('CPC (cost per link click) (USD)', 0))
    if cpc < 0.5:
        analysis['strengths'].append('Efficient cost per click')
        analysis['strengths'].append('Strong ad relevance score likely')
        analysis['strengths'].append('Competitive advantage in bidding')
    elif cpc > 2:
        analysis['weaknesses'].append('High CPC may indicate targeting or bidding issues')
        analysis['weaknesses'].append('Poor ad relevance score possible')
        analysis['weaknesses'].append('Competitive disadvantage in auction')
        analysis['recommendations'].append('Review bidding strategy and audience targeting')
        analysis['recommendations'].append('Optimize ad relevance through better targeting')
        analysis['recommendations'].append('Consider testing different ad formats')
        analysis['recommendations'].append('Analyze competitor bidding strategies')
    
    # Analyze Result Rate
    result_rate = float(data_entry.get('Result rate', 0))
    if result_rate > 10:
        analysis['strengths'].append('Excellent conversion rate')
        analysis['strengths'].append('Strong alignment between ad and landing page')
        analysis['strengths'].append('Effective qualification of traffic')
        analysis['strengths'].append('High-intent audience targeting')
    elif result_rate < 5:
        analysis['weaknesses'].append('Low conversion rate suggests landing page or offer improvements needed')
        analysis['weaknesses'].append('Possible disconnect between ad promise and landing page delivery')
        analysis['weaknesses'].append('Traffic quality issues')
        analysis['recommendations'].append('Optimize landing page and test different offers')
        analysis['recommendations'].append('Improve page load speed and mobile responsiveness')
        analysis['recommendations'].append('Implement trust signals and social proof')
        analysis['recommendations'].append('Streamline conversion process')
        analysis['recommendations'].append('Test different value propositions')
    
    return analysis

def add_analysis_page(pdf, entry, data_entry):
    """Add an analysis page to the PDF report with strengths, weaknesses, and recommendations.

    Args:
        pdf (FPDF): The PDF document object to add the page to
        entry (dict): Dictionary containing ad entry data
        data_entry (dict): Dictionary containing metric data for analysis
    """
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 10, 'Ad Performance Analysis', 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, "{} - {}".format(entry['Ad name'], entry['Ad Set Name']), 0, 1, 'C')
    pdf.ln(10)
    
    analysis = analyze_metrics(entry, data_entry)
    
    # Add Strengths section
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 10, 'Strengths:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    for strength in analysis['strengths']:
        pdf.cell(0, 8, '- ' + strength, 0, 1)  # Changed bullet point to hyphen
    pdf.ln(5)
    
    # Add Weaknesses section
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 10, 'Areas for Improvement:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    for weakness in analysis['weaknesses']:
        pdf.cell(0, 8, '- ' + weakness, 0, 1)  # Changed bullet point to hyphen
    pdf.ln(5)
    
    # Add Recommendations section
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 255)
    pdf.cell(0, 10, 'Recommendations:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    for recommendation in analysis['recommendations']:
        pdf.cell(0, 8, '- ' + recommendation, 0, 1)  # Changed bullet point to hyphen

def generate_ranking_page(pdf, results, data):
    """Generate a page showing rankings of all ads based on performance.

    Args:
        pdf (FPDF): The PDF document object to add the rankings to
        results (list): List of analysis results for each ad
        data (list): List of raw ad performance data
    """
    # Create a list of ads with their performance metrics
    ad_rankings = []
    for i, entry in enumerate(results):
        ad_metrics = {
            'Ad name': entry['Ad name'],
            'Ad Set Name': entry['Ad Set Name'],
            'CTR': float(data[i].get('CTR (link click-through rate)', 0)),
            'CPC': float(data[i].get('CPC (cost per link click) (USD)', 0)),
            'Leads': float(data[i].get('Leads', 0)) if pd.notna(data[i].get('Leads')) else 0,
            'Result rate': float(data[i].get('Result rate', 0))
        }
        
        # Calculate a composite score (higher is better)
        score = (ad_metrics['CTR'] * 0.3) + \
                (1 / ad_metrics['CPC'] if ad_metrics['CPC'] > 0 else 0) * 0.3 + \
                (ad_metrics['Leads'] * 0.2) + \
                (ad_metrics['Result rate'] * 0.2)
        
        ad_metrics['Score'] = score
        ad_rankings.append(ad_metrics)
    
    # Sort ads by score in descending order
    ad_rankings.sort(key=lambda x: x['Score'], reverse=True)
    
    # Add ranking page to PDF
    pdf.add_page('L')
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 10, 'Ad Performance Rankings', 0, 1, 'C')
    pdf.ln(10)
    
    # Add table headers
    pdf.set_font('Arial', 'B', 12)
    column_widths = [15, 60, 60, 25, 25, 25, 25, 30]
    headers = ['Rank', 'Ad Name', 'Ad Set', 'CTR (%)', 'CPC ($)', 'Leads', 'Result %', 'Score']
    
    for width, header in zip(column_widths, headers):
        pdf.cell(width, 10, header, 1, 0, 'C')
    pdf.ln()
    
    # Add table rows
    pdf.set_font('Arial', '', 10)
    for rank, ad in enumerate(ad_rankings, 1):
        pdf.cell(column_widths[0], 10, str(rank), 1, 0, 'C')
        pdf.cell(column_widths[1], 10, ad['Ad name'][:30], 1, 0, 'L')
        pdf.cell(column_widths[2], 10, ad['Ad Set Name'][:30], 1, 0, 'L')
        pdf.cell(column_widths[3], 10, f"{ad['CTR']:.2f}", 1, 0, 'C')
        pdf.cell(column_widths[4], 10, f"{ad['CPC']:.2f}", 1, 0, 'C')
        pdf.cell(column_widths[5], 10, f"{int(ad['Leads'])}", 1, 0, 'C')
        pdf.cell(column_widths[6], 10, f"{ad['Result rate']:.2f}", 1, 0, 'C')
        pdf.cell(column_widths[7], 10, f"{ad['Score']:.2f}", 1, 1, 'C')
    
    # Add explanation of scoring
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Scoring Methodology:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, 'The overall score is calculated using the following weights:\n- CTR (Click-Through Rate): 30%\n- CPC (Cost Per Click) Efficiency: 30%\n- Number of Leads: 20%\n- Result Rate: 20%\n\nHigher scores indicate better overall performance.', 0, 'L')

def create_bar_graph(metrics, width=800, height=400):
	image = Image.new('RGB', (width, height), 'white')
	draw = ImageDraw.Draw(image)
	
	# Define graph area margins
	margin_left = 100
	margin_right = 50
	margin_top = 50
	margin_bottom = 100
	
	# Calculate bar width and spacing
	num_bars = len(metrics)
	bar_width = (width - margin_left - margin_right) / (num_bars * 2)
	
	# Find max value for scaling (with NaN handling)
	valid_values = [v for v in metrics.values() if not pd.isna(v) and v is not None]
	
	# Handle case where there are no valid values
	if not valid_values:
		return image
	
	max_value = max(valid_values)
	
	# Draw axes
	draw.line([(margin_left, height - margin_bottom),
			  (width - margin_right, height - margin_bottom)], fill='black', width=2)
	draw.line([(margin_left, height - margin_bottom),
			  (margin_left, margin_top)], fill='black', width=2)
	
	# Draw bars
	for i, (metric_name, value) in enumerate(metrics.items()):
		# Skip NaN or None values
		if pd.isna(value) or value is None:
			continue
			
		# Calculate bar position and height
		x = margin_left + (i * 2 + 1) * bar_width
		bar_height = ((height - margin_top - margin_bottom) * value) / max_value if max_value > 0 else 0
		y = height - margin_bottom - bar_height
		
		# Draw bar with corrected coordinates
		y_top = min(y, height - margin_bottom)
		y_bottom = max(y, height - margin_bottom)
		draw.rectangle([
			(x, y_top),
			(x + bar_width, y_bottom)
		], fill='lightblue', outline='black')
		
		# Draw value on top of bar
		value_text = f'{value:.2f}'
		text_bbox = draw.textbbox((0, 0), value_text)
		text_width = text_bbox[2] - text_bbox[0]
		
		# Only draw text if we have valid coordinates
		if not pd.isna(y) and isinstance(y, (int, float)):
			draw.text((x + (bar_width - text_width) / 2, y - 20),
					 value_text, fill='black')
		
		# Draw label below x-axis
		draw.text((x + (bar_width - text_width) / 2,
				  height - margin_bottom + 10),
				 metric_name, fill='black')
	
	return image

def add_bar_graph_page(pdf, entry, data_entry):
    """Add a bar graph visualization page to the PDF report.

    Args:
        pdf (FPDF): The PDF document object to add the page to
        entry (dict): Dictionary containing ad entry data
        data_entry (dict): Dictionary containing metric data for visualization
    """
    pdf.add_page('L')
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 10, 'Metrics Overview', 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, entry['Ad name'], 0, 1, 'C')
    pdf.ln(10)
    
    # Prepare metrics for bar graph
    bar_metrics = {
        'CPM': float(data_entry.get('CPM (cost per 1,000 impressions) (USD)', 0)),
        'CTR': float(data_entry.get('CTR (link click-through rate)', 0)),
        'CPC': float(data_entry.get('CPC (cost per link click) (USD)', 0)),
        'Result Rate': float(data_entry.get('Result rate', 0))
    }
    
    # Generate and save temporary bar graph
    temp_graph_path = f'temp_bar_graph_{entry["Ad name"]}.png'
    bar_graph = create_bar_graph(bar_metrics)
    bar_graph.save(temp_graph_path)
    
    # Add graph to PDF
    pdf.image(temp_graph_path, x=10, y=70, w=270)
    
    # Clean up temporary file
    try:
        os.remove(temp_graph_path)
    except:
        pass

def create_final_summary_graphs(pdf, data):
    """Create and add summary graphs for overall campaign performance.

    Args:
        pdf (FPDF): The PDF document object to add the graphs to
        data (list): List of raw ad performance data
    """
    # Add a new page for summary graphs
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 10, 'Overall Campaign Performance Summary', 0, 1, 'C')
    pdf.ln(10)

    # Calculate averages for key metrics
    metrics = {
        'CPM': [],
        'CTR': [],
        'CPC': [],
        'Result Rate': [],
        'Leads': [],
        'Link Clicks': []
    }

    # Collect all metric values
    for entry in data:
        metrics['CPM'].append(float(entry.get('CPM (cost per 1,000 impressions) (USD)', 0)))
        metrics['CTR'].append(float(entry.get('CTR (link click-through rate)', 0)))
        metrics['CPC'].append(float(entry.get('CPC (cost per link click) (USD)', 0)))
        metrics['Result Rate'].append(float(entry.get('Result rate', 0)))
        metrics['Leads'].append(safe_convert_to_int(entry.get('Leads', 0)))
        metrics['Link Clicks'].append(safe_convert_to_int(entry.get('Link clicks', 0)))

    # Create graphs for each metric group
    metric_groups = [
        {'title': 'Cost Metrics', 'metrics': ['CPM', 'CPC']},
        {'title': 'Engagement Metrics', 'metrics': ['CTR', 'Result Rate']},
        {'title': 'Volume Metrics', 'metrics': ['Leads', 'Link Clicks']}
    ]

    y_position = 50
    for group in metric_groups:
        # Create metrics for this group
        group_metrics = {metric: sum(metrics[metric])/len(metrics[metric]) 
                        for metric in group['metrics']}

        # Generate and save temporary bar graph
        temp_graph_path = f'temp_summary_{group["title"]}.png'
        bar_graph = create_bar_graph(group_metrics, width=400, height=200)
        bar_graph.save(temp_graph_path)

        # Add title and graph to PDF
        pdf.set_font('Arial', 'B', 14)
        pdf.text(20, y_position-5, group['title'])
        pdf.image(temp_graph_path, x=10, y=y_position, w=135)

        # Clean up temporary file
        try:
            os.remove(temp_graph_path)
        except:
            pass

        y_position += 70

    # Add explanatory text
    pdf.set_font('Arial', '', 10)
    pdf.text(10, y_position + 10, 'Note: All values shown are campaign averages across all ads.')

def safe_convert_float(value, default=0.0):
    """Safely convert a value to float, handling NaN and other invalid values.

    Args:
        value: The value to convert
        default: Default value to return if conversion fails

    Returns:
        float: The converted float value or default
    """
    if pd.isna(value) or value == '-' or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_convert_int(value, default=0):
    """Safely convert a value to integer, handling NaN and other invalid values.

    Args:
        value: The value to convert
        default: Default value to return if conversion fails

    Returns:
        int: The converted integer value or default
    """
    if pd.isna(value) or value == '-' or value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def safe_convert_to_int(value, default=0):
    """Safely convert a value to integer, handling NaN and other invalid values.

    Args:
        value: The value to convert
        default: Default value to return if conversion fails

    Returns:
        int: The converted integer value or default
    """
    if pd.isna(value) or value == '-':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def add_continuation_summary(pdf, results, data):
    """Add a summary page with recommendations for continuing or discontinuing ads.

    Args:
        pdf (FPDF): The PDF document object to add the page to
        results (list): List of analysis results for each ad
        data (list): List of raw ad performance data
    """
    # Add a new page for continuation recommendations
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 10, 'Ad Continuation Recommendations', 0, 1, 'C')
    pdf.ln(10)

    # Calculate performance scores for each ad
    ad_scores = []
    for i, entry in enumerate(results):
        # Calculate composite score based on key metrics
        ctr = float(data[i].get('CTR (link click-through rate)', 0))
        cpc = float(data[i].get('CPC (cost per link click) (USD)', 0))
        result_rate = float(data[i].get('Result rate', 0))
        leads = safe_convert_to_int(data[i].get('Leads', 0))

        # Normalize CPC (lower is better)
        cpc_score = 1 / cpc if cpc > 0 else 0

        # Calculate weighted score
        score = (ctr * 0.3) + (cpc_score * 0.3) + (result_rate * 0.2) + (leads * 0.2)

        ad_scores.append({
            'ad_name': entry['Ad name'],
            'ad_set': entry['Ad Set Name'],
            'score': score,
            'metrics': {
                'CTR': ctr,
                'CPC': cpc,
                'Result Rate': result_rate,
                'Leads': leads
            }
        })

    # Sort ads by score
    ad_scores.sort(key=lambda x: x['score'], reverse=True)

    # Split into continue and discontinue groups
    median_score = ad_scores[len(ad_scores)//2]['score'] if ad_scores else 0
    continue_ads = [ad for ad in ad_scores if ad['score'] >= median_score]
    discontinue_ads = [ad for ad in ad_scores if ad['score'] < median_score]

    # Add Continue section
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 10, 'Recommended to Continue:', 0, 1)
    pdf.ln(5)

    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    for ad in continue_ads:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f"Ad: {ad['ad_name']}", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f"Ad Set: {ad['ad_set']}", 0, 1)
        pdf.cell(0, 6, f"Performance Score: {ad['score']:.2f}", 0, 1)
        pdf.cell(0, 6, f"Key Metrics: CTR: {ad['metrics']['CTR']:.2f}%, CPC: ${ad['metrics']['CPC']:.2f}, ", 0, 1)
        pdf.cell(0, 6, f"Result Rate: {ad['metrics']['Result Rate']:.2f}%, Leads: {ad['metrics']['Leads']}", 0, 1)
        pdf.ln(5)

    # Add Discontinue section
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 10, 'Recommended to Discontinue:', 0, 1)
    pdf.ln(5)

    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    for ad in discontinue_ads:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f"Ad: {ad['ad_name']}", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f"Ad Set: {ad['ad_set']}", 0, 1)
        pdf.cell(0, 6, f"Performance Score: {ad['score']:.2f}", 0, 1)
        pdf.cell(0, 6, f"Key Metrics: CTR: {ad['metrics']['CTR']:.2f}%, CPC: ${ad['metrics']['CPC']:.2f}, ", 0, 1)
        pdf.cell(0, 6, f"Result Rate: {ad['metrics']['Result Rate']:.2f}%, Leads: {ad['metrics']['Leads']}", 0, 1)
        pdf.ln(5)

    # Add explanation of scoring methodology
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Scoring Methodology:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, 'Performance scores are calculated using the following weights:\n- Click-Through Rate (CTR): 30%\n- Cost Per Click (CPC) Efficiency: 30%\n- Result Rate: 20%\n- Number of Leads: 20%\n\nAds scoring above the median are recommended to continue, while those below are recommended for discontinuation or significant optimization.')

def save_report_to_yaml(report_data: dict, yaml_file: str) -> None:
    """Save the report data to a YAML database file.

    Args:
        report_data (dict): Dictionary containing the report data
        yaml_file (str): Path to the YAML database file
    """
    try:
        # Load existing data if file exists
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as file:
                existing_data = yaml.safe_load(file) or {}
        else:
            existing_data = {}

        # Add timestamp to current report data
        timestamp = datetime.now().strftime('%Y-%m-%d')
        report_entry = {
            timestamp: {
                'metrics': report_data.get('metrics', {}),
                'performance_scores': report_data.get('performance_scores', {})
            }
        }

        # Merge with existing data
        existing_data.update(report_entry)

        # Save updated data back to file
        with open(yaml_file, 'w') as file:
            yaml.dump(existing_data, file, default_flow_style=False)

    except Exception as e:
        print(f'Error saving report to YAML database: {e}')

def analyze_historical_data(database_file: str) -> dict:
    try:
        with open(database_file, 'r') as file:
            data = yaml.safe_load(file) or {}

        if not data:
            return {'error': 'No historical data found'}

        historical_analysis = {
            'metrics_trends': {},
            'top_performers': {},
            'recommendations': [],
            'trend_analysis': {},  # Added missing trend_analysis dictionary
            'recurring_recommendations': {}
        }

        # Analyze metric trends over time
        for timestamp, report in data.items():
            for ad_name, metrics in report.get('metrics', {}).items():
                # Initialize trend analysis for new ads
                if ad_name not in historical_analysis['trend_analysis']:
                    historical_analysis['trend_analysis'][ad_name] = {
                        'CTR': {'values': []},
                        'CPC': {'values': []},
                        'Conversion_Rate': {'values': []}
                    }

                # Add metrics to trend analysis
                historical_analysis['trend_analysis'][ad_name]['CTR']['values'].append(metrics['CTR'])
                historical_analysis['trend_analysis'][ad_name]['CPC']['values'].append(metrics['CPC'])
                historical_analysis['trend_analysis'][ad_name]['Conversion_Rate']['values'].append(metrics['Result_Rate'])

        # Calculate trends for each metric
        for ad_name in historical_analysis['trend_analysis']:
            for metric in ['CTR', 'CPC', 'Conversion_Rate']:
                values = historical_analysis['trend_analysis'][ad_name][metric]['values']
                historical_analysis['trend_analysis'][ad_name][metric].update(calculate_trend(values))

        return historical_analysis

    except Exception as e:
        return {'error': f'Error analyzing historical data: {str(e)}',
                'trend_analysis': {}}  # Ensure trend_analysis exists even on error

def calculate_metric_trend(values: list) -> dict:
    if not values or len(values) < 2:
        return {'trend': 'insufficient_data'}

    first_value = values[0]
    last_value = values[-1]
    percent_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0

    return {
        'trend': 'increasing' if percent_change > 5 else 'decreasing' if percent_change < -5 else 'stable',
        'percent_change': percent_change
    }

def identify_seasonal_patterns(trends: dict) -> dict:
    timestamps = trends['timestamps']
    metrics = trends['ctr']  # Using CTR as primary metric for seasonal analysis

    if len(timestamps) < 12:  # Need at least a year of data
        return {'exists': False}

    # Convert timestamps to months and aggregate data
    monthly_data = {}
    for i, timestamp in enumerate(timestamps):
        month = datetime.strptime(timestamp, '%Y-%m-%d').month
        if month not in monthly_data:
            monthly_data[month] = []
        monthly_data[month].append(metrics[i])

    # Calculate monthly averages
    monthly_averages = {month: sum(values)/len(values) for month, values in monthly_data.items()}

    # Identify peak periods
    avg = sum(monthly_averages.values()) / len(monthly_averages)
    peak_months = [month for month, value in monthly_averages.items() if value > avg * 1.1]

    if peak_months:
        return {
            'exists': True,
            'pattern_type': 'seasonal',
            'peak_periods': ', '.join([calendar.month_name[m] for m in peak_months])
        }

    return {'exists': False}

def calculate_roi(metrics: dict) -> float:
    total_cost = metrics['CPC'] * metrics['Link_Clicks']
    total_value = metrics['Leads'] * 100  # Assuming average lead value of $100
    return (total_value - total_cost) / total_cost if total_cost > 0 else 0

def calculate_trend(values: list) -> dict:
    """Calculate trend information for a series of values.

    Args:
        values (list): List of numerical values

    Returns:
        dict: Dictionary containing trend analysis
    """
    if not values or len(values) < 2:
        return {'trend': 'insufficient_data'}

    first_value = values[0]
    last_value = values[-1]
    percent_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0

    return {
        'trend': 'increasing' if percent_change > 5 else 'decreasing' if percent_change < -5 else 'stable',
        'percent_change': percent_change,
        'first_value': first_value,
        'last_value': last_value,
        'average': sum(values) / len(values)
    }

def add_historical_analysis_page(pdf: FPDF, database_file: str) -> None:
    """Add a page analyzing historical performance data to the PDF report.

    Args:
        pdf (FPDF): The PDF document object to add the page to
        database_file (str): Path to the YAML database file
    """
    # Get historical analysis
    analysis = analyze_historical_data(database_file)

    if 'error' in analysis:
        return

    # Add new page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 10, 'Historical Performance Analysis', 0, 1, 'C')
    pdf.ln(10)

    # Add Top Performers section
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Top Performing Ads', 0, 1)
    pdf.ln(5)

    pdf.set_font('Arial', '', 12)
    for ad_name, score in analysis['top_performers'].items():
        pdf.cell(0, 8, f'{ad_name}: {score:.2f}', 0, 1)

    # Add Trend Analysis section
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Performance Trends', 0, 1)

    pdf.set_font('Arial', '', 12)
    for ad_name, trends in analysis['trend_analysis'].items():
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, ad_name, 0, 1)
        pdf.set_font('Arial', '', 12)
        for metric, trend_data in trends.items():
            if trend_data['trend'] != 'insufficient_data':
                pdf.cell(0, 6, 
                    f"{metric}: {trend_data['trend']} ({trend_data['percent_change']:.1f}% change)", 
                    0, 1)

    # Add Recurring Recommendations section with expanded insights
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Performance Insights & Recommendations', 0, 1)

    pdf.set_font('Arial', '', 12)
    # Add general performance insights
    pdf.multi_cell(0, 8, 'Overall Campaign Performance:', 0, 'L')
    pdf.ln(5)

    # Add specific recommendations based on historical data
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, 'Strategic Recommendations:', 0, 1)
    pdf.set_font('Arial', '', 12)

    # Budget Optimization
    pdf.multi_cell(0, 8, '1. Budget Allocation:\n- Redistribute budget from underperforming ads to top performers\n- Consider increasing investment in ads showing consistent growth\n- Implement dayparting based on historical performance patterns', 0, 'L')
    pdf.ln(5)

    # Creative Optimization
    pdf.multi_cell(0, 8, '2. Creative Strategy:\n- Analyze creative elements of top-performing ads\n- Test new variations based on successful patterns\n- Remove or revise consistently underperforming creatives', 0, 'L')
    pdf.ln(5)

    # Audience Insights
    pdf.multi_cell(0, 8, '3. Audience Targeting:\n- Refine audience targeting based on top performer demographics\n- Expand reach for ads with strong engagement metrics\n- Consider creating lookalike audiences from high-converting segments', 0, 'L')
    pdf.ln(5)

    # Performance Optimization
    pdf.multi_cell(0, 8, '4. Performance Optimization:\n- Adjust bid strategies based on historical CPC trends\n- Optimize ad scheduling around peak performance times\n- Consider placement modifications based on performance data', 0, 'L')
    pdf.ln(5)

    # Testing Strategy
    pdf.multi_cell(0, 8, '5. Testing Recommendations:\n- Implement A/B tests on successful ad elements\n- Test new ad formats based on platform trends\n- Experiment with different call-to-action variations', 0, 'L')
    pdf.ln(5)

    # Add specific action items
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, 'Immediate Action Items:', 0, 1)
    pdf.set_font('Arial', '', 12)
    for recommendation, count in analysis['recurring_recommendations'].items():
        pdf.multi_cell(0, 8, f'• {recommendation} (suggested {count} times)', 0, 'L')

    # Add future outlook section
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, 'Future Outlook:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, 'Based on historical trends, consider:\n- Seasonal adjustments to campaign strategy\n- Platform-specific optimization opportunities\n- Emerging audience targeting options\n- New ad format opportunities', 0, 'L')

def generate_pdf_report(results: list, data: list, output_file: str, organization: str) -> None:
    """Generate a comprehensive PDF report of ad performance analysis.

    Args:
        results: List of KPI analysis results.
        data: List of raw ad data.
        output_file: Path where the PDF report should be saved.
        organization: Name of the organization for the report.
    """
    pdf = FPDF()
    temp_files = []
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 40, 'Fractal Facebook Funnels', 0, 1, 'C')
    
    pdf.set_font('Arial', '', 16)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(0, 10, f'Generated on: {current_time}', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.cell(0, 10, f'Copyright © {datetime.now().strftime("%Y")} {organization}. All Rights Reserved.', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_font('Arial', 'B', 12)
    pdf.multi_cell(0, 10, f'PRIVACY NOTICE:\nThis document is confidential and may only be viewed by {organization}. Unauthorized access, distribution, or reproduction is strictly prohibited.', 0, 'C')
    
    generate_ranking_page(pdf, results, data)
    add_continuation_summary(pdf, results, data)
    add_historical_analysis_page(pdf, 'reports_database.yaml')
    create_final_summary_graphs(pdf, data)

    column_width = 120
    row_height = 10
    
    for i, entry in enumerate(results):
        pdf.add_page('L')

        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, row_height, "Key Performance Indicators", 0, 1, 'C')

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, row_height, entry['Ad name'], 0, 1, 'C')
        pdf.ln(row_height)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(column_width, row_height, 'Metric', 1)
        pdf.cell(20, row_height, 'Value', 1)
        pdf.cell(column_width, row_height, 'Status Explanation', 1)
        pdf.ln()
        
        for metric, details in entry.items():
            if metric not in ['Ad name', 'Ad Set Name', 'Ad set budget', 'Ad set budget type']:
                pdf.set_font('Arial', '', 12)
                
                if details['within_range']:
                    pdf.set_text_color(0, 128, 0)
                else:
                    pdf.set_text_color(255, 0, 0)
                
                pdf.cell(column_width, row_height, str(metric), 1)
                pdf.cell(20, row_height, "{:.2f}".format(details['value']), 1)
                pdf.cell(column_width, row_height, details['explanation'], 1)
                pdf.ln()
                
                pdf.set_text_color(0, 0, 0)
        
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, row_height, f'Funnel Diagram', 0, 1, 'C')

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, row_height, f'{entry["Ad name"]} - {entry["Ad Set Name"]}', 0, 1, 'C')
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, row_height, f'Ad Set Budget: ${entry["Ad set budget"]} / {entry["Ad set budget type"]}', 0, 1, 'C')
        pdf.ln(row_height)
        
        funnel_metrics = {
            'CPM ($)': float(data[i].get('CPM (cost per 1,000 impressions) (USD)', 0)),
            'CTR (%)': float(data[i].get('CTR (link click-through rate)', 0)),
            'Link Clicks (#)': safe_convert_int(data[i].get('Link clicks', 0)),
            'CPC ($)': float(data[i].get('CPC (cost per link click) (USD)', 0)),
            'Results (%)': float(data[i].get('Result rate', 0)),
            'Leads (#)': safe_convert_int(data[i].get('Leads', 0))
        }
        
        temp_image_path = f'temp_funnel_{i}.png'
        temp_files.append(temp_image_path)
        
        funnel_image = create_funnel_image(funnel_metrics, results)
        funnel_image.save(temp_image_path)
        
        pdf.image(temp_image_path, x=10, y=70, w=190)
        
        # Add the new analysis page
        add_bar_graph_page(pdf, entry, data[i])
        add_analysis_page(pdf, entry, data[i])

    pdf.output(output_file)
    
    # Clean up temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass

    # Collect report data for YAML database
    report_data = {
        'organization': organization,
        'metrics': {},
        'recommendations': [],
        'performance_scores': {}
    }

    # Collect metrics and scores for each ad
    for i, entry in enumerate(results):
        ad_name = entry['Ad name']
        report_data['metrics'][ad_name] = {
            'CPM': float(data[i].get('CPM (cost per 1,000 impressions) (USD)', 0)),
            'CTR': float(data[i].get('CTR (link click-through rate)', 0)),
            'CPC': float(data[i].get('CPC (cost per link click) (USD)', 0)),
            'Result_Rate': float(data[i].get('Result rate', 0)),
            'Leads': safe_convert_int(data[i].get('Leads', 0)),
            'Link_Clicks': safe_convert_int(data[i].get('Link clicks', 0))
        }

        # Calculate performance score
        ctr = float(data[i].get('CTR (link click-through rate)', 0))
        cpc = float(data[i].get('CPC (cost per link click) (USD)', 0))
        result_rate = float(data[i].get('Result rate', 0))
        leads = safe_convert_int(data[i].get('Leads', 0))

        cpc_score = 1 / cpc if cpc > 0 else 0
        score = (ctr * 0.3) + (cpc_score * 0.3) + (result_rate * 0.2) + (leads * 0.2)
        report_data['performance_scores'][ad_name] = score

        # Collect recommendations from analysis
        analysis = analyze_metrics(entry, data[i])
        report_data['recommendations'].extend(analysis['recommendations'])

    # Save report data to YAML database
    save_report_to_yaml(report_data, 'reports_database.yaml')

def generate_sphinx_docs(project_dir: str, output_dir: str = 'docs') -> None:
    """Generate Sphinx documentation for the project.

    Args:
        project_dir (str): Path to the project directory containing Python files
        output_dir (str): Directory where Sphinx documentation will be generated
    """
    # Create docs directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Configure Sphinx quickstart options
    options = {
        'path': output_dir,
        'project': 'FractalFB',
        'author': 'Project Maintainers',
        'version': '0.3.0',
        'release': '0.3.0',
        'language': 'en',
        'suffix': '.rst',
        'master': 'index',
        'makefile': True,
        'batchfile': True,
        'sep': False,
        'dot': '_',
        'extensions': [
            'sphinx.ext.autodoc',
            'sphinx.ext.napoleon',
            'sphinx.ext.viewcode'
        ]
    }

    # Generate Sphinx documentation structure
    generate(options)

    # Create conf.py with necessary extensions
    conf_content = f"""import os
import sys
sys.path.insert(0, os.path.abspath('../'))

project = '{options["project"]}'
copyright = '2024, {options["author"]}'
author = '{options["author"]}'

extensions = {options["extensions"]}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']"""

    with open(os.path.join(output_dir, 'conf.py'), 'w') as f:
        f.write(conf_content)

    # Create index.rst
    index_content = """Welcome to FractalFB's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""

    with open(os.path.join(output_dir, 'index.rst'), 'w') as f:
        f.write(index_content)

    # Create api.rst to document all modules
    api_content = """API Documentation
===============

.. automodule:: app
   :members:
   :undoc-members:
   :show-inheritance:
"""

    with open(os.path.join(output_dir, 'api.rst'), 'w') as f:
        f.write(api_content)

    print(f'Sphinx documentation structure generated in {output_dir}')
    print('Run `make html` in the docs directory to build the documentation')



def main():
    """Main entry point for the FractalFB tool."""
    parser = argparse.ArgumentParser(description='Facebook Ad Analysis Tool')
    parser.add_argument('--csv', default='facebook_ads.csv', help='Path to Facebook ads CSV file')
    parser.add_argument('--yaml', default='settings.yaml', help='Path to settings YAML file')
    parser.add_argument('--output', default='kpi_report.pdf', help='Output PDF file path')
    parser.add_argument('--database', default='reports_database.yaml', help='Path to YAML database file')
    parser.add_argument('--generate-docs', action='store_true', help='Generate Sphinx documentation')
    parser.add_argument('--docs-dir', default='docs', help='Directory for Sphinx documentation')
    args = parser.parse_args()

    if args.generate_docs:
        generate_sphinx_docs(os.getcwd(), args.docs_dir)

    data = read_facebook_data(args.csv)
    settings = read_kpi_settings(args.yaml)

    if data and settings:
        organization = settings.get('organization', 'All-Weather Seal of West Michigan')
        results = check_kpi_ranges(data, settings)
        generate_pdf_report(results, data, args.output, organization)
        print(f'Report generated successfully: {args.output}')
        print(f'Report data saved to database: {args.database}')

if __name__ == '__main__':
    main()
