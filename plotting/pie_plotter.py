import matplotlib.pyplot as plt

# 1. Setup Data
categories = [
    'Public Safety & Emergency Response',
    'Education',
    'Healthcare & Bioethics',
    'Personal & Interpersonal Relationships',
    'Legal & Compliance',
    'Public Policy & Governance',
    'Community & Social Services',
    'Environmental & Conservation',
    'Workplace & Business',
    'Customer Service',
    'Technology & Innovation',
    'Sports & Recreation',
    'Cultural & Religious',
    'Media & Journalism',
    'Other',
]

counts = [41, 8, 14, 36, 20, 3, 16, 5, 48, 4, 1, 4, 39, 1, 10]

# 2. Define Pastel Colors
# A mix of soft blues, reds, yellows, greens, and purples similar to the reference
colors = [
    '#6fa8dc', # Light Blue
    '#e06666', # Soft Red
    '#f6b26b', # Soft Orange
    '#93c47d', # Muted Green
    '#ffd966', # Pastel Yellow
    '#8e7cc3', # Soft Purple
    '#76a5af', # Teal
    '#a64d79', # Magenta shade
    '#b6d7a8', # Pale Green
    '#ea9999', # Pale Red
    '#ffe599', # Pale Yellow
    '#9fc5e8', # Pale Blue
    '#f9cb9c', # Peach
    '#d5a6bd', # Pinkish
    '#b4a7d6'  # Lavender
]

# 3. Create the Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create the pie chart with a hole (donut)
wedges, texts, autotexts = ax.pie(
    counts,
    colors=colors,
    autopct=lambda p: f'{p:.1f}%' if p > 2 else '', # Only show % if slice is > 2% to avoid clutter
    startangle=90,
    counterclock=False,
    pctdistance=0.85, # Position of the percentage text
    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1) # Width creates the donut hole
)

# Style the percentage text
plt.setp(autotexts, size=16, weight="bold", color="black")

# 4. Add "VLAF" to the Center
ax.text(0, 0, 'VLAF', ha='center', va='center', fontsize=28, fontweight='bold')

# 5. Create the Legend
# Positioned to the right as per the reference image
ax.legend(
    wedges, 
    categories,
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    frameon=False, # Remove box around legend for a cleaner look
    fontsize=15
)

# Final Layout Adjustments
plt.tight_layout()
plt.axis('equal') # Ensures the chart is a perfect circle
plt.savefig('plotting/data_diversity.pdf', dpi=400, bbox_inches='tight')