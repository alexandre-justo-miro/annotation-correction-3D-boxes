import matplotlib.pyplot as plt
import numpy as np

# Original values in seconds, convert to minutes
loading_seconds = [687.2136549949646, 1257.2108500003815, 634.7431638240814, 647.5445718765259, 641.8494741916656, 630.3314499855042, 657.3763649463654, 644.8884620666504, 639.9737448692322, 656.9002161026001]
optimizing_seconds = [6709.225378036499, 4981.204298019409, 8026.4322299957275, 4804.307642221451, 11045.593837738037, 128.13175320625305, 5338.726879119873, 30373.077843904495, 11124.0715072155, 22474.585848093033]

# Convert to minutes
loading = [t / 60 for t in loading_seconds]
optimizing = [t / 60 for t in optimizing_seconds]
total = [l + o for l, o in zip(loading, optimizing)]

# Original total values in seconds (converted to minutes for comparison)
total_original_seconds = [7396.439033031464, 6238.415148019791, 8661.175393819809, 5451.852214097977, 11687.443311929703, 758.4632031917572, 5996.103244066238, 31017.966305971146, 11764.045252084732, 23131.486064195633]
total_original = [t / 60 for t in total_original_seconds]

# Assert that calculated totals match original totals (within floating point tolerance)
for i, (calc, orig) in enumerate(zip(total, total_original)):
    assert abs(calc - orig) < 0.001, f"Bar {i+1}: Calculated total {calc:.4f} doesn't match original {orig:.4f}"
print("✓ All assertions passed - totals are correct!")

# Create bar plot
x = np.arange(10)
width = 0.6

fig, ax = plt.subplots(figsize=(12, 6))

# Create stacked bars
bar1 = ax.bar(x, loading, width, label='Loading', color='#3498db')
bar2 = ax.bar(x, optimizing, width, bottom=loading, label='Optimizing', color='#e74c3c')

# Plot original total as line to verify it matches stacked bars
ax.plot(x, total_original, 'ko-', linewidth=2, markersize=8, label='Total (Original)', zorder=10)

# Customize plot
total_time_minutes = sum(total)
avg_time_minutes = total_time_minutes / len(total)
total_time_hours = total_time_minutes / 60
avg_time_hours = avg_time_minutes / 60
ax.set_xlabel('Sequence Number', fontsize=12)
ax.set_ylabel('Time (minutes)', fontsize=12)
ax.set_title(f'Correction of 3D annotated boxes\nMAN TruckScenes mini dataset\nLoading and Optimizing Time per Sequence\nTotal: {total_time_hours:.2f} hours | Average: {avg_time_hours:.2f} hours', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{i+1}' for i in range(10)])
ax.yaxis.set_major_locator(plt.MultipleLocator(50))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
ax.legend()
ax.grid(axis='y', alpha=0.3, which='major')
ax.grid(axis='y', alpha=0.15, which='minor', linestyle=':')

plt.tight_layout()
plt.show()
