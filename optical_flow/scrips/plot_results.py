import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------
# Placeholder data — replace with your values
# -----------------------------------------

# Example resolution labels (adjust as needed)
labels = ["480p", "720p", "1080p", "4K"]

# Placeholder performance values
cpu_values = [1071.56, 2413.06, 5434.33, 9679.46]   # replace with real CPU numbers
gpu_values = [4.34, 9.02, 19.72, 36.64]   # replace with real GPU numbers

# -----------------------------------------
# Plotting
# -----------------------------------------
x = np.arange(len(labels))          # positions on x-axis
width = 0.35                        # width of the bars

fig, ax = plt.subplots(figsize=(8, 5))

cpu_bar = ax.bar(x - width/2, cpu_values, width, label='CPU')
gpu_bar = ax.bar(x + width/2, gpu_values, width, label='GPU')

# Add labels & title
ax.set_ylabel('Time (ms)')          # or FPS, throughput, etc.
ax.set_title('CPU vs GPU Performance')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels on bars
for bar in cpu_bar + gpu_bar:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Add a line plot visualization for comparison
plt.figure(figsize=(8, 5))
plt.plot(labels, cpu_values, marker='o', label='CPU')
plt.plot(labels, gpu_values, marker='o', label='GPU')
plt.ylabel('Average Time per Frame (ms)')
plt.xlabel('Resolution')
plt.title('CPU vs GPU Performance')
plt.legend()
plt.grid(True)

# Add value labels at each point
for i, (label, cpu, gpu) in enumerate(zip(labels, cpu_values, gpu_values)):
    plt.text(label, cpu, f'{cpu:.2f}', ha='center', va='bottom', color='black')
    plt.text(label, gpu, f'{gpu:.2f}', ha='center', va='bottom', color='black')

plt.tight_layout()
plt.show()