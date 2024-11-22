import matplotlib.pyplot as plt

# Data
document_set_size = [1, 5, 10, 20]
traditional_rag_ingestion_time = [0.28, 1.34, 3.034, 4.89]
e_rag_ingestion_time = [42.9, 470.36, 1156.7, 2734]

# Create the area plot
plt.figure(figsize=(10, 6))

# Plot areas with some transparency
plt.fill_between(document_set_size, traditional_rag_ingestion_time, alpha=0.3, color='blue', label='Traditional RAG')
plt.fill_between(document_set_size, e_rag_ingestion_time, alpha=0.3, color='orange', label='E-RAG')

# Add lines on top for better visibility
plt.plot(document_set_size, traditional_rag_ingestion_time, color='blue', marker='o', linestyle='-', linewidth=2)
plt.plot(document_set_size, e_rag_ingestion_time, color='orange', marker='o', linestyle='-', linewidth=2)

# Title and labels
plt.title('Ingestion Time Comparison: Traditional RAG vs. E-RAG')
plt.xlabel('Document Set Size (No. of Docs)')
plt.ylabel('Ingestion Time (sec)')
plt.yscale('log')

# Set specific x-axis ticks
plt.xticks(document_set_size, document_set_size)

# Format y-axis with specific values
y_ticks = [0.1, 1, 10, 100, 1000, 3000]
plt.yticks(y_ticks, y_ticks)

# Cleaner grid with only major lines
plt.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.2)
plt.legend()

# Instead of plt.show()
plt.savefig('ingestion_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()