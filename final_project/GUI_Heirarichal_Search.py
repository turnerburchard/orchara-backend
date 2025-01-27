import tkinter as tk
from tkinter import ttk

from cluster import Clusterer
from embed import Embedder
from pickle_helpers import load_from_pkl
from search import find_target_cluster

#Data Processing 

filename = 'final_project/Data/50k/data_50k.pkl'
all_papers = load_from_pkl(filename)
data = [paper.abstract_vector for paper in all_papers]
clusterer = Clusterer(data)
transform = clusterer.reduce_dimensions(273)
linkage_matrix = load_from_pkl('final_project/Data/50k/280/agg50k').linkage_matrix

# agg_clusters, linkage_matrix = clusterer.agglomerative()
embedder = Embedder()

# Example function to process inputs
def process_inputs(text, num):
    search_vector = embedder.embed_text(text)
    cluster_indices = find_target_cluster(linkage_matrix, search_vector, num, clusterer.data, transform)
    cluster_titles = [all_papers[i].title for i in cluster_indices]
    return cluster_titles

# Button click handler
def on_button_click():
    try:
        text_input = text_widget.get("1.0", tk.END).strip()
        num_input = int(num_var.get())
        result_list = process_inputs(text_input, num_input)
        result_list = ["\u2022"+result for result in result_list]
        result = "\n\n".join(result_list)

        output_text.config(state=tk.NORMAL)  
        output_text.delete(1.0, tk.END)  
        output_text.insert(tk.END, result) 
        output_text.config(state=tk.DISABLED)  
    except ValueError:
        output_text.config(state=tk.NORMAL)  
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, "Invalid input! Please enter an integer in the number field.")
        output_text.config(state=tk.DISABLED) 

#Data Processing 


# Create the main application window
root = tk.Tk()
root.title("Heirarichal Cluster Search")
root.geometry("1400x1000")
root.configure(bg="#f0f0f0")  # Light gray background

# Create StringVars to manage the input/output field values
text_var = tk.StringVar()
num_var = tk.StringVar()

# Create a frame for input fields
input_frame = ttk.Frame(root, padding="10 10 10 10")
input_frame.pack(fill="x", pady=10)

# intro_text = """
# Enter your own abstract, and a desired amount of papers, and Heirarichal Cluster Search will find
# the cluster closest in size to the given amount, which is closest to your abstract.
# """
# # Text Input
# ttk.Label(input_frame, text=intro_text, anchor="w")

# ttk.Label(input_frame, text="Query:", anchor="w").grid(row=0, column=0, sticky="w", padx=5, pady=5)
# text_entry = ttk.Entry(input_frame, textvariable=text_var, width=30)
# text_entry.grid(row=0, column=1, padx=5, pady=5)
ttk.Label(input_frame, text="Query:", anchor="w").grid(row=0, column=0, sticky="nw", padx=5, pady=5)
# Text Widget for multi-line input
text_widget = tk.Text(input_frame, wrap=tk.WORD, height=10, width=60)
text_widget.grid(row=0, column=1, padx=5, pady=5)

# Integer Input
ttk.Label(input_frame, text="Cluster Size:", anchor="w").grid(row=1, column=0, sticky="w", padx=5, pady=5)
num_entry = ttk.Entry(input_frame, textvariable=num_var, width=30)
num_entry.grid(row=1, column=1, padx=5, pady=5)

# Add the process button
process_button = ttk.Button(root, text="Process", command=on_button_click)
process_button.pack(pady=10)

# Create a frame for the output (scrollable area)
output_frame = ttk.Frame(root, padding="10 10 10 10", relief="groove")
output_frame.pack(fill="x", pady=10)

ttk.Label(output_frame, text="Output:", anchor="w").pack(anchor="w", padx=5)

# Add a Scrollbar
scrollbar = tk.Scrollbar(output_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create a Text widget for output, with a scrollbar
output_text = tk.Text(output_frame, wrap=tk.WORD, height=500, width=40, background="white")
output_text.pack(fill="x", padx=5, pady=5)

# Attach the scrollbar to the Text widget
output_text.config(yscrollcommand=scrollbar.set)
output_text.config(state=tk.DISABLED)
scrollbar.config(command=output_text.yview)


# Start the main event loop
root.mainloop()