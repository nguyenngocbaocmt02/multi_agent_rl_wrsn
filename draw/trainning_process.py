import pandas as pd
import matplotlib.pyplot as plt
import sys
if __name__ == "__main__":
    file_name = str(sys.argv[1])
    # Read the file into a pandas DataFrame
    data = pd.read_csv(file_name)  # Rfeplace 'your_file.csv' with the actual file name

    # Determine the number of columns and rows for subplots
    num_columns = len(data.columns)
    num_rows = 1

    # Calculate the appropriate number of rows and columns for subplots
    if num_columns > 3:
        num_rows = num_columns // 3
        if num_columns % 3 != 0:
            num_rows += 1
        num_columns = 3

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5))

    # Flatten the axs array if necessary
    if num_columns == 1 and num_rows == 1:
        axs = [axs]

    # Iterate over each column and plot the line graph
    for i, column in enumerate(data.columns):
        row = i // num_columns
        col = i % num_columns

        # Select the current axis
        ax = axs[row][col]

        # Plot the line graph for the current column
        ax.plot(data[column])
        ax.set_title(column)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')

    # Remove any empty subplots
    if i + 1 < num_rows * num_columns:
        for j in range(i + 1, num_rows * num_columns):
            axs.flat[j].set_visible(False)

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Display the figure with all the line graphs
    plt.savefig("tmp.png")
