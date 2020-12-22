# %matplotlib inline
# !pip install mne
import matplotlib.pyplot as plt
import numpy as np 
import random
import mne
print("Hello World!")

a = 1
b = 2


x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot
