import os
import tkinter as tk
from tkinter import filedialog
import scipy.io

os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Create a root window
root = tk.Tk()

# Hide the root window
root.withdraw()

#ask the user to select a file path the contains the .mat files 
#then ask the user to provide a string to be utilized to query the .mat files and extract the data
file_path = filedialog.askdirectory()
query = input("Please enter a string to query the .mat files: ")

#print the file path and the query string of those file that match the query string
print("The file path is: " + file_path)
print("The query string is: " + query)

#loop through the files in the file path that match the query string
for file in os.listdir(file_path):
    #the file name should have the query string in it and if it does store in a csv file called the query string as the file name
    # then store the path to the file in the csv file in a column called 'path2file'
    if query in file:
        #load the .mat file into a dictionary
        mat = scipy.io.loadmat(file_path + '/' + file)
        
            #find keys that are not '__header__', '__version__', '__globals__' and store them in a list called 'keys'
        keys = [key for key in mat.keys() if key not in ['__header__', '__version__', '__globals__']]#print the keys

        #now create a dictionary dynamically that will store the file name as the key name the the variable mat[key] as the value
        for key in keys:
            #create a dictionary dynamically where the file name without the .mat extension is the key and the value is the variable mat[key]
            dict = {file[:-4]: mat[key]} #remove the .mat extension from the file name 
            
    #if not than say no files match the query string
    if not file:
        print("No files match the query string")
        
#store the dictionary as pickle in the provided file path for later use outside of this script
import pickle
pickle.dump(dict, open(file_path + '/' + query + '.p', 'wb'))



# Close the root window 
root.quit()
root.destroy() 
