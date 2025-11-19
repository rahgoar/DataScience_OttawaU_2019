
install.packages("reticulate")

library(reticulate)

py_install("pandas")

# Import a Python module, e.g., pandas
pd <- import("pandas")

# Import a specific function from a module
#np_array <- import("numpy")$array

# Import the Python 'os' module
os <- import("os")

# Call a function from the 'os' module
current_directory <- os$listdir(".")
print(current_directory)

# Import the 'math' module and use a function
math <- import("math")
result <- math$sqrt(16)
print(result)

# Create a simple Python script (e.g., "my_script.py")
# In "my_script.py":
# def greet(name):
#     return f"Hello, {name} from Python!"
#
# my_variable = "Python variable"

# Source the Python script
source_python("my_script.py")
