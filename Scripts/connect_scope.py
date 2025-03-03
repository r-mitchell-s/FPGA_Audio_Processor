# scope_connection.py
import pyvisa
import time

# Connect to the resource manager
rm = pyvisa.ResourceManager()

# List all available resources
resources = rm.list_resources()
print("Available resources:")
for i, resource in enumerate(resources):
    print(f"{i+1}. {resource}")

# If no resources found, exit
if not resources:
    print("No resources found. Check your connection and drivers.")
    exit()

# If resources are found, try to connect
# If multiple resources, let user choose which one
if len(resources) > 1:
    choice = int(input("Enter the number of the resource to connect to: ")) - 1
    resource = resources[choice]
else:
    resource = resources[0]

try:
    # Connect to the oscilloscope
    print(f"\nConnecting to: {resource}")
    scope = rm.open_resource(resource)
    
    # Set timeout (ms)
    scope.timeout = 5000
    
    # Query device identity
    idn = scope.query("*IDN?")
    print(f"Successfully connected to: {idn}")
    
    # Test a simple command
    print("Testing scope communication...")
    scope.write("*CLS")  # Clear status
    scope.write("AUTOSET EXECUTE")  # Run autoset
    print("Autoset command sent successfully")
    
    # Close the connection
    scope.close()
    print("Connection closed properly")
    print("\nConnection test successful! Your scope is ready to use.")

except Exception as e:
    print(f"Error communicating with the scope: {e}")
    