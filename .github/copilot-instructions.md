# Project Guidelines

## Code Style

- Use PEP 8 for Python code.
- Add type hints to every new or modified function definition, including parameter types and return types.
- All variables should have a type annotation, even if the type can be inferred.
 - Write docstrings for all functions, including a description of the function's purpose, its parameters, and its return value. Use the following format for docstrings:

```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """
    Description of the function's purpose.
    Parameters:
    param1 (Type): Description of param1.
    param2 (Type): Description of param2.
    Returns:
    ReturnType: Description of the return value.
    """
    # Function implementation
```
- Use descriptive variable names that clearly indicate their purpose. Include the unit of measurement in the variable name if applicable (e.g., `waste_kg` for waste in kilograms).
- Avoid creating new variables if an existing variable can be reused without causing confusion. If a new variable is necessary, ensure it has a clear and descriptive name.
- Break long lines of code into multiple lines to improve readability, especially when dealing with complex expressions or function calls. Use parentheses to indicate that the line continues on the next line.
- Use smaller functions that perform a single task. If a function is too long or performs multiple tasks, consider breaking it into smaller, more focused functions.
- Use "_" as a prefix for private methods to indicate that they are intended for internal use within the class or module.

## Architecture

- ABM_CE_PV_Model.py is the main file that contains the core logic of the model. It initilaizes and instantiates all the agents.
- Each individual agent is defined in a separate file as a separate class. For example, ABM_CE_PV_ConsumerAgents.py contains the definition of the ConsumerAgent class.
- The file to run the model is ABM_CE_PV_MultipleRun.py, which imports the model and calls its run method.
- If any data input is reused across multiple agents, it needs to be loaded in the model level and passed to the agents as needed. This avoids redundant data loading and ensures consistency across agents. If an agent needs to load its own data, it should do so in its own file, but only if the data is specific to that agent and not shared with others.
