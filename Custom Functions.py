#using loops to iterate through nested dictionary/list python
def nested_loop_dict(obj):
    # Iterate over all key-value pairs of dict argument
    for key, value in obj.items():
        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for pair in  nested_loop_dict(value):
                yield (key, *pair)
        else:
            # If value is not dict type then yield the value
            yield (key, value)
#Loop through all key-value pairs of a nested dictionary
print('Iterating over Nested Dict:')
for pair in nested_loop_dict(details):
    print('\n',pair)
